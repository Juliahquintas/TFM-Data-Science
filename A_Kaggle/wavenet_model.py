"""
flexible_wavenet_experiment.py  —  Arquitectura Avanzada Flexible para detección de Parkinson
Colocar en: src/experiments/flexible_wavenet_experiment.py
Ejecutar  : python flexible_wavenet_experiment.py

Incluye de forma parametrizable:
- Convoluciones Deformables 1D
- Padding Causal (Estilo WaveNet) vs Circular (Estilo CDIL)
- Gated Activation Units (Tanh * Sigmoid) vs ReLU convencional
- Arquitectura Dual con Skip Connections acumulativas y Conexiones Residuales
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from pathlib import Path

# Importar utilidades compartidas (mismo directorio)
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics_utils import (
    seed_everything, split_train_val_by_subject,
    AudioDataset, train_fold, evaluate_fold,
    plot_learning_curves, plot_mean_roc,
    plot_accumulated_cm, print_summary, save_results,
)

# ======================================================================
#  CONFIGURACIÓN  —  Parámetros de control del Experimento
# ======================================================================
DATASET              = 'pc-gita'   # 'pc-gita' o 'neurovoz'
NHID                 = 32
KERNEL_SIZE          = 3
DROPOUT              = 0.0        # 0.0 para replicar exactamente a Marta Rey

# --- INTERRUPTORES DE INNOVACIÓN ---
CAUSAL               = False        # True: Causal (WaveNet) | False: Circular (CDIL)
GATED                = False        # True: Tanh * Sigmoid | False: ReLU Convencional
DEFORMABLE           = True        # True: Activa offsets deformables | False: Convolución rígida

BATCH_SIZE           = 32
N_EPOCHS             = 200
LR                   = 1e-3
WEIGHT_DECAY         = 1e-3
EARLY_STOP_PATIENCE  = 30
VAL_RATIO            = 0.10
SEED                 = 42
K_FOLDS              = 5

# Nombre dinámico del modelo para guardar los gráficos en carpetas separadas
MODEL_NAME           = f"Flexible_C{int(CAUSAL)}_G{int(GATED)}_D{int(DEFORMABLE)}"

# ======================================================================
#  RUTAS
# ======================================================================
PROJECT_ROOT  = Path(__file__).resolve().parent
while not (PROJECT_ROOT / 'data').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
SPLITS_FILE   = PROJECT_ROOT / 'data' / 'data_splits.json'
RESULTS_DIR   = PROJECT_ROOT / 'results' / f'{DATASET}_{MODEL_NAME}'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
#  COMPONENTES DEL MODELO FLEXIBLE
# ======================================================================

class DeformableConv1d(nn.Module):
    """
    Implementación adaptada de Convolución Deformable para señales 1D.
    Utiliza el operador 2D de torchvision proyectando la señal temporal en el eje de la altura.
    """
    def __init__(self, in_channels: int, out_channels: int, ks: int, padding: int, dilation: int):
        super().__init__()
        self.padding = (padding, 0)
        self.dilation = (dilation, 1)
        self.ks = (ks, 1)

        # Capa para predecir los desplazamientos (offsets) en el eje del tiempo
        # Cada punto del kernel necesita 2 offsets en 2D (alto, ancho). Dejamos el alto en 0 fijando constantes.
        self.offset_conv = nn.Conv2d(in_channels, 2 * ks, self.ks, padding=self.padding, dilation=self.dilation)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        # Capa para modular la importancia (máscara) de cada punto del kernel
        self.modulator_conv = nn.Conv2d(in_channels, ks, self.ks, padding=self.padding, dilation=self.dilation)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        # Convolución base que será deformada de forma adaptativa
        self.regular_conv = nn.Conv2d(in_channels, out_channels, self.ks, padding=self.padding, dilation=self.dilation, bias=False)

    def forward(self, x):
        # x llega como 1D: (batch, channels, seq_len) -> Convertimos a 2D: (batch, channels, seq_len, 1)
        x_2d = x.unsqueeze(-1)
        
        h, w = x_2d.shape[2:]
        max_offset = max(h, w) / 4.
        
        # Calcular desfases y limitarlos para que no se salgan drásticamente de la señal
        offset = self.offset_conv(x_2d).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x_2d))

        # Operación nativa de deformación de grilla espacial
        out_2d = torchvision.ops.deform_conv2d(
            input=x_2d, offset=offset, weight=self.regular_conv.weight, 
            bias=self.regular_conv.bias, padding=self.padding, 
            dilation=self.dilation, mask=modulator
        )
        return out_2d.squeeze(-1) # Devolver a formato 1D: (batch, channels, seq_len)


class FlexibleBlock(nn.Module):
    """
    Bloque configurable que conmuta matemáticamente entre CDIL y WaveNet Avanzado.
    """
    def __init__(self, c_in: int, c_out: int, ks: int, dil: int, 
                 causal: bool, gated: bool, deformable: bool, dropout: float = 0.0):
        super().__init__()
        self.causal = causal
        self.gated = gated
        self.deformable = deformable
        
        # Determinar canales internos según tipo de activación (Gated duplica canales para Tanh * Sigmoid)
        self.conv_out_channels = c_out * 2 if gated else c_out

        # Definir estrategia de Padding
        if self.causal:
            # Padding Causal: Se calcula el espacio total y se inyecta manualmente a la izquierda en el forward
            self.manual_padding = dil * (ks - 1)
            auto_pad = 0
            pad_mode = 'zeros'
        else:
            # Padding Circular Simétrico (Estilo CDIL original)
            self.manual_padding = 0
            auto_pad = int(dil * (ks - 1) / 2)
            pad_mode = 'circular'

        # Instanciar el núcleo convolucional (Deformable Inteligente vs Tradicional Rígido con Weight Norm)
        if self.deformable:
            # Las capas deformables manejan su propio acolchado interno estructurado en 2D
            self.conv = DeformableConv1d(c_in, self.conv_out_channels, ks, padding=auto_pad, dilation=dil)
        else:
            self.conv = weight_norm(
                nn.Conv1d(c_in, self.conv_out_channels, ks, padding=auto_pad, dilation=dil, padding_mode=pad_mode)
            )

        # Proyecciones lineales 1x1 para ramificaciones de WaveNet
        if self.gated:
            self.res_project = nn.Conv1d(c_out, c_out, 1)
            self.skip_project = nn.Conv1d(c_out, c_out, 1)

        # Conexión residual clásica (Marta Rey) para mantener consistencia de canales
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(self, x):
        # 1. Aplicar Padding Causal manual si corresponde
        if self.causal:
            x_conv = F.pad(x, (self.manual_padding, 0))
        else:
            x_conv = x

        # 2. Ejecutar Convolución
        out = self.conv(x_conv)

        # 3. Aplicar Activación según la configuración seleccionada
        if self.gated:
            # Mecanismo de compuertas WaveNet: Tanh (Filtro) * Sigmoid (Regulador)
            out_tanh, out_sigmoid = torch.chunk(out, 2, dim=1)
            gated_out = torch.tanh(out_tanh) * torch.sigmoid(out_sigmoid)
            
            if self.dropout is not None:
                gated_out = self.dropout(gated_out)
                
            res_out = self.res_project(gated_out)
            skip_out = self.skip_project(gated_out)
        else:
            # Activación Estilo CDIL Convencional
            gated_out = self.relu(out)
            if self.dropout is not None:
                gated_out = self.dropout(gated_out)
            res_out = gated_out
            skip_out = gated_out # En modo CDIL puro, skip hereda la salida estándar

        # 4. Cálculo de la identidad de retorno (Residual)
        identity = x if self.res is None else self.res(x)
        
        return res_out + identity, skip_out


class FlexibleWaveNetClassifier(nn.Module):
    """
    Arquitectura unificada con selectores para reproducir fielmente la red CDIL-CNN original
    o activar de forma aislada o conjunta los componentes avanzados de WaveNet.
    """
    def __init__(self, seq_len: int, nhid: int = 32, ks: int = 3,
                 causal: bool = False, gated: bool = False, deformable: bool = False,
                 dropout: float = 0.0, n_classes: int = 2):
        super().__init__()
        self.gated = gated
        
        # Cálculo dinámico idéntico de capas basado en log2 de la longitud temporal
        n_layers = min(int(np.log2(seq_len)), 8)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            c_in = 1 if i == 0 else nhid
            self.layers.append(
                FlexibleBlock(
                    c_in=c_in, c_out=nhid, ks=ks, dil=2**i, 
                    causal=causal, gated=gated, deformable=deformable, dropout=dropout
                )
            )
            
        self.linear = nn.Linear(nhid, n_classes)

    def forward(self, x):
        # Permutación idéntica a la de Marta Rey: (batch, seq_len, 1) -> (batch, 1, seq_len)
        x = x.permute(0, 2, 1).float()   
        
        total_skip = 0
        current_input = x
        
        for block in self.layers:
            current_input, skip_out = block(current_input)
            
            # Acumulación de Skip Connections (Solo sumamos si está en modo WaveNet Gated)
            if self.gated:
                total_skip = total_skip + skip_out

        # Selección de la representación final que va al Clasificador Lineal
        if self.gated:
            y = F.relu(total_skip)
        else:
            # En modo CDIL tradicional se toma la salida directa de la última capa residual
            y = current_input
            
        # Global Average Pooling idéntico
        y = torch.mean(y, dim=2)          
        return self.linear(y)


# ======================================================================
#  MAIN EXECUTION (Mismo flujo exacto de tu script original)
# ======================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(SEED)

    print(f"\n{'='*55}")
    print(f"  Experimento : {MODEL_NAME} (Estructura Flexible)")
    print(f"  Dataset     : {DATASET}")
    print(f"  Epochs      : {N_EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
    print(f"  NHID={NHID}, KS={KERNEL_SIZE}, Dropout={DROPOUT}")
    print(f"  Config      : Causal={CAUSAL} | Gated={GATED} | Deformable={DEFORMABLE}")
    print(f"  Device      : {device}")
    print(f"{'='*55}\n")

    assert SPLITS_FILE.exists(), f"No se encuentra {SPLITS_FILE}\nEjecuta primero data_split.py"

    with open(SPLITS_FILE) as f:
        splits = json.load(f)

    # Normalizar separadores de ruta (Windows → Unix)
    for ds in splits:
        if isinstance(splits[ds], list):
            for fd in splits[ds]:
                for k in ['train_files', 'test_files']:
                    if k in fd:
                        fd[k] = [p.replace('\\', '/') for p in fd[k]]

    folds_data = splits[DATASET]

    # Detectar longitud del audio desde el primer fichero
    first_file     = PROCESSED_DIR / folds_data[0]['train_files'][0]
    y0, sr0        = __import__('librosa').load(first_file, sr=None, mono=True)
    TARGET_SAMPLES = len(y0)
    n_layers       = min(int(np.log2(TARGET_SAMPLES)), 8)
    print(f"Muestras por audio   : {TARGET_SAMPLES}  ({TARGET_SAMPLES/sr0:.3f}s @ {sr0}Hz)")
    print(f"Capas convolucionales: {n_layers}")

    # Instanciar el clasificador flexible con la configuración del experimento
    _tmp     = FlexibleWaveNetClassifier(TARGET_SAMPLES, NHID, KERNEL_SIZE, CAUSAL, GATED, DEFORMABLE, DROPOUT)
    n_params = sum(p.numel() for p in _tmp.parameters() if p.requires_grad)
    print(f"Parámetros del modelo: {n_params:,}\n")
    del _tmp

    cfg = dict(LR=LR, N_EPOCHS=N_EPOCHS,
               WEIGHT_DECAY=WEIGHT_DECAY,
               EARLY_STOP_PATIENCE=EARLY_STOP_PATIENCE)

    # ------------------------------------------------------------------
    #  FOLD 1 — comprobación rápida
    # ------------------------------------------------------------------
    print("="*55)
    print("  FOLD 1 — COMPROBACIÓN RÁPIDA")
    print("="*55)

    f0                 = folds_data[0]
    train_f0, val_f0   = split_train_val_by_subject(
        list(f0['train_files']), VAL_RATIO, SEED)
    test_f0            = f0['test_files']

    print(f"\n  Train: {len(train_f0)} | Val: {len(val_f0)} | Test: {len(test_f0)}")
    ds_tr = AudioDataset(train_f0, TARGET_SAMPLES, PROCESSED_DIR)
    ds_v  = AudioDataset(val_f0,   TARGET_SAMPLES, PROCESSED_DIR)
    ds_te = AudioDataset(test_f0,  TARGET_SAMPLES, PROCESSED_DIR)
    ld_tr = DataLoader(ds_tr, BATCH_SIZE, shuffle=True)
    ld_v  = DataLoader(ds_v,  BATCH_SIZE, shuffle=False)
    ld_te = DataLoader(ds_te, BATCH_SIZE, shuffle=False)

    net = FlexibleWaveNetClassifier(TARGET_SAMPLES, NHID, KERNEL_SIZE, CAUSAL, GATED, DEFORMABLE, DROPOUT).to(device)
    net, tr_l, tr_a, v_l, v_a = train_fold(
        net, ld_tr, ld_v, device, 1, RESULTS_DIR, cfg)
    plot_learning_curves(tr_l, tr_a, v_l, v_a, 1, RESULTS_DIR, MODEL_NAME, DATASET)
    m1, *_ = evaluate_fold(net, ld_te, device, 1, RESULTS_DIR)
    print(f"\n  ✓ Fold 1 — Test Accuracy: {m1['accuracy']*100:.1f}%")

    # ------------------------------------------------------------------
    #  5-FOLD CROSS-VALIDATION COMPLETO
    # ------------------------------------------------------------------
    print("\n\n" + "="*55)
    print("  5-FOLD CROSS-VALIDATION COMPLETO")
    print("="*55)

    all_metrics = []
    all_true, all_pred = [], []
    mean_fpr = np.linspace(0, 1, 100)
    all_tprs = []

    for fold_idx in range(1, K_FOLDS + 1):
        print(f"\n\n{'─'*55}")
        print(f"  FOLD {fold_idx} / {K_FOLDS}")
        print(f"{'─'*55}")
        seed_everything(SEED + fold_idx)

        fd                       = folds_data[fold_idx - 1]
        train_files, val_files   = split_train_val_by_subject(
            list(fd['train_files']), VAL_RATIO, SEED + fold_idx)
        test_files               = fd['test_files']

        print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
        ds_tr = AudioDataset(train_files, TARGET_SAMPLES, PROCESSED_DIR)
        ds_v  = AudioDataset(val_files,   TARGET_SAMPLES, PROCESSED_DIR)
        ds_te = AudioDataset(test_files,  TARGET_SAMPLES, PROCESSED_DIR)
        ld_tr = DataLoader(ds_tr, BATCH_SIZE, shuffle=True)
        ld_v  = DataLoader(ds_v,  BATCH_SIZE, shuffle=False)
        ld_te = DataLoader(ds_te, BATCH_SIZE, shuffle=False)

        net = FlexibleWaveNetClassifier(TARGET_SAMPLES, NHID, KERNEL_SIZE, CAUSAL, GATED, DEFORMABLE, DROPOUT).to(device)
        net, tr_l, tr_a, v_l, v_a = train_fold(
            net, ld_tr, ld_v, device, fold_idx, RESULTS_DIR, cfg)
        plot_learning_curves(
            tr_l, tr_a, v_l, v_a, fold_idx, RESULTS_DIR, MODEL_NAME, DATASET)

        metrics, y_true, y_pred, y_prob, fpr_c, tpr_c = evaluate_fold(
            net, ld_te, device, fold_idx, RESULTS_DIR)

        all_metrics.append(metrics)
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
        interp       = np.interp(mean_fpr, fpr_c, tpr_c)
        interp[0]    = 0.0
        all_tprs.append(interp)

    # ------------------------------------------------------------------
    #  Resultados agregados
    # ------------------------------------------------------------------
    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics)
    print_summary(metrics_df, MODEL_NAME, DATASET, K_FOLDS)
    save_results(metrics_df, RESULTS_DIR, MODEL_NAME, DATASET)
    plot_accumulated_cm(all_true, all_pred, RESULTS_DIR, MODEL_NAME, DATASET)
    mean_row = metrics_df.drop('fold', axis=1).mean().to_dict()
    std_row  = metrics_df.drop('fold', axis=1).std().to_dict()
    plot_mean_roc(all_tprs, mean_fpr,
                  mean_row['auc'], std_row['auc'], RESULTS_DIR, MODEL_NAME, DATASET)

    print(f"\n\nResultados guardados en: {RESULTS_DIR}")
    print("¡Experimento completado!")


if __name__ == '__main__':
    main()