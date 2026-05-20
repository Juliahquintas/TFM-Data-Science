"""
Experiment: CDIL-CNN (modificado con Dropout) + TCN para detección de Parkinson
Basado en: Marta Rey-Paredes et al., 2024 (TFM / IEEE Open Journal)

Diferencia respecto a Marta Rey:
  - CDIL-CNN: misma arquitectura pero con Dropout=0.2 tras cada bloque
              (regularización, su código no tiene dropout)
  - TCN:      arquitectura nueva; Marta Rey no la probó para Parkinson

Cómo usar:
  1. Asegúrate de que data/data_splits.json existe (ejecuta data_split.py antes)
  2. Cambia DATASET y MODEL al principio según lo que quieras correr
  3. Ejecuta: python parkinson_experiment.py
  4. Los resultados se guardan en results/<DATASET>_<MODEL>/

Colocar este archivo en: src/experiments/parkinson_experiment.py
"""

import os
import sys
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ======================================================================
#  CONFIGURACIÓN — CAMBIA AQUÍ SEGÚN EL EXPERIMENTO
# ======================================================================
DATASET     = 'pc-gita'   # 'neurovoz' o 'pc-gita'
MODEL       = 'CDIL'       # 'CDIL' o 'TCN'

# Hiperparámetros del modelo
# NHID: numero de feature maps en cada capa convolucional.
NHID        = 32           # canales ocultos (igual que Marta Rey)
# KERNEL SIZE: tamaño del kernel, cada filtro ve 3 muestras seguidas, antes de que entren las convoluciones.
KERNEL_SIZE = 3            # tamaño del kernel (igual que Marta Rey)
DROPOUT     = 0.2          # 0.2 para CDIL (nuestra modificación); 0.0 para TCN

# Hiperparámetros de entrenamiento
# BATCH SIZE: numero de audios procesados juntos en cada paso de gradiente.
BATCH_SIZE  = 32           # dataset pequeño → batch pequeño
N_EPOCHS    = 100          # Marta Rey usó 200; reduce a 50 para pruebas rápidas
LR          = 1e-4
SEED        = 42
K_FOLDS     = 5
VAL_RATIO   = 0.10         # fracción del train usada como validación


# ======================================================================
#  SEMILLA
# ======================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


# ======================================================================
#  RUTAS
# ======================================================================
# Sube hasta encontrar la carpeta 'data/' (funciona desde src/experiments/)
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / 'data').exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
SPLITS_FILE   = PROJECT_ROOT / 'data' / 'data_splits.json'
RESULTS_DIR   = PROJECT_ROOT / 'results' / f'{DATASET}_{MODEL}'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT  : {PROJECT_ROOT}")
print(f"PROCESSED_DIR : {PROCESSED_DIR}")
print(f"SPLITS_FILE   : {SPLITS_FILE}")
print(f"RESULTS_DIR   : {RESULTS_DIR}")

assert SPLITS_FILE.exists(), (
    f"\nNo se encuentra {SPLITS_FILE}\n"
    f"Ejecuta primero: python src/preprocessing/data_split.py"
)



# ======================================================================
#  ARQUITECTURA DEL MODELO
#  Basada en net_conv.py de Marta Rey.
#  Modificación en CDIL: Dropout tras cada bloque convolucional.
# ======================================================================

# Para recortar el padding que se añade al final, porque eso supone informacion futura, y al estar usando causal, no podemos añadirlo.
# CausalCrop recorta exactamente ese padding derecho, dejando solo el contexto del pasado. Es lo que hace que TCN sea causal.
class CausalCrop(nn.Module):
    """Elimina el padding extra de la convolución causal (TCN)."""
    def __init__(self, crop_size: int):
        super().__init__()
        self.crop = crop_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.crop].contiguous()


## ========== BLOQUE CONVOLUCIONAL ===================
class ConvBlock(nn.Module):
    """
    Bloque convolucional dilatado con conexión residual.
    Soporta modos CDIL (circular) y TCN (causal).

    Modificación respecto a Marta Rey:
        - Añadido Dropout tras la activación (Marta Rey no tiene dropout).
    """
    # Se configuran las bases del bloque convolucional
    def __init__(self, model: str, c_in: int, c_out: int,
                 ks: int, pad: int, dil: int, dropout: float = 0.0):
        super().__init__()
        self.model = model
        pad_mode   = 'circular' if model == 'CDIL' else 'zeros'

        self.conv = weight_norm(
            nn.Conv1d(c_in, c_out, ks,
                      padding=pad, dilation=dil, padding_mode=pad_mode)
        )
        nn.init.normal_(self.conv.weight_g, 0, 0.01)
        nn.init.normal_(self.conv.bias,     0, 0.01)

        # Para TCN causal: eliminamos el padding de la derecha
        self.causal_crop = CausalCrop(pad) if model == 'TCN' else None

        # Skip connections; bloques residuales: 
        # Proyección residual (solo cuando cambia la dimensión)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        if self.res is not None:
            nn.init.normal_(self.res.weight, 0, 0.01)
            nn.init.normal_(self.res.bias,   0, 0.01)

        # Dropout
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None


    # Aqui se ejecuta el bloque convolucional. (entrada --> salida)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.causal_crop is not None:
            out = self.causal_crop(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        res = x if self.res is None else self.res(x)
        return out + res


## ========== BLOQUE DE CLASIFICACION ===================
class ParkinsonClassifier(nn.Module):
    """
    Clasificador completo: pila de ConvBlocks + pooling + cabeza lineal.

    Entrada: (batch, seq_len, 1)  →  permuta a  (batch, 1, seq_len)
    Salida:  (batch, n_classes)   logits sin softmax

    Número de capas = floor(log2(seq_len))  →  igual que Marta Rey
    """
    def __init__(self, model: str, seq_len: int,
                 nhid: int = 32, ks: int = 3,
                 dropout: float = 0.0, n_classes: int = 2):
        super().__init__()
        self.model   = model
        n_layers     = int(np.log2(seq_len))   # ej: log2(10760) ≈ 13
        # el número de capas conv. se calcula en base al audio, para que la red cubra todo el audio

        layers = []
        for i in range(n_layers):
            c_in = 1    if i == 0 else nhid
            c_out = nhid
            dil   = 2 ** i  # las dilataciones van aumentando expo. y por tanto el campo receptivo tambn.
            if model == 'TCN':
                pad = dil * (ks - 1)           # padding causal
            else:                              # CDIL: dilation simétrica
                pad = int(dil * (ks - 1) / 2)
            layers.append(ConvBlock(model, c_in, c_out, ks, pad, dil, dropout))

        self.conv_net = nn.Sequential(*layers)
        self.linear   = nn.Linear(nhid, n_classes) # capa lineal de clasificacion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).float()   # (batch, 1, seq_len)
        y = self.conv_net(x)              # (batch, nhid, seq_len)
        # Agregación temporal
        if self.model == 'TCN':
            y = y[:, :, -1]              # último paso (causal)
        else:
            y = torch.mean(y, dim=2)     # global average pooling
        return self.linear(y)            # (batch, n_classes)



# ======================================================================
#  DATASET
# ======================================================================
CLASS_NAMES = ['Control', 'Parkinson']

class AudioDataset(Dataset):
    """
    Carga audios ya preprocesados desde data/processed/.
    Infiere la etiqueta del nombre de carpeta: Control→0, Patologicas→1
    """
    def __init__(self, file_paths: list, target_samples: int):
        self.x, self.y, n_skip = [], [], 0
        for fp in file_paths:
            full_path = PROCESSED_DIR / fp
            label     = 0 if 'Control' in str(fp) else 1
            try:
                audio, _ = librosa.load(full_path, sr=None, mono=True)
                # Ajustar longitud exacta
                if len(audio) >= target_samples:
                    audio = audio[:target_samples]
                else:
                    audio = np.pad(audio, (0, target_samples - len(audio)))
                # Normalización amplitud (por si el preprocesamiento no lo hizo)
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
                self.x.append(audio.astype(np.float32))
                self.y.append(label)
            except Exception as e:
                n_skip += 1
        n_ctrl = sum(1 for l in self.y if l == 0)
        n_park = sum(1 for l in self.y if l == 1)
        print(f"    Cargados: {len(self.x)} audios "
              f"(Control={n_ctrl}, Parkinson={n_park}"
              f"{', omitidos='+str(n_skip) if n_skip else ''})")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Salida: (seq_len, 1) y etiqueta escalar
        return (torch.tensor(self.x[idx]).unsqueeze(-1),
                torch.tensor(self.y[idx], dtype=torch.long))



# ======================================================================
#  FUNCIONES DE ENTRENAMIENTO
# ======================================================================
def train_one_epoch(net, loader, optimizer, criterion, device):
    net.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred  = net(X)
        loss  = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (pred.argmax(1) == Y).sum().item()
        total      += len(Y)
    return total_loss / total, correct / total * 100.0


@torch.no_grad()
def eval_epoch(net, loader, criterion, device):
    net.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        pred  = net(X)
        total_loss += criterion(pred, Y).item()
        correct    += (pred.argmax(1) == Y).sum().item()
        total      += len(Y)
    return total_loss / total, correct / total * 100.0


def train_fold(net, train_loader, val_loader, device, fold_idx, results_dir):
    criterion     = nn.CrossEntropyLoss(reduction='sum')
    optimizer     = torch.optim.Adam(net.parameters(), lr=LR)
    best_val_acc  = -1.0
    model_path    = results_dir / f'best_model_fold{fold_idx}.pt'

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(N_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(net, train_loader, optimizer, criterion, device)
        v_loss,  v_acc  = eval_epoch(net,  val_loader,   criterion, device)

        train_losses.append(tr_loss); train_accs.append(tr_acc)
        val_losses.append(v_loss);   val_accs.append(v_acc)

        # Guardar mejor modelo (por accuracy en validación)
        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            torch.save(net.state_dict(), model_path)

        # Log cada 10 epochs + primera y última
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == N_EPOCHS - 1:
            print(f"    Epoch [{epoch+1:3d}/{N_EPOCHS}] "
                  f"Train: loss={tr_loss:.4f}, acc={tr_acc:.1f}%  |  "
                  f"Val: loss={v_loss:.4f}, acc={v_acc:.1f}%")

    # Cada vez que la valid. accuracy mejora, se guardan los pesos. Al final se carga el mejor checkpoint.
    # Cargar el mejor checkpoint
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"    → Mejor val acc: {best_val_acc:.1f}%  (checkpoint: {model_path.name})")
    return net, train_losses, train_accs, val_losses, val_accs



# ======================================================================
#  EVALUACIÓN COMPLETA
# ======================================================================
@torch.no_grad()
def evaluate_fold(net, test_loader, device, fold_idx, results_dir):
    net.eval()
    y_true, y_pred, y_prob = [], [], []

    for X, Y in test_loader:
        X, Y  = X.to(device), Y.to(device)
        out   = net(X)
        probs = torch.softmax(out, dim=1)[:, 1]   # P(Parkinson)
        preds = out.argmax(1)
        y_true.extend(Y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    # — Métricas —
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_prob)

    # — Print —
    print(f"\n  {'─'*45}")
    print(f"  FOLD {fold_idx} — Métricas en TEST")
    print(f"  {'─'*45}")
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision   : {prec:.4f}")
    print(f"  Recall/Sens : {rec:.4f}")
    print(f"  Specificity : {spec:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # — Gráfica: Matriz de Confusión —
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title(f'Confusion Matrix — Fold {fold_idx}')
    plt.tight_layout()
    plt.savefig(results_dir / f'cm_fold{fold_idx}.png', dpi=150)
    plt.close()

    # — Gráfica: Curva ROC —
    plt.figure(figsize=(5, 4))
    plt.plot(fpr_curve, tpr_curve, 'b-', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curve — Fold {fold_idx}')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / f'roc_fold{fold_idx}.png', dpi=150)
    plt.close()

    metrics = dict(fold=fold_idx, accuracy=acc, precision=prec,
                   recall=rec, specificity=spec, f1=f1, auc=auc)
    return metrics, np.array(y_true), np.array(y_pred), np.array(y_prob), fpr_curve, tpr_curve



# ======================================================================
#  GRÁFICAS GLOBALES
# ======================================================================
def plot_learning_curves(train_losses, train_accs, val_losses, val_accs,
                         fold_idx, results_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_losses, label='Train')
    ax1.plot(epochs, val_losses,   label='Validation')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, train_accs, label='Train')
    ax2.plot(epochs, val_accs,   label='Validation')
    ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f'Learning Curves — Fold {fold_idx}  [{MODEL} | {DATASET}]')
    plt.tight_layout()
    plt.savefig(results_dir / f'curves_fold{fold_idx}.png', dpi=150)
    plt.close()


def plot_mean_roc(all_tprs, mean_fpr, mean_auc, std_auc, results_dir):
    mean_tpr      = np.mean(all_tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr       = np.std(all_tprs,  axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(mean_fpr, mean_tpr, 'b-', lw=2,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    color='grey', alpha=0.2, label='±1 std')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Mean ROC Curve (5-Fold CV)  [{MODEL} | {DATASET}]')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'mean_roc_curve.png', dpi=150)
    plt.close()


def plot_accumulated_cm(all_true, all_pred, results_dir):
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title(f'Accumulated Confusion Matrix (5-Fold CV)  [{MODEL} | {DATASET}]')
    plt.tight_layout()
    plt.savefig(results_dir / 'accumulated_cm.png', dpi=150)
    plt.close()
    print("\nMatriz de Confusión Acumulada (5 folds):")
    print(cm)
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES))


def print_summary(metrics_df):
    mean = metrics_df.drop('fold', axis=1).mean()
    std  = metrics_df.drop('fold', axis=1).std()
    cv   = std / mean   # coeficiente de variación

    print("\n" + "="*55)
    print(f"  RESUMEN FINAL — {K_FOLDS}-Fold CV  [{MODEL} | {DATASET}]")
    print("="*55)
    print(f"  {'Métrica':<16} {'Media':>8} {'Std':>8} {'CV':>8}")
    print(f"  {'─'*44}")
    for col in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']:
        print(f"  {col.capitalize():<16} {mean[col]:>8.4f} {std[col]:>8.4f} {cv[col]:>8.4f}")
    print("="*55)
    # Tabla por fold
    print("\n  Detalle por fold:")
    cols = ['fold', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
    print(metrics_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ======================================================================
#  MAIN
# ======================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*55}")
    print(f"  Experimento: {MODEL} para detección de Parkinson")
    print(f"  Dataset    : {DATASET}")
    print(f"  Epochs     : {N_EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
    print(f"  NHID={NHID}, KS={KERNEL_SIZE}, Dropout={DROPOUT}")
    print(f"  Device     : {device}")
    print(f"{'='*55}\n")

    # Cargar splits
    with open(SPLITS_FILE) as f:
        splits = json.load(f)
    folds_data = splits[DATASET]   # lista de K dicts con 'train_files' y 'test_files'


    # Detectar target_samples del primer audio
    first_file  = PROCESSED_DIR / folds_data[0]['train_files'][0]
    y0, sr0     = librosa.load(first_file, sr=None, mono=True)
    TARGET_SAMPLES = len(y0)
    n_layers       = int(np.log2(TARGET_SAMPLES))
    print(f"Muestras por audio  : {TARGET_SAMPLES}  ({TARGET_SAMPLES/sr0:.3f}s @ {sr0}Hz)")
    print(f"Capas convolucionales: {n_layers}  (= floor(log2({TARGET_SAMPLES})))")

    # Número de parámetros (instanciamos una red de prueba)
    _net_test = ParkinsonClassifier(MODEL, TARGET_SAMPLES, NHID, KERNEL_SIZE, DROPOUT)
    n_params  = sum(p.numel() for p in _net_test.parameters() if p.requires_grad)
    print(f"Parámetros del modelo: {n_params:,}")
    del _net_test

    # ==================================================================
    #  FOLD 1 — Comprobación rápida antes de lanzar los 5 folds
    # ==================================================================
    print("\n" + "="*55)
    print("  FOLD 1 — COMPROBACIÓN RÁPIDA")
    print("="*55)

    f0        = folds_data[0]
    all_train = list(f0['train_files'])
    random.shuffle(all_train)
    n_val        = max(1, int(len(all_train) * VAL_RATIO))
    val_f0       = all_train[:n_val]
    train_f0     = all_train[n_val:]
    test_f0      = f0['test_files']

    print(f"\n  Partición Fold 1 — Train: {len(train_f0)} | Val: {len(val_f0)} | Test: {len(test_f0)}")
    print("  Cargando audios (Fold 1):")
    ds_tr = AudioDataset(train_f0, TARGET_SAMPLES)
    ds_v  = AudioDataset(val_f0,   TARGET_SAMPLES)
    ds_te = AudioDataset(test_f0,  TARGET_SAMPLES)

    ld_tr = DataLoader(ds_tr, BATCH_SIZE, shuffle=True,  drop_last=False)
    ld_v  = DataLoader(ds_v,  BATCH_SIZE, shuffle=False, drop_last=False)
    ld_te = DataLoader(ds_te, BATCH_SIZE, shuffle=False, drop_last=False)

    net = ParkinsonClassifier(MODEL, TARGET_SAMPLES, NHID, KERNEL_SIZE, DROPOUT).to(device)
    print("\n  Entrenando Fold 1...")
    net, tr_l, tr_a, v_l, v_a = train_fold(net, ld_tr, ld_v, device, 1, RESULTS_DIR)
    plot_learning_curves(tr_l, tr_a, v_l, v_a, 1, RESULTS_DIR)
    m1, _, _, _, _, _ = evaluate_fold(net, ld_te, device, 1, RESULTS_DIR)

    print(f"\n  ✓ Fold 1 completado — Test Accuracy: {m1['accuracy']*100:.1f}%")
    print(f"  Las curvas y matrices se han guardado en: {RESULTS_DIR}")

    # ==================================================================
    #  5-FOLD CROSS-VALIDATION COMPLETO
    # ==================================================================
    print("\n\n" + "="*55)
    print("  5-FOLD CROSS-VALIDATION COMPLETO")
    print("="*55)

    all_metrics = []
    all_true    = []
    all_pred    = []
    mean_fpr    = np.linspace(0, 1, 100)
    all_tprs    = []

    for fold_idx in range(1, K_FOLDS + 1):
        print(f"\n\n{'─'*55}")
        print(f"  FOLD {fold_idx} / {K_FOLDS}")
        print(f"{'─'*55}")
        seed_everything(SEED + fold_idx)   # semilla diferente por fold

        fd        = folds_data[fold_idx - 1]
        all_tr    = list(fd['train_files'])
        random.shuffle(all_tr)
        n_v       = max(1, int(len(all_tr) * VAL_RATIO))
        val_files  = all_tr[:n_v]
        train_files = all_tr[n_v:]
        test_files  = fd['test_files']

        print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
        print("  Cargando audios...")
        ds_tr = AudioDataset(train_files, TARGET_SAMPLES)
        ds_v  = AudioDataset(val_files,   TARGET_SAMPLES)
        ds_te = AudioDataset(test_files,  TARGET_SAMPLES)

        ld_tr = DataLoader(ds_tr, BATCH_SIZE, shuffle=True,  drop_last=False)
        ld_v  = DataLoader(ds_v,  BATCH_SIZE, shuffle=False, drop_last=False)
        ld_te = DataLoader(ds_te, BATCH_SIZE, shuffle=False, drop_last=False)

        net = ParkinsonClassifier(MODEL, TARGET_SAMPLES, NHID, KERNEL_SIZE, DROPOUT).to(device)
        print("\n  Entrenando...")
        net, tr_l, tr_a, v_l, v_a = train_fold(
            net, ld_tr, ld_v, device, fold_idx, RESULTS_DIR)
        plot_learning_curves(tr_l, tr_a, v_l, v_a, fold_idx, RESULTS_DIR)

        metrics, y_true, y_pred, y_prob, fpr_c, tpr_c = evaluate_fold(
            net, ld_te, device, fold_idx, RESULTS_DIR)

        all_metrics.append(metrics)
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        # Interpolamos TPR a FPR común para media de curva ROC
        interp_tpr    = np.interp(mean_fpr, fpr_c, tpr_c)
        interp_tpr[0] = 0.0
        all_tprs.append(interp_tpr)

    # ==================================================================
    #  RESULTADOS AGREGADOS
    # ==================================================================
    metrics_df = pd.DataFrame(all_metrics)
    print_summary(metrics_df)

    # Guardar CSVs
    metrics_df.to_csv(RESULTS_DIR / 'per_fold_metrics.csv', index=False)
    mean_row = metrics_df.drop('fold', axis=1).mean().to_dict()
    std_row  = metrics_df.drop('fold', axis=1).std().to_dict()
    summary  = {f'mean_{k}': v for k, v in mean_row.items()}
    summary.update({f'std_{k}': v for k, v in std_row.items()})
    summary['model']   = MODEL
    summary['dataset'] = DATASET
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / 'summary_metrics.csv', index=False)

    # Gráficas acumuladas
    plot_accumulated_cm(all_true, all_pred, RESULTS_DIR)
    plot_mean_roc(all_tprs, mean_fpr,
                  mean_row['auc'], std_row['auc'], RESULTS_DIR)

    print(f"\n\nTodos los resultados guardados en: {RESULTS_DIR}")
    print("Archivos generados:")
    for f in sorted(RESULTS_DIR.glob('*')):
        print(f"  {f.name}")
    print("\n¡Experimento completado!")


if __name__ == '__main__':
    main()
