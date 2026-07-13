"""
train.py — Entrenamiento y evaluación CDIL CNN (5-Fold CV)
══════════════════════════════════════════════════════════════════════════════
Uso:
    python train.py

Antes de ejecutar:
    · Se ajustan los hiperparámetros en config.py

Salidas (en results/<nombre_experimento>/):
    · curves_fold{k}.png          curvas loss y accuracy por época
    · cm_fold{k}.png              matriz de confusión por fold
    · roc_fold{k}.png             curva ROC por fold
    · accumulated_cm.png          matriz de confusión acumulada (5 folds)
    · mean_roc_curve.png          curva ROC media (5 folds)
    · per_fold_metrics.csv        métricas detalladas por fold
    · summary_metrics.csv         media ± std de todas las métricas
    · config_used.py              copia exacta del config empleado
    · results/experiments_log.csv tabla comparativa de todos los experimentos
══════════════════════════════════════════════════════════════════════════════
"""

from turtle import fd
import os, sys, json, random, warnings, shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ── Importaciones locales ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import importlib.util

# from modelos.temp_attention.model import CDILClassifier
# CDILAttentionClassifier
from model import CDILAttentionClassifier

CLASS_NAMES = ['Control', 'Parkinson']


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_paths():
    """Sube carpetas hasta encontrar 'data/' para ser robusto a desde dónde se lanza."""
    root = Path(__file__).resolve().parent
    while not (root / 'data').exists() and root != root.parent:
        root = root.parent
    processed = root / cfg.PROCESSED_DIR
    splits    = root / cfg.SPLITS_FILE
    results   = Path(__file__).resolve().parent / 'results'
    return processed, splits, results


def get_patient_id(fp: str) -> str:
    """
    Extrae el ID de paciente del nombre de fichero.
      neurovoz : '0034_A1.wav'             →  '0034'
      pc-gita  : 'AVPEPUDEAC0003a1.wav'   →  'AVPEPUDEAC0003'
    """
    stem = Path(fp).stem
    if '_' in stem:
        return stem.rsplit('_', 1)[0]
    # pc-gita: quitar los últimos 2 caracteres (vocal+número, ej. 'a1')
    return stem[:-2] if len(stem) > 2 else stem

def build_experiment_name() -> str:
    if cfg.EXPERIMENT_NAME:
        return cfg.EXPERIMENT_NAME
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{ts}_{cfg.DATASET}_lr{cfg.LR}_bs{cfg.BATCH_SIZE}_nhid{cfg.NHID}_do{cfg.DROPOUT}"


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class AudioDataset(Dataset):
    def __init__(self, file_paths: list, processed_dir: Path, target_samples: int):
        self.x, self.y, n_skip = [], [], 0
        for fp in file_paths:
            full_path = processed_dir / fp
            label     = 0 if 'Control' in str(fp) else 1
            try:
                audio, _ = librosa.load(full_path, sr=None, mono=True)
                # Ajustar longitud
                if len(audio) >= target_samples:
                    audio = audio[:target_samples]
                else:
                    audio = np.pad(audio, (0, target_samples - len(audio)))
                # Normalización de amplitud
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
                self.x.append(audio.astype(np.float32))
                self.y.append(label)
            except Exception:
                n_skip += 1

        n_ctrl = sum(1 for l in self.y if l == 0)
        n_park = sum(1 for l in self.y if l == 1)
        skip_msg = f', omitidos={n_skip}' if n_skip else ''
        print(f"      {len(self.x)} audios  (Control={n_ctrl}, Parkinson={n_park}{skip_msg})")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx]).unsqueeze(-1),
                torch.tensor(self.y[idx], dtype=torch.long))


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(net, loader, optimizer, criterion, device):
    net.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = net(X)
        loss = criterion(pred, Y)
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
        pred = net(X)
        total_loss += criterion(pred, Y).item()
        correct    += (pred.argmax(1) == Y).sum().item()
        total      += len(Y)
    return total_loss / total, correct / total * 100.0


# def train_fold(net, train_loader, val_loader, device, fold_idx, run_dir):
def train_fold(net, train_loader, device, fold_idx, run_dir):

    criterion    = nn.CrossEntropyLoss(reduction='sum')
    optimizer    = torch.optim.Adam(net.parameters(), lr=cfg.LR)
    # best_val_acc = -1.0
    best_val_loss = float('inf')
    model_path   = run_dir / f'best_model_fold{fold_idx}.pt'

    tr_losses, tr_accs, vl_losses, vl_accs = [], [], [], []

    for epoch in range(cfg.N_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(net, train_loader, optimizer, criterion, device)
        # vl_loss, vl_acc = eval_epoch(net,  val_loader,  criterion, device)

        tr_losses.append(tr_loss); tr_accs.append(tr_acc)
        # vl_losses.append(vl_loss); vl_accs.append(vl_acc)

        # if vl_acc >= best_val_acc:
        #     best_val_acc = vl_acc
        #     torch.save(net.state_dict(), model_path)

        # if vl_loss <= best_val_loss:
        #     best_val_loss = vl_loss
        #     torch.save(net.state_dict(), model_path)

        torch.save(net.state_dict(), model_path)

        # Log cada N épocas + primera + última
        first_or_last = (epoch == 0 or epoch == cfg.N_EPOCHS - 1)
        if first_or_last or (epoch + 1) % cfg.LOG_EVERY_N_EPOCHS == 0:
            print(f"      Epoch [{epoch+1:3d}/{cfg.N_EPOCHS}]  "
                  f"Train — loss: {tr_loss:.4f}  acc: {tr_acc:.1f}%")
                #   f"Val — loss: {vl_loss:.4f}  acc: {vl_acc:.1f}%")

    net.load_state_dict(torch.load(model_path, map_location=device))
    # print(f"      ✓ Mejor val accuracy: {best_val_acc:.1f}%")
    # print(f"      ✓ Mejor val loss: {best_val_loss:.4f}")
    return net, tr_losses, tr_accs


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_fold(net, test_loader, device, fold_idx, run_dir):
    net.eval()
    y_true, y_pred, y_prob = [], [], []

    for X, Y in test_loader:
        X, Y  = X.to(device), Y.to(device)
        out   = net(X)
        probs = torch.softmax(out, dim=1)[:, 1]
        preds = out.argmax(1)
        y_true.extend(Y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true,    y_pred, zero_division=0)
    f1   = f1_score(y_true,        y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)

    print(f"\n      {'─'*50}")
    print(f"      Test Fold {fold_idx}  →  "
          f"Acc: {acc:.4f}  |  Prec: {prec:.4f}  |  Rec: {rec:.4f}  |  "
          f"Spec: {spec:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}")
    print(f"      {'─'*50}")

    # Matriz de confusión
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title(f'Confusion Matrix — Fold {fold_idx}')
    plt.tight_layout()
    plt.savefig(run_dir / f'cm_fold{fold_idx}.png', dpi=150); plt.close()

    # Curva ROC
    plt.figure(figsize=(4, 3.5))
    plt.plot(fpr_c, tpr_c, 'b-', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0,1],[0,1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — Fold {fold_idx}')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / f'roc_fold{fold_idx}.png', dpi=150); plt.close()

    metrics = dict(fold=fold_idx, accuracy=acc, precision=prec,
                   recall=rec, specificity=spec, f1=f1, auc=auc)
    return metrics, np.array(y_true), np.array(y_pred), np.array(y_prob), fpr_c, tpr_c


# ══════════════════════════════════════════════════════════════════════════════
#  GRÁFICAS FINALES
# ══════════════════════════════════════════════════════════════════════════════

# def plot_learning_curves(tr_losses, tr_accs, vl_losses, vl_accs, fold_idx, run_dir):
# def plot_learning_curves(tr_losses, tr_accs, fold_idx, run_dir):
#     epochs = range(1, len(tr_losses) + 1)
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
#     ax1.plot(epochs, tr_losses, label='Train')
#     # ax1.plot(epochs, vl_losses, label='Validation')
#     ax1.set_title('Loss'); ax1.set_xlabel('Epoch')
#     ax1.legend(); ax1.grid(alpha=0.3)
#     ax2.plot(epochs, tr_accs, label='Train')
#     # ax2.plot(epochs, vl_accs, label='Validation')
#     ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch')
#     ax2.legend(); ax2.grid(alpha=0.3)
#     fig.suptitle(f'Learning Curves — Fold {fold_idx}  [{cfg.DATASET}]')
#     plt.tight_layout()
#     plt.savefig(run_dir / f'curves_fold{fold_idx}.png', dpi=150); plt.close()


def plot_accumulated_cm(all_true, all_pred, run_dir):
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title(f'Accumulated Confusion Matrix — {cfg.K_FOLDS}-Fold CV')
    plt.tight_layout()
    plt.savefig(run_dir / 'accumulated_cm.png', dpi=150); plt.close()


def plot_mean_roc(all_tprs, mean_fpr, mean_auc, std_auc, run_dir):
    mean_tpr    = np.mean(all_tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr     = np.std(all_tprs,  axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_fpr, mean_tpr, 'b-', lw=2,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr,
                    np.maximum(mean_tpr - std_tpr, 0),
                    np.minimum(mean_tpr + std_tpr, 1),
                    color='grey', alpha=0.2, label='±1 std')
    ax.plot([0,1],[0,1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Mean ROC Curve — {cfg.K_FOLDS}-Fold CV  [{cfg.DATASET}]')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / 'mean_roc_curve.png', dpi=150); plt.close()


def print_summary(metrics_df: pd.DataFrame):
    mean = metrics_df.drop('fold', axis=1).mean()
    std  = metrics_df.drop('fold', axis=1).std()
    cv   = (std / mean).fillna(0)

    print("\n" + "═"*60)
    print(f"  RESULTADOS FINALES — {cfg.K_FOLDS}-Fold CV  [{cfg.DATASET}]")
    print("═"*60)
    print(f"  {'Métrica':<16} {'Media':>8}  {'±Std':>8}  {'CV':>8}")
    print(f"  {'─'*48}")
    for col in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']:
        print(f"  {col.capitalize():<16} {mean[col]:>8.4f}  {std[col]:>8.4f}  {cv[col]:>8.4f}")
    print("═"*60)
    print("\n  Detalle por fold:")
    cols = ['fold', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
    print(metrics_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRO DE EXPERIMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def update_experiments_log(results_dir: Path, run_name: str, metrics_df: pd.DataFrame):
    """
    Añade una fila al fichero experiments_log.csv con los hiperparámetros
    y resultados de esta ejecución. Permite comparar todos los experimentos.
    """
    mean = metrics_df.drop('fold', axis=1).mean()
    std  = metrics_df.drop('fold', axis=1).std()

    row = {
        'experiment':  run_name,
        'timestamp':   datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset':     cfg.DATASET,
        'nhid':        cfg.NHID,
        'kernel_size': cfg.KERNEL_SIZE,
        'dropout':     cfg.DROPOUT,
        'batch_size':  cfg.BATCH_SIZE,
        'n_epochs':    cfg.N_EPOCHS,
        'lr':          cfg.LR,
        # 'val_ratio':   cfg.VAL_RATIO,
        'seed':        cfg.SEED,
    }
    for col in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']:
        row[f'mean_{col}'] = round(mean[col], 4)
        row[f'std_{col}']  = round(std[col],  4)

    log_path = results_dir / 'experiments_log.csv'
    new_row  = pd.DataFrame([row])

    if log_path.exists():
        existing = pd.read_csv(log_path)
        updated  = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    updated.to_csv(log_path, index=False)
    print(f"\n  ✓ Experimento registrado en: {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    seed_everything(cfg.SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processed_dir, splits_file, results_dir = resolve_paths()
    results_dir.mkdir(parents=True, exist_ok=True)

    assert splits_file.exists(), (
        f"\n✗ No se encuentra {splits_file}\n"
        f"  Ejecuta primero el script de división de datos."
    )

    # ── Nombre del experimento y carpeta de salida ────────────────────────────
    run_name = build_experiment_name()
    run_dir  = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Guardar copia del config usado en este experimento
    shutil.copy(Path(__file__).resolve().parent / 'config.py',
                run_dir / 'config_used.py')

    # ── Cabecera ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  CDIL CNN — Detección de Parkinson")
    print(f"  Experimento : {run_name}")
    print(f"  Dataset     : {cfg.DATASET}")
    print(f"  Config      : nhid={cfg.NHID}, ks={cfg.KERNEL_SIZE}, "
          f"dropout={cfg.DROPOUT}")
    print(f"  Entrenamiento: epochs={cfg.N_EPOCHS}, lr={cfg.LR}, "
          f"batch={cfg.BATCH_SIZE}")
    print(f"  Dispositivo : {device}")
    print(f"  Resultados  : {run_dir}")
    print(f"{'═'*60}\n")

    # ── Cargar splits ─────────────────────────────────────────────────────────
    with open(splits_file) as f:
        splits = json.load(f)

    # Normalizar separadores de ruta (Windows → Unix)
    folds_data = splits[cfg.DATASET]
    for fold in folds_data:
        for key in ['train_files', 'test_files']:
            if key in fold:
                fold[key] = [p.replace('\\', '/') for p in fold[key]]

    # ── Detectar longitud de señal desde el primer audio ─────────────────────
    first_file     = processed_dir / folds_data[0]['train_files'][0]
    y0, sr0        = librosa.load(first_file, sr=None, mono=True)
    TARGET_SAMPLES = len(y0)

    # Mostrar info del modelo
    _net_tmp  = CDILAttentionClassifier(TARGET_SAMPLES, cfg.NHID, cfg.KERNEL_SIZE, cfg.DROPOUT)
    print(f"  Muestras/audio : {TARGET_SAMPLES}  ({TARGET_SAMPLES/sr0:.3f}s @ {sr0}Hz)")
    print(f"  Capas CDIL     : {_net_tmp.n_layers}  "
          f"(= min(floor(log2({TARGET_SAMPLES})), 8))")
    print(f"  Parámetros     : {_net_tmp.n_params:,}\n")
    del _net_tmp

    # ── Bucle de folds ────────────────────────────────────────────────────────
    all_metrics = []
    all_true    = []
    all_pred    = []
    mean_fpr    = np.linspace(0, 1, 100)
    all_tprs    = []

    for fold_idx in range(1, cfg.K_FOLDS + 1):
        print(f"{'─'*60}")
        print(f"  FOLD {fold_idx} / {cfg.K_FOLDS}")
        print(f"{'─'*60}")
        seed_everything(cfg.SEED + fold_idx)

        fd         = folds_data[fold_idx - 1]
        train_f = fd['train_files']   # todos los audios de train, sin split
        test_f = fd['test_files']



        print(f"    Sujetos → Train: {len(set(get_patient_id(f) for f in train_f))}  "
            f"Test: {len(set(get_patient_id(f) for f in test_f))}")

        print(f"    Audios  → Train: {len(train_f)}  "
            f"Test: {len(test_f)}")


        print("    Cargando audios...")
        ds_tr = AudioDataset(train_f, processed_dir, TARGET_SAMPLES)
        ds_te = AudioDataset(test_f,  processed_dir, TARGET_SAMPLES)

        ld_tr = DataLoader(ds_tr, cfg.BATCH_SIZE, shuffle=True,  drop_last=False)
        ld_te = DataLoader(ds_te, cfg.BATCH_SIZE, shuffle=False, drop_last=False)

        net = CDILAttentionClassifier(TARGET_SAMPLES, cfg.NHID,
                             cfg.KERNEL_SIZE, cfg.DROPOUT).to(device)

        print(f"\n    Entrenando ({cfg.N_EPOCHS} épocas, log cada "
              f"{cfg.LOG_EVERY_N_EPOCHS})...")

        net, tr_l, tr_a = train_fold(net, ld_tr, device, fold_idx, run_dir)
        # plot_learning_curves(tr_l, tr_a, vl_l, vl_a, fold_idx, run_dir)
        # plot_learning_curves(tr_l, tr_a, fold_idx, run_dir)


        metrics, y_true, y_pred, y_prob, fpr_c, tpr_c = evaluate_fold(
            net, ld_te, device, fold_idx, run_dir
        )

        all_metrics.append(metrics)
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
        interp_tpr    = np.interp(mean_fpr, fpr_c, tpr_c); interp_tpr[0] = 0.
        all_tprs.append(interp_tpr)

    # ── Resultados agregados ──────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics)
    print_summary(metrics_df)

    mean_m = metrics_df.drop('fold', axis=1).mean()
    std_m  = metrics_df.drop('fold', axis=1).std()

    metrics_df.to_csv(run_dir / 'per_fold_metrics.csv', index=False)
    summary = {f'mean_{k}': round(v, 4) for k, v in mean_m.items()}
    summary.update({f'std_{k}': round(v, 4) for k, v in std_m.items()})
    summary.update({'model': 'CDIL', 'dataset': cfg.DATASET})
    pd.DataFrame([summary]).to_csv(run_dir / 'summary_metrics.csv', index=False)

    plot_accumulated_cm(all_true, all_pred, run_dir)
    plot_mean_roc(all_tprs, mean_fpr, mean_m['auc'], std_m['auc'], run_dir)

    # ── Registro comparativo de experimentos ─────────────────────────────────
    update_experiments_log(results_dir, run_name, metrics_df)

    print(f"\n  ✓ Todos los resultados guardados en: {run_dir}")
    print(f"\n  Para comparar experimentos abre: {results_dir / 'experiments_log.csv'}")
    print("\n  ¡Experimento completado!\n")


if __name__ == '__main__':
    main()
