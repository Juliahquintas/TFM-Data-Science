"""
metrics_utils.py  —  Utilidades compartidas: datos, entrenamiento y evaluación
Colocar en: src/experiments/metrics_utils.py
No se ejecuta directamente; lo importan cdil_experiment.py y tcn_experiment.py.
"""

import os
import random
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

CLASS_NAMES = ['Control', 'Parkinson']


# ======================================================================
#  SEMILLA
# ======================================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================================
#  SPLIT VALIDACIÓN POR PACIENTE
# ======================================================================
def split_train_val_by_subject(train_files: list,
                                val_ratio: float = 0.20,
                                seed: int = 42):
    """
    Divide train_files en train/val sin mezclar pacientes entre grupos.
        'neurovoz/Control/0034_A1.wav'          →  paciente '0034'
        'pc-gita/Control/AVPEPUDEA0001_a1.wav'  →  paciente 'AVPEPUDEA0001'
    """
    def get_pid(fp):
        return Path(fp).stem.rsplit('_', 1)[0]

    groups = defaultdict(list)
    for fp in train_files:
        groups[get_pid(fp)].append(fp)

    patients = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_val      = max(1, int(len(patients) * val_ratio))
    val_pids   = set(patients[:n_val])
    train_pids = set(patients[n_val:])

    return (
        [fp for fp in train_files if get_pid(fp) in train_pids],
        [fp for fp in train_files if get_pid(fp) in val_pids],
    )


# ======================================================================
#  DATASET
# ======================================================================
class AudioDataset(Dataset):
    """
    Carga audios preprocesados desde PROCESSED_DIR.
    Etiqueta: Control → 0,  Patologicas → 1  (inferida de la ruta).
    """
    def __init__(self, file_paths: list, target_samples: int, processed_dir: Path):
        self.x, self.y, n_skip = [], [], 0
        for fp in file_paths:
            label = 0 if 'Control' in str(fp) else 1
            try:
                audio, _ = librosa.load(processed_dir / fp, sr=None, mono=True)
                if len(audio) >= target_samples:
                    audio = audio[:target_samples]
                else:
                    audio = np.pad(audio, (0, target_samples - len(audio)))
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak
                self.x.append(audio.astype(np.float32))
                self.y.append(label)
            except Exception:
                n_skip += 1

        n_ctrl = sum(1 for l in self.y if l == 0)
        n_park = sum(1 for l in self.y if l == 1)
        print(f"    Cargados: {len(self.x)} audios "
              f"(Control={n_ctrl}, Parkinson={n_park}"
              f"{f', omitidos={n_skip}' if n_skip else ''})")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx]).unsqueeze(-1),   # (seq_len, 1)
                torch.tensor(self.y[idx], dtype=torch.long))


# ======================================================================
#  BUCLE DE ENTRENAMIENTO
# ======================================================================
def train_one_epoch(net, loader, optimizer, criterion, device):
    net.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = net(X)
        loss = criterion(pred, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
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


def train_fold(net, train_loader, val_loader, device,
               fold_idx, results_dir, cfg):
    """
    Entrena una fold completa.
    Guarda el checkpoint con mejor val_loss (más fiable que val_acc con pocos datos).
    Incluye early stopping y ReduceLROnPlateau.
    cfg: dict con LR, N_EPOCHS, WEIGHT_DECAY, EARLY_STOP_PATIENCE
    """
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=cfg['LR'],
                                 weight_decay=cfg['WEIGHT_DECAY'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=15, verbose=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss    = float('inf')
    patience_counter = 0
    model_path       = results_dir / f'best_model_fold{fold_idx}.pt'

    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(cfg['N_EPOCHS']):
        tr_loss, tr_acc = train_one_epoch(net, train_loader, optimizer, criterion, device)
        v_loss,  v_acc  = eval_epoch(net, val_loader, criterion, device)
        scheduler.step(v_loss)

        train_losses.append(tr_loss); train_accs.append(tr_acc)
        val_losses.append(v_loss);   val_accs.append(v_acc)

        is_best = v_loss < best_val_loss
        if is_best:
            best_val_loss    = v_loss
            patience_counter = 0
            torch.save(net.state_dict(), model_path)
        else:
            patience_counter += 1

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == cfg['N_EPOCHS'] - 1:
            mark = '  ← best' if is_best else ''
            print(f"    Epoch [{epoch+1:3d}/{cfg['N_EPOCHS']}] "
                  f"Train: loss={tr_loss:.4f}, acc={tr_acc:.1f}%  |  "
                  f"Val: loss={v_loss:.4f}, acc={v_acc:.1f}%{mark}")

        if patience_counter >= cfg['EARLY_STOP_PATIENCE']:
            print(f"    Early stopping en epoch {epoch+1}")
            break

    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"    → Mejor val_loss: {best_val_loss:.4f}  ({model_path.name})")
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
        probs = torch.softmax(out, dim=1)[:, 1]
        preds = out.argmax(1)
        y_true.extend(Y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)

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

    # Confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title(f'Confusion Matrix — Fold {fold_idx}')
    plt.tight_layout()
    plt.savefig(results_dir / f'cm_fold{fold_idx}.png', dpi=150)
    plt.close()

    # ROC curve plot
    plt.figure(figsize=(5, 4))
    plt.plot(fpr_c, tpr_c, 'b-', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — Fold {fold_idx}')
    plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / f'roc_fold{fold_idx}.png', dpi=150)
    plt.close()

    metrics = dict(fold=fold_idx, accuracy=acc, precision=prec,
                   recall=rec, specificity=spec, f1=f1, auc=auc)
    return metrics, np.array(y_true), np.array(y_pred), np.array(y_prob), fpr_c, tpr_c


# ======================================================================
#  PLOTS GLOBALES
# ======================================================================
def plot_learning_curves(train_losses, train_accs, val_losses, val_accs,
                         fold_idx, results_dir, model_name, dataset_name):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_losses, label='Train')
    ax1.plot(epochs, val_losses,   label='Validation')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, train_accs, label='Train')
    ax2.plot(epochs, val_accs,   label='Validation')
    ax2.set_title('Accuracy (%)'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f'Learning Curves — Fold {fold_idx}  [{model_name} | {dataset_name}]')
    plt.tight_layout()
    plt.savefig(results_dir / f'curves_fold{fold_idx}.png', dpi=150)
    plt.close()


def plot_mean_roc(all_tprs, mean_fpr, mean_auc, std_auc,
                  results_dir, model_name, dataset_name):
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
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Mean ROC Curve (5-Fold CV)  [{model_name} | {dataset_name}]')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(results_dir / 'mean_roc_curve.png', dpi=150)
    plt.close()


def plot_accumulated_cm(all_true, all_pred, results_dir, model_name, dataset_name):
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.title(f'Accumulated CM (5-Fold CV)  [{model_name} | {dataset_name}]')
    plt.tight_layout()
    plt.savefig(results_dir / 'accumulated_cm.png', dpi=150)
    plt.close()
    print("\nMatriz de Confusión Acumulada (5 folds):")
    print(cm)
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES))


def print_summary(metrics_df, model_name, dataset_name, k_folds):
    mean = metrics_df.drop('fold', axis=1).mean()
    std  = metrics_df.drop('fold', axis=1).std()
    print("\n" + "="*55)
    print(f"  RESUMEN FINAL — {k_folds}-Fold CV  [{model_name} | {dataset_name}]")
    print("="*55)
    print(f"  {'Métrica':<16} {'Media':>8} {'Std':>8}")
    print(f"  {'─'*34}")
    for col in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']:
        print(f"  {col.capitalize():<16} {mean[col]:>8.4f} {std[col]:>8.4f}")
    print("="*55)
    print("\n  Detalle por fold:")
    cols = ['fold', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
    print(metrics_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def save_results(metrics_df, results_dir, model_name, dataset_name):
    metrics_df.to_csv(results_dir / 'per_fold_metrics.csv', index=False)
    mean_row = metrics_df.drop('fold', axis=1).mean().to_dict()
    std_row  = metrics_df.drop('fold', axis=1).std().to_dict()
    summary  = {f'mean_{k}': v for k, v in mean_row.items()}
    summary.update({f'std_{k}': v for k, v in std_row.items()})
    summary['model']   = model_name
    summary['dataset'] = dataset_name
    pd.DataFrame([summary]).to_csv(results_dir / 'summary_metrics.csv', index=False)
