"""
train_parkinson_folds.py — Punto de entrada y configuración
═══════════════════════════════════════════════════════════════════════════════

ESTRUCTURA DEL PROYECTO
────────────────────────
CDIL_CNN/
├── train_parkinson_folds.py   ← ESTE ARCHIVO — configura y ejecuta todo
├── time_train_folds.py        ← bucle de entrenamiento y evaluación
└── Models/
    ├── net_conv.py            ← arquitectura CDIL CNN (no tocar)
    └── utils.py               ← DatasetCreator, seed_everything

TUS DATOS
─────────
  processed/
    ├── neurovoz/Control/A/0105_A1.wav
    ├── neurovoz/Control/A/0105_A2.wav
    └── ...
  data_splits.json             ← tu JSON con la división de folds

FORMATO DEL JSON (tu fichero real)
────────────────────────────────────
{
  "metadata": {"K_folds": 5},
  "neurovoz": [                       ← lista de 5 folds
    {
      "train_files": ["neurovoz\\Control\\A\\0105_A1.wav", ...],
      "test_files":  ["neurovoz\\Control\\A\\0129_A1.wav", ...]
    },
    ...
  ],
  "pc-gita": [ ... ]                  ← misma estructura
}

LABEL
──────
  La etiqueta se extrae automáticamente del path:
    "Control"   → 0 (HC)
    cualquier otra carpeta → 1 (Parkinson)

VALIDACIÓN
──────────
  El JSON solo tiene train/test. Se reserva automáticamente el
  VAL_SPLIT (20 % por defecto) de los sujetos de train como validación.

USO
───
  python train_parkinson_folds.py
  python train_parkinson_folds.py --dataset neurovoz --seed 42 --saving mis_resultados
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, shutil, logging, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import librosa
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from time_train_folds import TrainModel, plot_metrics, evaluate
from Models.net_conv   import CONV
from Models.utils      import seed_everything, DatasetCreator


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN — ajusta estas variables a tu entorno
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, '..', '..', 'data', 'processed')
)       # carpeta con los .wav de audio
FOLDS_JSON = os.path.normpath(
    os.path.join(SCRIPT_DIR, '..', '..', 'data', 'data_splits.json')
)

# ── Hiperparámetros (idénticos a Marta Rey) ───────────────────────────────────
BATCH_SIZE  = 256
N_EPOCHS    = 200
CNN_HIDDEN  = 32    # canales por capa
KERNEL_SIZE = 3
LR          = 1e-4

# ── Validación automática desde train ────────────────────────────────────────
#    Porcentaje de SUJETOS de train reservados para validación.
VAL_SPLIT   = 0.20  # 20 %

# ══════════════════════════════════════════════════════════════════════════════

# ─── Argumentos CLI (opcionales, sobreescriben las variables de arriba) ───────
parser = argparse.ArgumentParser(description='CDIL CNN — Parkinson Detection')
parser.add_argument('--dataset', type=str, default='neurovoz',
                    choices=['neurovoz', 'pc-gita'],
                    help='Dataset a entrenar: neurovoz | pc-gita')
parser.add_argument('--model',   type=str, default='CDIL',
                    choices=['CDIL', 'DIL', 'TCN', 'CNN'],
                    help='Variante del modelo')
parser.add_argument('--seed',    type=int, default=1)
parser.add_argument('--saving',  type=str, default='results',
                    help='Carpeta de salida para resultados')
args = parser.parse_args()

DATASET = args.dataset
MODEL   = args.model
SEED    = args.seed
SAVING  = args.saving
TASK    = 'Parkinson'


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def path_to_wav(file_path: str) -> str:
    """
    Convierte el path del JSON (Windows, .wav) a la ruta real en data/processed.
    Ej: 'neurovoz\\Control\\A\\0105_A1.wav' →
        '.../data/processed/neurovoz/Control/A/0105_A1.wav'
    """
    rel_path = file_path.replace('\\', '/').lstrip('/\\')
    return os.path.join(PROCESSED_DIR, *rel_path.split('/'))


def get_label(file_path: str) -> int:
    """
    Extrae la etiqueta del path:
      'Control'  → 0  (sano)
      cualquier otro directorio → 1  (Parkinson)
    """
    parts = file_path.replace('\\', '/').split('/')
    # La carpeta de clase es el segundo componente (dataset/CLASE/vocal/file)
    return 0 if 'Control' in parts else 1


def load_files(file_list):
    """
    Carga los .wav correspondientes a una lista de paths del JSON.
    Devuelve (X array, y array, lista de paths no encontrados).
    """
    arrays, labels, missing = [], [], []
    for fp in file_list:
        wav_path = path_to_wav(fp)
        if not os.path.exists(wav_path):
            missing.append(wav_path)
            continue
        arr, _ = librosa.load(wav_path, sr=None, mono=True)
        arrays.append(arr)
        labels.append(get_label(fp))
    if missing:
        print(f'  ⚠ {len(missing)} archivos no encontrados en {PROCESSED_DIR}/')
        print(f'    Primer ejemplo: {missing[0]}')
    if arrays:
        max_len = max(len(a) for a in arrays)
        arrays = [a if len(a) == max_len else np.pad(a, (0, max_len - len(a)))
                  for a in arrays]
        X = np.stack(arrays)
    else:
        X = np.array([])
    y = np.array(labels)
    return X, y


def to_dataloader(X, y, batch_size, shuffle):
    """Convierte arrays numpy a DataLoader de PyTorch."""
    t = torch.tensor(X).float()
    if t.ndim == 2:              # (N, seq_len) → (N, seq_len, 1)
        t = t.unsqueeze(-1)
    N = len(y)
    t = t.view(N, -1, 1)
    l = torch.tensor(y).long()
    return DataLoader(DatasetCreator(t, l), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False), t, l


def split_val_from_train(X_train, y_train, train_files, val_split, seed):
    """
    Divide el conjunto de train en train/val respetando que los sujetos
    no se mezclen entre splits (subject-independent split).
    Extrae el ID de sujeto del nombre de fichero.
    """
    # Extraer IDs de sujeto del path (primer token del nombre de archivo)
    subject_ids = [os.path.basename(fp.replace('\\', '/')).split('_')[0]
                   for fp in train_files]
    subject_ids = np.array(subject_ids)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(gss.split(X_train, y_train, groups=subject_ids))
    return (X_train[train_idx], y_train[train_idx],
            X_train[val_idx],   y_train[val_idx])


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARAR CARPETA DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
if os.path.exists(SAVING):
    shutil.rmtree(SAVING)
os.makedirs(f'{SAVING}/logs',   exist_ok=True)
os.makedirs(f'{SAVING}/models', exist_ok=True)
print(f"Carpeta de resultados '{SAVING}/' creada.")

# ── Cargar JSON ───────────────────────────────────────────────────────────────
with open(FOLDS_JSON) as f:
    splits = json.load(f)

folds_data = splits[DATASET]          # lista de 5 dicts (un fold cada uno)
n_folds    = len(folds_data)
print(f'Dataset: {DATASET} | Folds: {n_folds} | Modelo: {MODEL} | Seed: {SEED}')
print(f'Dispositivo: {"CUDA (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_names = ['Control', 'Parkinson']

seed_everything(SEED)

# ── Acumuladores globales ─────────────────────────────────────────────────────
all_predictions = []
all_true_labels = []
all_metrics     = []
tprs            = []
mean_fpr        = np.linspace(0, 1, 100)


# ══════════════════════════════════════════════════════════════════════════════
#  BUCLE DE FOLDS
# ══════════════════════════════════════════════════════════════════════════════
for fold_idx, fold in enumerate(folds_data, start=1):
    print(f'\n{"="*65}')
    print(f'  FOLD {fold_idx}/{n_folds}')
    print(f'{"="*65}')

    # ── Cargar arrays ─────────────────────────────────────────────────────────
    X_train_all, y_train_all = load_files(fold['train_files'])
    X_test,      y_test      = load_files(fold['test_files'])

    if len(X_train_all) == 0 or len(X_test) == 0:
        print(f'  ✗ Fold {fold_idx} sin datos, saltando.')
        continue

    # ── Separar validación de train (subject-independent) ────────────────────
    X_train, y_train, X_val, y_val = split_val_from_train(
        X_train_all, y_train_all, fold['train_files'], VAL_SPLIT, SEED
    )

    print(f'  train={len(y_train)} | val={len(y_val)} | test={len(y_test)}')
    print(f'  Clases train → {np.unique(y_train, return_counts=True)}')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    trainloader, ft, lt = to_dataloader(X_train, y_train, BATCH_SIZE, shuffle=True)
    valloader,   _,  _  = to_dataloader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    testloader,  _,  _  = to_dataloader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    SEQ_LEN   = ft.shape[1]
    N_CLASSES = len(torch.unique(lt))
    LAYER     = int(np.log2(SEQ_LEN))
    print(f'  seq_len={SEQ_LEN} | capas={LAYER} | clases={N_CLASSES}')

    # ── Modelo (exacto Marta Rey) ─────────────────────────────────────────────
    net = CONV(
        task        = TASK,
        model       = MODEL,
        input_size  = 1,
        output_size = N_CLASSES,
        num_channels= [CNN_HIDDEN] * LAYER,
        kernel_size = KERNEL_SIZE
    ).to(device)

    n_params  = sum(p.numel() for p in net.parameters() if p.requires_grad)
    run_name  = f'{TASK}_P{n_params}_{MODEL}_S{SEED}_L{LAYER}_H{CNN_HIDDEN}'
    print(f'  Parámetros: {n_params:,}  |  {run_name}')

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file   = f'{SAVING}/logs/fold{fold_idx}_{run_name}.txt'
    model_file = f'{SAVING}/models/fold{fold_idx}_{run_name}.ph'
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    loginf = logging.info
    loginf(run_name)

    # ── Optimizador y loss (idénticos a Marta Rey) ────────────────────────────
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn   = torch.nn.CrossEntropyLoss(reduction='sum')

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    tr_losses, tr_accs, vl_losses, vl_accs = TrainModel(
        net=net, device=device,
        trainloader=trainloader, valloader=valloader, testloader=testloader,
        n_epochs=N_EPOCHS, optimizer=optimizer, loss_fn=loss_fn,
        loginf=loginf, file_name=model_file
    )

    plot_metrics(N_EPOCHS, tr_losses, tr_accs, vl_losses, vl_accs,
                 f'{SAVING}/loss_fold{fold_idx}.png',
                 f'{SAVING}/accuracy_fold{fold_idx}.png')

    # ── Evaluación con el mejor checkpoint ────────────────────────────────────
    net.load_state_dict(torch.load(model_file, map_location=device))
    accuracy, precision, recall, f1, auc_score, specificity, \
        fpr, tpr, _, y_true, y_pred = evaluate(
            net, device, testloader, class_names, SAVING, fold_idx, loss_fn
        )

    all_predictions.extend(y_pred)
    all_true_labels.extend(y_true)
    all_metrics.append([accuracy, precision, recall, f1, auc_score, specificity])

    interp_tpr    = np.interp(mean_fpr, fpr, tpr); interp_tpr[0] = 0.
    tprs.append(interp_tpr)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTADOS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
if all_metrics:
    mean_m = np.mean(all_metrics, axis=0)
    std_m  = np.std(all_metrics,  axis=0)
    cv_m   = std_m / mean_m

    labels_m = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity']
    print('\n' + '='*65)
    print(f'RESULTADOS MEDIOS — {DATASET} — {MODEL}')
    print('='*65)
    for name, mean, std, cv in zip(labels_m, mean_m, std_m, cv_m):
        print(f'  {name:<12}: {mean:.4f} ± {std:.4f}  (CV={cv:.4f})')

    # CSV de métricas medias
    pd.DataFrame({'Metrics': labels_m, 'Mean': mean_m, 'Std': std_m, 'CV': cv_m})\
      .to_csv(f'{SAVING}/mean_test_metrics.csv', index=False)

    # Matriz de confusión acumulada
    cm_all = confusion_matrix(all_true_labels, all_predictions)
    print('\n' + classification_report(all_true_labels, all_predictions, target_names=class_names))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Accumulated Confusion Matrix')
    plt.savefig(f'{SAVING}/accumulated_confusion_matrix.png', dpi=300); plt.close()

    # Curva ROC media
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.
    std_tpr  = np.std(tprs,  axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_fpr, mean_tpr, color='b', lw=1,
            label=f'Mean ROC (AUC={mean_m[4]:.2f} ± {std_m[4]:.2f})')
    ax.fill_between(mean_fpr, np.maximum(mean_tpr-std_tpr, 0),
                                np.minimum(mean_tpr+std_tpr, 1),
                    color='grey', alpha=0.2, label='± 1 std. dev.')
    ax.plot([0,1],[0,1], color='black', lw=1, linestyle='--')
    ax.grid(); ax.set(xlabel='FPR', ylabel='TPR', title='Mean ROC Curve')
    ax.legend(loc='lower right')
    plt.savefig(f'{SAVING}/mean_roc_curve.png', dpi=300); plt.close()
else:
    print('\n✗ No se obtuvieron métricas porque ningún fold tuvo datos válidos.')
    pd.DataFrame(columns=['Metrics', 'Mean', 'Std', 'CV'])\
      .to_csv(f'{SAVING}/mean_test_metrics.csv', index=False)

print(f'\n✓ Todos los resultados en: {SAVING}/')
