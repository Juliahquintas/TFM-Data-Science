"""
Audio Dataset Loader para TFM Parkinson
========================================
Carga audio desde data_splits.json (particion subject-wise K-Fold ya generada).
Admite tanto NeuroVoz como PC-GITA con la misma interfaz.

Uso tipico:
    from src.training.audio_dataset import ParkinsonAudioDataset, get_dataloaders

    train_loader, val_loader, test_loader = get_dataloaders(
        splits_json  = "data/data_splits.json",
        dataset_name = "neurovoz",   # o "pc-gita"
        fold_index   = 0,            # 0..4 para 5-Fold CV
        data_root    = "data/processed",
        sr           = 22050,
        duration     = 0.48891,
        batch_size   = 32,
    )
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa


# ──────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────

def _get_label_from_path(filepath: str) -> int:
    """
    Determina la etiqueta binaria a partir de la ruta del archivo.
    Control/Healthy → 0   |   Patologicas/PD → 1
    """
    fp = filepath.replace("\\", "/").lower()
    if "control" in fp or "healthy" in fp:
        return 0
    elif "patologicas" in fp or "parkinson" in fp or "/pd/" in fp:
        return 1
    raise ValueError(f"No se puede determinar la etiqueta para: {filepath}")


def _load_waveform(filepath: str, sr: int, n_samples: int) -> np.ndarray:
    """
    Carga un archivo WAV, lo remuestrea a `sr` Hz y lo ajusta a `n_samples` muestras
    (padding con ceros o recorte si es necesario).
    """
    try:
        wav, _ = librosa.load(filepath, sr=sr, mono=True)
    except Exception as e:
        raise IOError(f"Error al cargar {filepath}: {e}")

    # Ajuste a longitud fija
    if len(wav) < n_samples:
        wav = np.pad(wav, (0, n_samples - len(wav)), mode='constant')
    else:
        wav = wav[:n_samples]

    return wav.astype(np.float32)


# ──────────────────────────────────────────────
# Dataset principal
# ──────────────────────────────────────────────

class ParkinsonAudioDataset(Dataset):
    """
    Dataset de audio para clasificacion binaria Parkinson/Control.

    Args:
        file_list   (list of str) : Rutas relativas a la carpeta `data_root`.
        data_root   (str)         : Raiz del directorio de datos procesados.
        sr          (int)         : Sample rate objetivo (Hz). Default 22050.
        duration    (float)       : Duracion objetivo en segundos. Default 0.48891.
        augment     (bool)        : Aplica data augmentation (solo en train). Default False.
    """
    def __init__(self, file_list, data_root, sr=22050, duration=0.48891, augment=False):
        self.data_root = data_root
        self.sr        = sr
        self.n_samples = int(sr * duration)
        self.augment   = augment

        # Construye la lista (ruta_absoluta, etiqueta)
        self.samples = []
        for rel_path in file_list:
            # Normalizar separadores
            rel_path_clean = rel_path.replace("\\", os.sep).replace("/", os.sep)
            abs_path = os.path.join(data_root, rel_path_clean)
            label    = _get_label_from_path(rel_path)
            self.samples.append((abs_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        wav = _load_waveform(filepath, self.sr, self.n_samples)

        if self.augment:
            wav = self._augment(wav)

        # Formato (1, T) para Conv1d
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        return wav_tensor, label

    def _augment(self, wav: np.ndarray) -> np.ndarray:
        """
        Data augmentation suave para audio de voz patologica.
        - Ruido gaussiano leve
        - Desplazamiento temporal aleatorio
        - Escalado de amplitud aleatorio
        """
        # Ruido gaussiano
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.005, wav.shape)
            wav = wav + noise

        # Desplazamiento temporal (max 10% de la senyal)
        if np.random.rand() < 0.5:
            shift = int(np.random.uniform(-0.1, 0.1) * len(wav))
            wav = np.roll(wav, shift)

        # Escalado de amplitud
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            wav = wav * scale

        # Re-normalizar
        max_amp = np.abs(wav).max()
        if max_amp > 0:
            wav = wav / max_amp

        return wav.astype(np.float32)


# ──────────────────────────────────────────────
# Factory function de DataLoaders
# ──────────────────────────────────────────────

def get_dataloaders(
    splits_json: str,
    dataset_name: str,
    fold_index: int,
    data_root: str,
    sr: int = 22050,
    duration: float = 0.48891,
    batch_size: int = 32,
    num_workers: int = 0,
    augment_train: bool = True,
):
    """
    Crea los DataLoaders de Train / Val / Test para un fold especifico.

    Args:
        splits_json   : Ruta al JSON generado por data_split.py.
        dataset_name  : "neurovoz" o "pc-gita".
        fold_index    : Indice del fold (0 a K-1).
        data_root     : Carpeta raiz donde estan los audios procesados.
        sr            : Sample rate. Default 22050.
        duration      : Duracion en segundos. Default 0.48891.
        batch_size    : Tamano del mini-batch. Default 32.
        num_workers   : Workers para DataLoader. Default 0 (Windows-safe).
        augment_train : Si aplicar augmentation al train set. Default True.

    Returns:
        train_loader, val_loader, test_loader
    """
    with open(splits_json, "r") as f:
        splits = json.load(f)

    fold_data = splits[dataset_name][fold_index]
    train_files = fold_data["train_files"]
    val_files   = fold_data["val_files"]
    test_files  = fold_data["test_files"]

    print(f"[DataLoader] Dataset: {dataset_name.upper()} | Fold {fold_index + 1}")
    print(f"  Train: {len(train_files)} archivos | Val: {len(val_files)} | Test: {len(test_files)}")

    train_ds = ParkinsonAudioDataset(train_files, data_root, sr, duration, augment=augment_train)
    val_ds   = ParkinsonAudioDataset(val_files,   data_root, sr, duration, augment=False)
    test_ds  = ParkinsonAudioDataset(test_files,  data_root, sr, duration, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_class_weights(file_list: list) -> torch.Tensor:
    """
    Calcula pesos de clase para manejar el desbalanceo.
    Retorna tensor [w_control, w_parkinson] para usar en nn.CrossEntropyLoss(weight=...).
    """
    labels = [_get_label_from_path(f) for f in file_list]
    n_total = len(labels)
    n_pos   = sum(labels)
    n_neg   = n_total - n_pos
    w0 = n_total / (2 * n_neg) if n_neg > 0 else 1.0
    w1 = n_total / (2 * n_pos) if n_pos > 0 else 1.0
    return torch.tensor([w0, w1], dtype=torch.float32)
