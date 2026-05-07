"""
Familia 1: 1D Convolutional Neural Networks (CNNs)
===================================================
Tres variantes para capturar patrones locales en la señal de audio cruda.

Modelos disponibles:
  - BaselineCNN1D      : CNN 1D estándar (punto de partida)
  - ResidualCNN1D      : CNN 1D profunda con residual connections
  - DilatedCNN1D       : CNN 1D con convoluciones dilatadas (receptive field ampliado)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Bloque auxiliar: Residual Block
# ─────────────────────────────────────────────
class ResidualBlock1D(nn.Module):
    """Bloque residual con dos capas Conv1D + BatchNorm + ReLU."""
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# ─────────────────────────────────────────────
# Bloque auxiliar: Dilated Conv Block
# ─────────────────────────────────────────────
class DilatedBlock1D(nn.Module):
    """Bloque de convolución dilatada con skip connection."""
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              dilation=dilation, padding=pad)
        self.bn   = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return F.relu(self.drop(self.bn(self.conv(x)))) + x


# ═══════════════════════════════════════════════
# MODELO 1A: Baseline CNN 1D
# ═══════════════════════════════════════════════
class BaselineCNN1D(nn.Module):
    """
    CNN 1D estándar — baseline de referencia.

    Hiperparámetros clave:
        n_filters    (list) : Número de filtros en cada bloque conv. Default [32, 64, 128].
        kernel_size  (int)  : Tamaño del kernel. Default 7.
        dropout      (float): Tasa de dropout. Default 0.3.
        num_classes  (int)  : Número de clases (2 para binario). Default 2.
    """
    def __init__(self, n_filters=None, kernel_size=7, dropout=0.3, num_classes=2):
        super().__init__()
        if n_filters is None:
            n_filters = [32, 64, 128]

        layers = []
        in_ch = 1
        for out_ch in n_filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(4),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.classifier  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, T)
        out = self.conv_blocks(x)
        out = self.pool(out)
        return self.classifier(out)


# ═══════════════════════════════════════════════
# MODELO 1B: Deep Residual CNN 1D
# ═══════════════════════════════════════════════
class ResidualCNN1D(nn.Module):
    """
    CNN 1D profunda con residual connections.

    Hiperparámetros clave:
        n_filters      (list) : Canales en cada stage. Default [32, 64, 128, 256].
        kernel_size    (int)  : Tamaño de kernel en los residual blocks. Default 3.
        n_res_blocks   (int)  : Número de residual blocks por stage. Default 2.
        dropout        (float): Tasa de dropout. Default 0.2.
        num_classes    (int)  : Número de clases. Default 2.
    """
    def __init__(self, n_filters=None, kernel_size=3, n_res_blocks=2,
                 dropout=0.2, num_classes=2):
        super().__init__()
        if n_filters is None:
            n_filters = [32, 64, 128, 256]

        # Capa inicial de proyección
        self.stem = nn.Sequential(
            nn.Conv1d(1, n_filters[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(n_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        stages = []
        in_ch = n_filters[0]
        for out_ch in n_filters[1:]:
            # Transición de canal
            stages.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ))
            # Residual blocks
            for _ in range(n_res_blocks):
                stages.append(ResidualBlock1D(out_ch, kernel_size, dropout))
            in_ch = out_ch

        self.stages     = nn.Sequential(*stages)
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.stages(out)
        out = self.pool(out)
        return self.classifier(out)


# ═══════════════════════════════════════════════
# MODELO 1C: Dilated CNN 1D
# ═══════════════════════════════════════════════
class DilatedCNN1D(nn.Module):
    """
    CNN 1D con convoluciones dilatadas — receptive field exponencialmente más grande
    sin aumentar parámetros.

    Hiperparámetros clave:
        n_channels   (int)         : Canales en todos los bloques dilatados. Default 64.
        kernel_size  (int)         : Tamaño del kernel. Default 3.
        dilations    (list of int) : Factores de dilatación. Default [1, 2, 4, 8, 16].
        dropout      (float)       : Tasa de dropout. Default 0.2.
        num_classes  (int)         : Número de clases. Default 2.
    """
    def __init__(self, n_channels=64, kernel_size=3, dilations=None,
                 dropout=0.2, num_classes=2):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]

        self.stem = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_channels),
            nn.ReLU(),
        )

        self.dilated_blocks = nn.Sequential(
            *[DilatedBlock1D(n_channels, kernel_size, d, dropout) for d in dilations]
        )

        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.dilated_blocks(out)
        out = self.pool(out)
        return self.classifier(out)


# ─────────────────────────────────────────────
# Registro de modelos de la familia (útil para los notebooks)
# ─────────────────────────────────────────────
CNN_MODELS = {
    "baseline_cnn1d":  BaselineCNN1D,
    "residual_cnn1d":  ResidualCNN1D,
    "dilated_cnn1d":   DilatedCNN1D,
}
