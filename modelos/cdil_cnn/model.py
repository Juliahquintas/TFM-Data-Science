"""
model.py — Arquitectura CDIL CNN para detección de Parkinson
═════════════════════════════════════════════════════════════
Basado en Marta Rey-Paredes et al., 2024.
Adaptaciones respecto al original:
  · Número de capas limitado a min(log2(seq_len), MAX_LAYERS) para
    evitar sobreajuste con datasets pequeños.
  · Inicialización de pesos por defecto de PyTorch (el código original
    reinicializa sobre weight_norm de forma que no tiene efecto real).
  · Dropout opcional tras cada bloque (por defecto desactivado).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


MAX_LAYERS = 8   # límite de profundidad (evita redes excesivamente profundas)


class ConvBlock(nn.Module):
    """
    Bloque convolucional dilatado con conexión residual.

    Características CDIL:
      · Padding circular (los extremos de la señal se conectan entre sí,
        permitiendo contexto global sin aumentar parámetros).
      · Dilatación 2^i → campo receptivo crece exponencialmente por capa.
      · Skip connection: suma la entrada a la salida (como ResNet).
      · Dropout opcional tras la activación (desactivado por defecto).
    """
    def __init__(self, c_in: int, c_out: int,
                 kernel_size: int, padding: int, dilation: int,
                 dropout: float = 0.0):
        super().__init__()

        self.conv = weight_norm(
            nn.Conv1d(c_in, c_out, kernel_size,
                      padding=padding, dilation=dilation,
                      padding_mode='circular')   # clave de CDIL
        )

        # Proyección residual solo cuando cambia el número de canales
        self.res = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else None

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv(x))
        if self.dropout is not None:
            out = self.dropout(out)
        res = x if self.res is None else self.res(x)
        return out + res


class CDILClassifier(nn.Module):
    """
    Clasificador CDIL CNN completo.

    Flujo:
      (batch, seq_len, 1)
        → permuta → (batch, 1, seq_len)
        → L bloques ConvBlock con dil = 1, 2, 4, ..., 2^(L-1)
        → Global Average Pooling → (batch, nhid)
        → Linear → (batch, n_classes)   [logits sin softmax]

    Parámetros:
      seq_len    : longitud de la señal de entrada
      nhid       : canales por capa (feature maps)
      kernel_size: tamaño del kernel convolucional
      dropout    : tasa de dropout (0.0 = desactivado)
      n_classes  : número de clases de salida (2 para HC/PD)
    """
    def __init__(self, seq_len: int,
                 nhid: int       = 32,
                 kernel_size: int = 3,
                 dropout: float   = 0.0,
                 n_classes: int   = 2):
        super().__init__()

        # n_layers = min(int(np.log2(seq_len)), MAX_LAYERS)
        n_layers = int(np.log2(seq_len))

        layers = []
        for i in range(n_layers):
            c_in    = 1    if i == 0 else nhid
            dil     = 2 ** i
            pad     = int(dil * (kernel_size - 1) / 2)   # padding simétrico
            layers.append(ConvBlock(c_in, nhid, kernel_size, pad, dil, dropout))

        self.conv_net  = nn.Sequential(*layers)
        self.classifier = nn.Linear(nhid, n_classes)

        # Info del modelo
        self.n_layers    = n_layers
        self.n_params    = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).float()     # (batch, 1, seq_len)
        y = self.conv_net(x)               # (batch, nhid, seq_len)
        y = torch.mean(y, dim=2)           # Global Average Pooling
        return self.classifier(y)          # (batch, n_classes)
