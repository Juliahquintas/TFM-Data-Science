"""
model.py — CDIL CNN + Temporal Attention Pooling
═════════════════════════════════════════════════════════════

Basado en Marta Rey-Paredes et al., 2024.

Modificaciones respecto al CDIL original:
  · Profundidad limitada a min(log2(seq_len), MAX_LAYERS)
    para evitar sobreajuste en datasets pequeños.
  · Inicialización por defecto de PyTorch.
  · Dropout opcional tras cada bloque.
  · Sustitución de Global Average Pooling por
    Temporal Attention Pooling aprendido.

La atención temporal permite que el modelo aprenda
qué regiones temporales del audio contienen información
más relevante para la clasificación.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


MAX_LAYERS = 8


class ConvBlock(nn.Module):
    """
    Bloque convolucional dilatado residual.

    Características:
      · Convolución dilatada
      · Padding circular
      · Residual connection
      · ReLU
      · Dropout opcional
    """

    def __init__(self,
                 c_in: int,
                 c_out: int,
                 kernel_size: int,
                 padding: int,
                 dilation: int,
                 dropout: float = 0.0):
        super().__init__()

        self.conv = weight_norm(
            nn.Conv1d(
                c_in,
                c_out,
                kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode='circular'
            )
        )

        # Proyección residual si cambian canales
        self.res = (
            nn.Conv1d(c_in, c_out, kernel_size=1)
            if c_in != c_out else None
        )

        self.relu = nn.ReLU()

        self.dropout = (
            nn.Dropout(dropout)
            if dropout > 0.0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.relu(self.conv(x))

        if self.dropout is not None:
            out = self.dropout(out)

        res = x if self.res is None else self.res(x)

        return out + res


class TemporalAttentionPooling(nn.Module):
    """
    Attention pooling temporal.

    Aprende pesos de atención sobre la dimensión temporal:

        z = Σ α_t h_t

    donde:
        α_t = softmax(score(h_t))

    Esto permite al modelo enfatizar regiones
    temporalmente relevantes del audio.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # MLP pequeña para calcular scores de atención
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, hidden_dim)

        Returns:
            pooled : (batch, hidden_dim)
            weights: (batch, seq_len, 1)
        """

        # Attention scores
        scores = self.attention(x)  # (batch, seq_len, 1)

        # Normalización temporal
        weights = torch.softmax(scores, dim=1)

        # Weighted sum
        pooled = torch.sum(x * weights, dim=1)

        return pooled, weights


class CDILAttentionClassifier(nn.Module):
    """
    CDIL CNN con Temporal Attention Pooling.

    Flujo:
      (batch, seq_len, 1)
        → permute → (batch, 1, seq_len)
        → bloques CDIL
        → temporal attention pooling
        → classifier

    Parámetros:
      seq_len     : longitud señal
      nhid        : nº canales ocultos
      kernel_size : tamaño kernel
      dropout     : dropout
      n_classes   : nº clases
    """

    def __init__(self,
                 seq_len: int,
                 nhid: int = 32,
                 kernel_size: int = 3,
                 dropout: float = 0.0,
                 n_classes: int = 2):

        super().__init__()

        # Profundidad limitada
        n_layers = min(int(np.log2(seq_len)), MAX_LAYERS)

        layers = []

        for i in range(n_layers):

            c_in = 1 if i == 0 else nhid

            dilation = 2 ** i

            # Padding simétrico circular
            padding = int(
                dilation * (kernel_size - 1) / 2
            )

            layers.append(
                ConvBlock(
                    c_in=c_in,
                    c_out=nhid,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.conv_net = nn.Sequential(*layers)

        # NUEVO:
        # Attention pooling temporal
        self.attention_pool = TemporalAttentionPooling(nhid)

        # Clasificador final
        self.classifier = nn.Linear(nhid, n_classes)

        # Información modelo
        self.n_layers = n_layers

        self.n_params = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False):

        # (batch, seq_len, 1)
        # → (batch, 1, seq_len)
        x = x.permute(0, 2, 1).float()

        # CDIL blocks
        y = self.conv_net(x)
        # (batch, nhid, seq_len)

        # → (batch, seq_len, nhid)
        y = y.permute(0, 2, 1)

        # Attention pooling
        pooled, attn_weights = self.attention_pool(y)

        # Classifier
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attn_weights

        return logits