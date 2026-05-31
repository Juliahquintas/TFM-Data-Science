"""
model_wavenet.py — WaveNet-style Classifier para detección de Parkinson
════════════════════════════════════════════════════════════════════════
Basado en Van den Oord et al., "WaveNet: A Generative Model for Raw Audio"
(arXiv:1609.03499), adaptado para clasificación binaria.

Novedades respecto al CDIL-CNN baseline:
  1. Gated Activation Unit: tanh(W_f*x) ⊙ σ(W_g*x)
     Permite inhibición selectiva de componentes frecuenciales,
     más expresivo que ReLU para señales de voz oscilatorias.
  2. Skip connections acumulativas: la suma de las salidas de TODAS
     las capas alimenta el clasificador (representación multi-escala).
  3. Padding: causal (zeros, unilateral) como en el WaveNet original.
     Alternativa no causal disponible activando CIRCULAR_PADDING=True.

Parámetros a configurar en config.py (mismos que CDIL-CNN):
  NHID, KERNEL_SIZE, DROPOUT, N_EPOCHS, LR, BATCH_SIZE
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


MAX_LAYERS      = 8      # mismo límite que CDIL-CNN
CIRCULAR_PADDING = False  # True → no causal (como CDIL); False → causal (WaveNet original)


class GatedBlock(nn.Module):
    """
    Bloque WaveNet con:
      · Gated Activation Unit: tanh(filter) ⊙ sigmoid(gate)
      · Residual connection (add input to gated output)
      · Skip connection output (se acumula externamente)

    Dos convoluciones independientes por bloque (filter y gate),
    ambas con la misma dilatación y padding.
    """
    def __init__(self, c_in: int, c_out: int,
                 kernel_size: int, padding: int,
                 dilation: int, pad_mode: str):
        super().__init__()

        # Rama filter (tanh)
        self.conv_filter = weight_norm(
            nn.Conv1d(c_in, c_out, kernel_size,
                      padding=padding, dilation=dilation,
                      padding_mode=pad_mode)
        )
        # Rama gate (sigmoid)
        self.conv_gate = weight_norm(
            nn.Conv1d(c_in, c_out, kernel_size,
                      padding=padding, dilation=dilation,
                      padding_mode=pad_mode)
        )

        # Proyección residual si cambian los canales
        self.res = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else None

        # En TCN causal hay que recortar el padding derecho
        self.causal  = (pad_mode == 'zeros')
        self.crop    = padding if self.causal else 0

    def forward(self, x: torch.Tensor):
        f = self.conv_filter(x)
        g = self.conv_gate(x)

        # Recorte causal (elimina padding derecho añadido)
        if self.causal and self.crop > 0:
            f = f[:, :, :-self.crop].contiguous()
            g = g[:, :, :-self.crop].contiguous()

        # Gated Activation Unit
        gated = torch.tanh(f) * torch.sigmoid(g)      # ← novedad vs CDIL

        # Residual
        res = x if self.res is None else self.res(x)
        out = gated + res

        # Devuelve (output_residual, skip_contribution)
        # El skip es la salida gated antes de sumar el residual,
        # proyectada a la dimensión de canales de salida.
        return out, gated


class WaveNetClassifier(nn.Module):
    """
    Clasificador WaveNet para señales de audio crudas.

    Flujo:
      (B, T, 1)
        → permuta → (B, 1, T)
        → L × GatedBlock  [dil = 2^i]
        → suma de skip connections de todas las capas  ← novedad vs CDIL
        → ReLU → GAP → Linear → (B, n_classes)

    Parámetros:
      seq_len    : longitud de la señal de entrada
      nhid       : canales por capa
      kernel_size: tamaño del kernel
      dropout    : dropout tras la suma de skips (0.0 = desactivado)
      n_classes  : número de clases de salida
    """
    def __init__(self, seq_len: int,
                 nhid: int        = 32,
                 kernel_size: int = 3,
                 dropout: float   = 0.0,
                 n_classes: int   = 2):
        super().__init__()

        n_layers = min(int(np.log2(seq_len)), MAX_LAYERS)
        pad_mode = 'zeros' if not CIRCULAR_PADDING else 'circular'

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            c_in  = 1    if i == 0 else nhid
            dil   = 2 ** i
            if CIRCULAR_PADDING:
                pad = int(dil * (kernel_size - 1) / 2)   # simétrico
            else:
                pad = dil * (kernel_size - 1)             # causal
            self.blocks.append(
                GatedBlock(c_in, nhid, kernel_size, pad, dil, pad_mode)
            )

        self.dropout    = nn.Dropout(p=dropout) if dropout > 0.0 else None
        self.relu       = nn.ReLU()
        self.classifier = nn.Linear(nhid, n_classes)

        self.n_layers = n_layers
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).float()      # (B, 1, T)

        skip_sum = None
        for block in self.blocks:
            x, skip = block(x)
            # Acumulación de skip connections (suma multi-escala)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        # Post-procesado de la suma de skips
        out = self.relu(skip_sum)
        if self.dropout is not None:
            out = self.dropout(out)

        # Global Average Pooling
        out = torch.mean(out, dim=2)        # (B, nhid)
        return self.classifier(out)         # (B, n_classes)