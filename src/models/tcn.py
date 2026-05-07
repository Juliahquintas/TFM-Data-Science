"""
Familia 2: Temporal Convolutional Architectures (TCN / WaveNet-like)
=====================================================================
Modela dependencias temporales largas sin recurrencia ni atencion.

Modelos disponibles:
  - TCN          : Temporal Convolutional Network (causal + residual + dropout)
  - WaveNetLike  : Dilated causal convolutions + gated activation + skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ──────────────────────────────────────────────
# Bloque TCN: Causal Residual Block
# ──────────────────────────────────────────────
class CausalResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           dilation=dilation, padding=self.pad))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           dilation=dilation, padding=self.pad))
        self.bn = nn.BatchNorm1d(n_outputs)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def _chomp(self, x):
        return x[:, :, :-self.pad].contiguous() if self.pad > 0 else x

    def forward(self, x):
        out = self.drop(F.relu(self._chomp(self.conv1(x))))
        out = self.drop(F.relu(self._chomp(self.conv2(out))))
        out = self.bn(out)
        res = self.downsample(x) if self.downsample is not None else x
        return F.relu(out + res)


# ══════════════════════════════════════════════
# MODELO 2A: TCN
# ══════════════════════════════════════════════
class TCN(nn.Module):
    """
    Temporal Convolutional Network.

    Hiperparametros:
        num_channels (list): Canales por bloque. Default [64, 64, 64, 64].
        kernel_size  (int) : Tamanyo del kernel causal. Default 3.
        dropout      (float): Dropout. Default 0.2.
        num_classes  (int) : Clases de salida. Default 2.
    """
    def __init__(self, num_channels=None, kernel_size=3, dropout=0.2, num_classes=2):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(CausalResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.network(x)))


# ──────────────────────────────────────────────
# Bloque WaveNet: Gated Activation + Skip
# ──────────────────────────────────────────────
class WaveNetBlock(nn.Module):
    def __init__(self, res_ch, skip_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv1d(res_ch, res_ch, kernel_size, dilation=dilation, padding=self.pad)
        self.gate_conv   = nn.Conv1d(res_ch, res_ch, kernel_size, dilation=dilation, padding=self.pad)
        self.res_conv    = nn.Conv1d(res_ch, res_ch, 1)
        self.skip_conv   = nn.Conv1d(res_ch, skip_ch, 1)

    def forward(self, x):
        f = torch.tanh(self.filter_conv(x)[:, :, :-self.pad] if self.pad > 0 else self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x)[:, :, :-self.pad] if self.pad > 0 else self.gate_conv(x))
        act = f * g
        skip = self.skip_conv(act)
        t = act.size(2)
        res = self.res_conv(act) + x[:, :, -t:]
        return res, skip


# ══════════════════════════════════════════════
# MODELO 2B: WaveNet-like
# ══════════════════════════════════════════════
class WaveNetLike(nn.Module):
    """
    Arquitectura estilo WaveNet.

    Hiperparametros:
        residual_channels (int): Canales residuales. Default 32.
        skip_channels     (int): Canales de skip. Default 64.
        n_layers          (int): Numero de bloques. Default 8.
        kernel_size       (int): Kernel causal. Default 2.
        num_classes       (int): Clases. Default 2.
    """
    def __init__(self, residual_channels=32, skip_channels=64, n_layers=8,
                 kernel_size=2, num_classes=2):
        super().__init__()
        self.stem = nn.Conv1d(1, residual_channels, 1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, skip_channels, kernel_size, 2 ** (i % 9))
            for i in range(n_layers)
        ])
        self.post = nn.Sequential(nn.ReLU(), nn.Conv1d(skip_channels, skip_channels, 1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(skip_channels, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.stem(x)
        skip_sum = None
        for block in self.blocks:
            out, skip = block(out)
            skip_sum = skip if skip_sum is None else skip_sum[:, :, -skip.size(2):] + skip
        return self.classifier(self.pool(self.post(skip_sum)))


TCN_MODELS = {
    "tcn":          TCN,
    "wavenet_like": WaveNetLike,
}
