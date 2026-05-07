"""
Familia 3: Transformer-based Models for Raw Audio
==================================================
Modelos de atencion global para senales de audio crudas.

Modelos disponibles:
  - TemporalTransformer   : Transformer encoder aplicado directamente a la waveform
  - CNNTransformerHybrid  : Front-end CNN + Transformer encoder (mejor para audio largo)
"""

import torch
import torch.nn as nn
import math


# ──────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Codificacion posicional sinusoidal estandar."""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.drop(x)


# ══════════════════════════════════════════════
# MODELO 3A: Temporal Transformer
# ══════════════════════════════════════════════
class TemporalTransformer(nn.Module):
    """
    Transformer encoder aplicado a audio crudo (temporal Transformer).
    La waveform se divide en parches (patches) de longitud fija.

    Hiperparametros:
        patch_size   (int) : Muestras por patch. Default 64.
        d_model      (int) : Dimension del embedding. Default 128.
        nhead        (int) : Cabezas de atencion. Default 4.
        num_layers   (int) : Capas del encoder Transformer. Default 4.
        dim_feedforward (int): Dimension de la capa FFN. Default 256.
        dropout      (float): Dropout. Default 0.1.
        num_classes  (int) : Clases de salida. Default 2.
    """
    def __init__(self, patch_size=64, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, dropout=0.1, num_classes=2):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, T)  →  eliminar canal
        x = x.squeeze(1)                       # (B, T)
        T = x.size(1)
        # Recortar para que T sea multiplo de patch_size
        n_patches = T // self.patch_size
        x = x[:, :n_patches * self.patch_size]
        x = x.view(x.size(0), n_patches, self.patch_size)  # (B, n_patches, patch_size)
        x = self.patch_embed(x)                             # (B, n_patches, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)                             # (B, n_patches, d_model)
        x = x.mean(dim=1)                                   # Global average pooling
        return self.classifier(x)


# ══════════════════════════════════════════════
# MODELO 3B: CNN + Transformer Hybrid
# ══════════════════════════════════════════════
class CNNTransformerHybrid(nn.Module):
    """
    Front-end CNN (extrae features locales) + Transformer encoder (atencion global).
    Arquitectura CNN+Transformer hibrida, muy efectiva en audio biomedico.

    Hiperparametros:
        cnn_channels  (list): Canales de cada bloque CNN front-end. Default [32, 64, 128].
        d_model       (int) : Dimension del Transformer. Default 128.
        nhead         (int) : Cabezas de atencion. Default 4.
        num_layers    (int) : Capas del encoder. Default 3.
        dim_feedforward (int): Dim FFN. Default 256.
        dropout       (float): Dropout. Default 0.1.
        num_classes   (int) : Clases de salida. Default 2.
    """
    def __init__(self, cnn_channels=None, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, num_classes=2):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        # ── Front-end CNN ──────────────────────
        cnn_layers = []
        in_ch = 1
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(4),
            ]
            in_ch = out_ch
        self.cnn_frontend = nn.Sequential(*cnn_layers)

        # Proyeccion al espacio del Transformer
        self.proj = nn.Linear(in_ch, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # ── Transformer Encoder ────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, T)
        feat = self.cnn_frontend(x)       # (B, C, T')
        feat = feat.permute(0, 2, 1)      # (B, T', C)
        feat = self.proj(feat)            # (B, T', d_model)
        feat = self.pos_enc(feat)
        feat = self.transformer(feat)     # (B, T', d_model)
        feat = feat.mean(dim=1)           # Global average pooling
        return self.classifier(feat)


TRANSFORMER_MODELS = {
    "temporal_transformer":   TemporalTransformer,
    "cnn_transformer_hybrid": CNNTransformerHybrid,
}
