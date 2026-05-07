"""
src/models/__init__.py
Registro centralizado de todos los modelos disponibles para el TFM Parkinson.
"""

from src.models.cnn_1d import CNN_MODELS
from src.models.tcn import TCN_MODELS
from src.models.transformer_audio import TRANSFORMER_MODELS

ALL_MODELS = {
    **CNN_MODELS,
    **TCN_MODELS,
    **TRANSFORMER_MODELS,
}

__all__ = ["ALL_MODELS", "CNN_MODELS", "TCN_MODELS", "TRANSFORMER_MODELS"]
