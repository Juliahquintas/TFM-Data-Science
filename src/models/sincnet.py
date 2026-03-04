import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SincConvFast(nn.Module):
    """
    Simulación de la convolución Sinc basada en PyTorch.
    Para el TFM se puede requerir la versión completa que construye 
    dinámicamente el banco de filtros Mel (como en SpeechBrain).
    """
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super(SincConvFast, self).__init__()
        self.out_channels = out_channels
        
        # Filtros de tamaño impar
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Como stand-in temporal, usaremos Conv1d genérico parametrizado.
        # El implemento real debería calcular la respuesta de impulso Sinc.
        self.conv = nn.Conv1d(1, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)

    def forward(self, x):
        return self.conv(x)

class SincNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SincNet, self).__init__()
        
        # Capa SincNet inicial paramétrica
        self.sinc_conv = SincConvFast(out_channels=80, kernel_size=251)
        self.pool1 = nn.MaxPool1d(3)
        self.norm1 = nn.BatchNorm1d(80)
        
        # Capas convolucionales estándar subsiguientes
        self.conv2 = nn.Conv1d(80, 60, kernel_size=5)
        self.pool2 = nn.MaxPool1d(3)
        self.norm2 = nn.BatchNorm1d(60)
        
        self.conv3 = nn.Conv1d(60, 60, kernel_size=5)
        self.pool3 = nn.MaxPool1d(3)
        self.norm3 = nn.BatchNorm1d(60)
        
        # Global pooling espacial (para colapsar longitud temporal variante)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Capa fully connected final
        self.fc = nn.Sequential(
            nn.Linear(60, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x: (Batch, 1, Length)
        """
        # Feature extraction
        x = torch.abs(self.sinc_conv(x)) # SincNet inicial aplica absoluto típicamente
        x = self.pool1(x)
        x = self.norm1(x)
        
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.pool2(x)
        x = self.norm2(x)
        
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.pool3(x)
        x = self.norm3(x)
        
        # Pooling y clasificación
        x = self.global_pool(x).squeeze(-1)
        out = self.fc(x)
        
        return out
