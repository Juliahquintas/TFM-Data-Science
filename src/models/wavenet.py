import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(WaveNetBlock, self).__init__()
        # Convolución causal usando padding=dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=dilation, padding=dilation)
        self.proj = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        # Recortar para hacerla causal estricta
        out = out[:, :, :-self.conv.padding[0]]
        
        # Activación Gated (Tanh * Sigmoid)
        tanh_out = torch.tanh(out)
        sigm_out = torch.sigmoid(out)
        
        z = tanh_out * sigm_out
        z = self.proj(z)
        
        return z + residual

class WaveNet(nn.Module):
    def __init__(self, num_classes=2, num_blocks=5, channels=32):
        super(WaveNet, self).__init__()
        # Proyección inicial
        self.initial_conv = nn.Conv1d(1, channels, kernel_size=1)
        
        # Bloques dilatados
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(WaveNetBlock(channels, channels, dilation))
            
        # Clasificador final global
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(channels // 2, num_classes)
        )
        
    def forward(self, x):
        """
        x: tensor de audios de forma (batch_size, 1, seq_length)
        """
        out = self.initial_conv(x)
        
        for block in self.blocks:
            out = block(out)
            
        out = self.pool(out)             # (Batch, Channels, 1)
        out = out.squeeze(-1)            # (Batch, Channels)
        out = self.classifier(out)       # (Batch, NumClasses)
        
        return out
