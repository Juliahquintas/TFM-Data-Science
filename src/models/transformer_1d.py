import torch
import torch.nn as nn

class Transformer1D(nn.Module):
    def __init__(self, num_classes=2, d_model=64, nhead=8, num_layers=4, patch_size=160, max_seq_len=100000):
        super(Transformer1D, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding en 1D
        self.conv_embed = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Token especial de clase
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Embeddings posicionales
        max_patches = max_seq_len // patch_size + 2
        self.positional_encoding = nn.Parameter(torch.randn(1, max_patches, d_model))
        
        # Transformers Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Clasificador
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        """
        x: (Batch, 1, Seq_Length)
        """
        # Extraer patches mediante Conv1d
        patches = self.conv_embed(x)             # (B, d_model, num_patches)
        patches = patches.transpose(1, 2)        # (B, num_patches, d_model)
        
        B = x.shape[0]
        # Expandir cls_token para el batch actual
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Concatenar token a los patches
        x = torch.cat((cls_tokens, patches), dim=1)
        seq_len_patches = x.shape[1]
        
        # Añadir las posiciones temporales
        x = x + self.positional_encoding[:, :seq_len_patches, :]
        
        # Bloque transformer
        x = self.transformer(x)
        
        # Usar sólo el token CLS de salida para la clasificación final
        cls_out = x[:, 0]
        out = self.mlp_head(cls_out)
        
        return out
