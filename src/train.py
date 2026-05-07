import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) )

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split # Para una división simple y no liarnos con Cross Validation

from src.models.Antiguo_wavenet import WaveNet
from src.models.transformer_1d import Transformer1D
from src.models.sincnet import SincNet
from src.models.mamba_audio import MambaAudio
from src.training.trainer import Trainer
from src.evaluation.metrics import calculate_metrics, plot_roc_curve

def get_model(model_name, num_classes=2, config_params=None):
    if config_params is None:
        config_params = {}
    model_configs = config_params.get(model_name, {})
    if model_name == 'wavenet':
        return WaveNet(num_classes=num_classes, **model_configs)
    elif model_name == 'transformer_1d':
        return Transformer1D(num_classes=num_classes, **model_configs)
    elif model_name == 'sincnet':
        return SincNet(num_classes=num_classes)
    elif model_name == 'mamba':
        return MambaAudio(num_classes=num_classes, **model_configs)
    else:
        raise ValueError(f"Modelo {model_name} no disponible.")

def main():
    parser = argparse.ArgumentParser(description="Script de Entrenamiento (TFM Parkinson)")
    parser.add_argument('--config', type=str, default='src/training/config/wavenet_pcgita.yaml')
    parser.add_argument('--model', type=str, default='wavenet')
    parser.add_argument('--dataset', type=str, default='PC-GITA')
    parser.add_argument('--vocal', type=str, default='a')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo activo: {device}")
    
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 16)
    lr = config.get('learning_rate', 1e-3)
    patience = config.get('early_stopping_patience', 5)
    
    print(f"Dataset simulado de prueba: {args.dataset}, vocal: {args.vocal}")
    num_samples = 100
    seq_length = 16000 
    
    X = torch.randn(num_samples, 1, seq_length)
    y = torch.randint(0, 2, (num_samples,))
    
    # ------------------
    # División simple (80% Entrenamiento, 20% Validación)
    # Por ahora en el TFM, esto es suficiente para poder lanzar la red.
    # ------------------
    # Extraemos índices para simular
    indices = torch.arange(num_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n=======================")
    print(f"--- FASE DE ENTRENAMIENTO ---")
    print(f"=======================")
    
    train_dataset = TensorDataset(X[train_idx], y[train_idx])
    val_dataset = TensorDataset(X[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model_params = config.get('model_params', {})
    model = get_model(args.model, num_classes=2, config_params=model_params)
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, optimizer, criterion, device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_preds, train_labels = trainer.train_epoch(train_loader)
        val_loss, val_preds, val_probs, val_labels = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1:02d}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early Stopping automático alcanzado tras {epoch+1} epochs.")
                break
        
    print(f"\nCalculando métricas finales...")
    metrics = calculate_metrics(val_labels, val_preds, val_probs)
    
    for k, v in metrics.items():
        print(f" - {k.capitalize()}: {v:.4f}")
        
    if len(torch.unique(torch.tensor(val_labels))) > 1:
        plot_roc_curve(val_labels, val_probs, title=f"ROC Curve {args.model.upper()}", 
                       save_path=f"roc_curve_final.png")
    print("-----------------------")

if __name__ == '__main__':
    main()
