import torch
from tqdm import tqdm

class Trainer:
    """
    Bucle de entrenamiento por Fold encapsulado en una clase.
    Contiene la lógica de entrenamiento, validación y métricas por lote (batch).
    """
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def train_epoch(self, dataloader, clip_grad_norm=1.0):
        """
        Realiza un paso completo (epoch) por los datos de entrenamiento.
        """
        self.model.train()
        total_loss = 0.0
        
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(dataloader, desc="Training Batch", leave=False)
        
        for waveforms, labels in progress_bar:
            # Mover a GPU/CPU
            waveforms, labels = waveforms.to(self.device), labels.to(self.device)
            
            # Limpiar gradientes
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(waveforms)
            loss = self.criterion(outputs, labels)
            
            # Backward pass 
            loss.backward()
            
            # Prevenir exploding gradients en series de tiempo
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                
            self.optimizer.step()
            
            total_loss += loss.item() * waveforms.size(0) # Ponderado por tamaño real del lote
            
            # Predicciones binarias (clase con la mayor puntuación lógica)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Actualizar barra gráfica
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss, all_preds, all_labels
        
    def evaluate(self, dataloader):
        """
        Realiza la validación (sin modificar gradientes ni pesos)
        """
        self.model.eval()
        total_loss = 0.0
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        progress_bar = tqdm(dataloader, desc="Validation Batch", leave=False)
        
        with torch.no_grad():
            for waveforms, labels in progress_bar:
                # Mover tensor a CPU/GPU
                waveforms, labels = waveforms.to(self.device), labels.to(self.device)
                
                outputs = self.model(waveforms)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * waveforms.size(0)
                
                # Para la curva ROC AUC, calculamos softmax probabilístico 
                probs = torch.softmax(outputs, dim=1)[:, 1] # Probabilidad de PD (clase 1)
                preds = torch.argmax(outputs, dim=1)        # Predicción rígida
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss, all_preds, all_probs, all_labels
