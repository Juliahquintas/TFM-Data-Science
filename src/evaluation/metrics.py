import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef, 
    roc_auc_score, 
    roc_curve,
    confusion_matrix
)

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calcula las métricas principales de evaluación para clasificación clínica (Parkinson vs Sano).
    
    Args:
        y_true: etiquetas reales (1 y 0)
        y_pred: etiquetas predichas rígidas por el modelo
        y_prob: probabilidades de la clase positiva (Parkinson) para calcular el AUC y ROC.
    """
    metrics = {}
    
    # Exactitud Global
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Sensibilidad (True Positive Rate) = Recall para clase 1
    # Capacidad para detectar correctamente el Parkinson.
    metrics['sensibilidad'] = recall_score(y_true, y_pred)
    
    # Especificidad (True Negative Rate) = Recall para clase 0.
    # Capacidad para identificar correctamente a los sujetos sanos normales.
    metrics['especificidad'] = recall_score(y_true, y_pred, pos_label=0)
    
    # F1 Score
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # MCC - Coeficiente de Correlación de Matthews (muy robusto en datasets médicos desbalanceados)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Área bajo la Curva Operativa del Receptor (AUC-ROC)
    if len(np.unique(y_true)) > 1:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    else:
        # En caso de que se pase un único lote donde todos son la misma clase por error
        metrics['auc_roc'] = float('nan')
        
    return metrics

def plot_roc_curve(y_true, y_prob, title="ROC Curve Parkinson Detection", save_path=None):
    """
    Dibuja y guarda el gráfico formal del rendimiento predictivo del fold/modelo.
    """
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='crimson', lw=2, label=f'Model ROC (AUC = {auc:.3f})')
        
        # Línea de rendimiento aleatorio (Peor de los casos)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        # Ajustes visuales de la gráfica
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close() # Liberar memoria en Pyplot
        else:
            return plt.gcf()
    return None
