# WaveNet — Detección de Parkinson

Modelo que utiliza la estructura de la arquitectura **WaveNet** (Van den Oord et al., *"WaveNet: A Generative Model for Raw Audio"*, arXiv:1609.03499), adaptada a un problema de clasificación binaria en lugar de generación de audio.

---

## Arquitectura

```
Input (batch, seq_len, 1)
  → permuta → (batch, 1, seq_len)
  → L × GatedBlock   [dil = 2^i, Gated Activation Unit, skip connection]
  → suma de las skip connections de TODAS las capas
  → ReLU → Global Average Pooling
  → Linear(nhid → 2)
  → Logits
```

Con `L = min(floor(log2(seq_len)), 8)` capas, igual que en CDIL CNN.

---

## Estructura de la carpeta

```
wavenet/
├── model.py               GatedBlock + WaveNetClassifier
├── config.py                hiperparámetros del experimento a ejecutar
├── train.py                 entrenamiento, evaluación (5-Fold CV) y registro de resultados
├── experimentos/             notebooks con las configuraciones probadas (Neurovoz/PC-GITA × 5 experimentos)
└── results/
    └── experiments_log.csv    tabla comparativa de todos los experimentos lanzados
```

> Esta carpeta no incluye un `requirements.txt` propio (a diferencia de `cdil_cnn` y `temp_attention`); usa las dependencias del `requirements.txt` de la raíz del repositorio.

---

## Uso

```bash
pip install -r ../../requirements.txt

# 1. Ajusta los parámetros en config.py (dataset, arquitectura, entrenamiento)
# 2. Lanza el entrenamiento
python train.py
```

Igual que en el resto de modelos, `train.py` localiza automáticamente la carpeta `data/` y requiere que exista `data/data_splits.json` (generado por `preprocessing/data_split.py`).

---

## Comparación de experimentos

Cada ejecución añade una fila a `results/experiments_log.csv` con los hiperparámetros y las métricas medias (accuracy, precision, recall, especificidad, F1 y AUC) de esa configuración, lo que permite comparar el efecto de las gated activations y las skip connections acumulativas frente al backbone CDIL simple.

---

## Validación

Al igual que en el resto de modelos, la evaluación se realiza mediante **validación cruzada de 5 folds a nivel de sujeto**, calculando las métricas por fold y agregándolas como media ± desviación típica.
