# CDIL CNN + Temporal Attention — Detección de Parkinson

Variante del modelo [CDIL CNN](../cdil_cnn) en la que el *Global Average Pooling* final se sustituye por un mecanismo de **atención temporal aprendida**, que permite al modelo ponderar de forma distinta las distintas regiones de la señal de voz en lugar de promediarlas todas por igual.

---

## Arquitectura

```
Input (batch, seq_len, 1)
  → permuta → (batch, 1, seq_len)
  → L × ConvBlock  [dil = 2^i, kernel dilatado, padding circular, skip residual, ReLU]
  → permuta → (batch, seq_len, nhid)
  → Temporal Attention Pooling   ← diferencia frente a CDIL CNN
  → Linear(nhid → 2)
  → Logits
```


---

## Estructura de la carpeta

```
temp_attention/
├── model.py               ConvBlock + TemporalAttentionPooling + CDILAttentionClassifier
├── config.py               hiperparámetros del experimento a ejecutar
├── train.py                entrenamiento, evaluación (5-Fold CV) y registro de resultados
├── requirements.txt
├── experimentos/           notebooks con las configuraciones probadas (Neurovoz/PC-GITA × 5 experimentos)
└── results/
    └── experiments_log.csv   tabla comparativa de todos los experimentos lanzados
```

---

## Uso

```bash
pip install -r requirements.txt

# 1. Ajusta los parámetros en config.py (dataset, arquitectura, entrenamiento)
# 2. Lanza el entrenamiento
python train.py
```

Igual que en el resto de modelos, `train.py` localiza automáticamente la carpeta `data/` y requiere que exista `data/data_splits.json` (generado por `preprocessing/data_split.py`).

---

## Comparación de experimentos

Cada ejecución añade una fila a `results/experiments_log.csv` con los hiperparámetros y las métricas medias (accuracy, precision, recall, especificidad, F1 y AUC) de esa configuración, lo que permite comparar directamente el efecto de la atención temporal frente al pooling simple de CDIL CNN, así como distintos valores de `nhid`, `dropout` y `learning rate`.

---

## Validación

Al igual que en CDIL CNN, la evaluación se realiza mediante **validación cruzada de 5 folds a nivel de sujeto**, calculando las métricas por fold y agregándolas como media ± desviación típica.
