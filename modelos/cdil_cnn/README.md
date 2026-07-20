# CDIL CNN — Detección de Parkinson

Modelo de línea base: Implematación del modelo **CDIL CNN** , adaptado para su aplicación sobre las bases de datos **NeuroVoz** y **PC-GITA**, trabajando directamente sobre la señal de audio cruda y sin aumento de datos.

---

## Arquitectura

```
Input (batch, seq_len, 1)
  → permuta → (batch, 1, seq_len)
  → L × ConvBlock  [dil = 2^i, kernel dilatado, padding circular, skip residual, ReLU]
  → Global Average Pooling
  → Linear(nhid → 2)
  → Logits
```

Cada `ConvBlock` es una convolución 1D dilatada con `padding_mode='circular'`

---


---

## Estructura de la carpeta

```
cdil_cnn/
├── model.py              arquitectura CDIL CNN (ConvBlock + CDILClassifier)
├── config.py              hiperparámetros del experimento a ejecutar
├── train.py               entrenamiento, evaluación (5-Fold CV) y registro de resultados
├── requirements.txt
├── experimentos/          notebooks con las configuraciones probadas (Neurovoz/PC-GITA × 5 experimentos)
└── results/
    ├── experiments_log.csv        tabla comparativa de todos los experimentos lanzados
    └── <nombre_experimento>/      resultados detallados de cada ejecución
```

---

## Uso

```bash
pip install -r requirements.txt

# 1. Ajusta los parámetros en config.py (dataset, arquitectura, entrenamiento)
# 2. Lanza el entrenamiento
python train.py
```


---

## Comparación de experimentos

Cada ejecución crea su propia carpeta dentro de `results/` y añade una fila a `results/experiments_log.csv`, que acumula hiperparámetros y métricas de todos los experimentos realizados. Esto permite comparar de un vistazo distintos valores de `nhid`, `dropout`, `learning rate`, etc.


---

## Salidas por experimento

```
results/<nombre_experimento>/
├── config_used.py              copia exacta del config empleado en esa ejecución
├── curves_fold{k}.png          curvas de loss/accuracy de entrenamiento por época y fold
├── accumulated_cm.png          matriz de confusión acumulada (5 folds)
├── mean_roc_curve.png          curva ROC media con banda ±1 std
├── per_fold_metrics.csv        accuracy, precision, recall, especificidad, F1 y AUC por fold
└── summary_metrics.csv         media ± desviación típica de todas las métricas
```

---

## Validación

El entrenamiento se evalúa mediante **validación cruzada de 5 folds a nivel de sujeto** (los folds se generan en `preprocessing/data_split.py`, garantizando que ningún paciente aparezca a la vez en train y test). Las métricas reportadas (accuracy, precision, recall, especificidad, F1 y AUC) se calculan por fold y se agregan como media ± desviación típica.
