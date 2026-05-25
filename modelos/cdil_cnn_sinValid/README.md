# CDIL CNN — Detección de Parkinson

Reimplementación del modelo **CDIL CNN** propuesto por Marta Rey-Paredes et al. (2024), con adaptaciones para su aplicación sobre las bases de datos **NeuroVoz** y **PC-GITA** sin augmentación de datos.

---

## Adaptaciones respecto al trabajo original

| Aspecto | Marta Rey (original) | Esta implementación |
|---|---|---|
| Nº de capas | `floor(log2(seq_len))` sin límite | `min(floor(log2(seq_len)), 8)` |
| Inicialización | Reinit sobre `weight_norm` (sin efecto real) | Inicialización por defecto de PyTorch |
| Batch size | 256 | 32 |
| Learning rate | 1e-4 | 1e-3 |
| Augmentación | Sí | No (evaluación sobre señal original) |

---

## Estructura

```
CDIL_CNN/
├── model.py        ← arquitectura CDIL CNN
├── config.py       ← hiperparámetros del experimento
├── train.py        ← entrenamiento, evaluación y registro de resultados
├── requirements.txt
└── results/
    └── experiments_log.csv   ← tabla comparativa de todos los experimentos
```

---

## Uso

```bash
pip install -r requirements.txt

# 1. Ajusta los parámetros en config.py
# 2. Lanza el entrenamiento
python train.py
```

---

## Comparación de experimentos

Cada ejecución genera su propia carpeta dentro de `results/` y añade una fila a `results/experiments_log.csv`, que acumula hiperparámetros y métricas de todos los experimentos realizados.

Ejemplo de `experiments_log.csv`:

| experiment | dataset | nhid | lr | batch_size | dropout | mean_accuracy | mean_f1 | mean_auc |
|---|---|---|---|---|---|---|---|---|
| baseline | neurovoz | 32 | 0.001 | 32 | 0.0 | 0.6821 | 0.6734 | 0.7102 |
| dropout02 | neurovoz | 32 | 0.001 | 32 | 0.2 | 0.6543 | 0.6412 | 0.6890 |

---

## Salidas por experimento

```
results/<nombre_experimento>/
├── config_used.py              copia del config empleado
├── curves_fold{k}.png          curvas loss/accuracy por época
├── cm_fold{k}.png              matriz de confusión por fold
├── roc_fold{k}.png             curva ROC por fold
├── accumulated_cm.png          matriz de confusión acumulada (5 folds)
├── mean_roc_curve.png          curva ROC media con banda ±1 std
├── per_fold_metrics.csv        métricas detalladas por fold
└── summary_metrics.csv         media ± std de todas las métricas
```

---

## Arquitectura

```
Input (batch, seq_len, 1)
  → permuta → (batch, 1, seq_len)
  → L × ConvBlock  [dil = 2^i, padding circular, skip connection, ReLU]
  → Global Average Pooling
  → Linear(nhid → 2)
  → Logits
```

Con `L = min(floor(log2(seq_len)), 8)`.
