# 🧠 Entorno de Modelado — TFM Parkinson

Guía completa del entorno creado para entrenar, experimentar y evaluar modelos de redes neuronales para la detección de Parkinson en voz.

---

## 📁 Estructura de carpetas nueva

```
TFM-Data-Science/
│
├── notebooks/                         ← NOTEBOOKS DE TRABAJO
│   ├── 01_familia_cnn1d.ipynb         🟦 CNN 1D (3 modelos)
│   ├── 02_familia_tcn.ipynb           🟨 TCN / WaveNet-like (2 modelos)
│   ├── 03_familia_transformers.ipynb  🟥 Transformers (2 modelos)
│   └── 04_evaluacion_modelos.ipynb    📊 Evaluación y comparativa
│
├── src/
│   ├── models/
│   │   ├── cnn_1d.py                  ← Familia 1: BaselineCNN1D, ResidualCNN1D, DilatedCNN1D
│   │   ├── tcn.py                     ← Familia 2: TCN, WaveNetLike
│   │   └── transformer_audio.py       ← Familia 3: TemporalTransformer, CNNTransformerHybrid
│   │
│   └── training/
│       ├── audio_dataset.py           ← DataLoader desde data_splits.json
│       ├── experiment_logger.py       ← Logger de experimentos (CSV + JSON)
│       └── trainer.py                 ← Bucle de entrenamiento (sin cambios)
│
└── experiments/                       ← SE CREA AUTOMÁTICAMENTE al entrenar
    ├── experiments_log.csv            ← Tabla de todos los experimentos
    ├── saved_models/                  ← Pesos .pt de cada modelo entrenado
    ├── figures/                       ← ROC curves y matrices de confusión
    └── <run_id>.json                  ← Detalle de cada experimento
```

---

## 🚀 Cómo usar el entorno

### Paso 1: Entrenar un modelo

1. Abre uno de los notebooks de entrenamiento según la familia:
   - `01_familia_cnn1d.ipynb` para CNNs
   - `02_familia_tcn.ipynb` para TCN/WaveNet
   - `03_familia_transformers.ipynb` para Transformers

2. **Edita SOLO la celda `⚙️ CONFIGURACION`**:

```python
MODEL_NAME   = "baseline_cnn1d"   # ← el modelo que quieres probar
DATASET_NAME = "neurovoz"          # ← "neurovoz" o "pc-gita"
FOLD_INDEX   = 0                   # ← qué fold usar (0..4)

MODEL_HYPERPARAMS = {
    "n_filters":   [32, 64, 128],  # ← cambia esto para experimentar
    "kernel_size": 7,
    "dropout":     0.3,
}

NOTES = "Mi primera prueba"        # ← nota libre para el log
```

3. Ejecuta: `Kernel → Restart & Run All`

Al terminar:
- El modelo queda guardado en `experiments/saved_models/`
- Los resultados quedan en `experiments/experiments_log.csv`
- Se muestra la tabla de todos los experimentos hasta ahora

---

## 🔬 Modelos disponibles

### 🟦 Familia 1: CNN 1D

| Modelo | Clase | Hiperparámetros clave |
|--------|-------|----------------------|
| `baseline_cnn1d` | `BaselineCNN1D` | `n_filters`, `kernel_size`, `dropout` |
| `residual_cnn1d` | `ResidualCNN1D` | `n_filters`, `n_res_blocks`, `kernel_size`, `dropout` |
| `dilated_cnn1d` | `DilatedCNN1D` | `n_channels`, `dilations`, `kernel_size`, `dropout` |

### 🟨 Familia 2: Temporal Convolutional

| Modelo | Clase | Hiperparámetros clave |
|--------|-------|----------------------|
| `tcn` | `TCN` | `num_channels`, `kernel_size`, `dropout` |
| `wavenet_like` | `WaveNetLike` | `residual_channels`, `skip_channels`, `n_layers`, `kernel_size` |

### 🟥 Familia 3: Transformers

| Modelo | Clase | Hiperparámetros clave |
|--------|-------|----------------------|
| `temporal_transformer` | `TemporalTransformer` | `patch_size`, `d_model`, `nhead`, `num_layers`, `dim_feedforward` |
| `cnn_transformer_hybrid` | `CNNTransformerHybrid` | `cnn_channels`, `d_model`, `nhead`, `num_layers` |

---

## 📊 Evaluación

### Notebook `04_evaluacion_modelos.ipynb`

Solo necesitas indicar la ruta al modelo guardado:

```python
MODEL_PT_PATH = "../experiments/saved_models/20260507_143012_baseline_cnn1d_neurovoz_fold1.pt"
EVAL_SPLIT = "test"   # o "val"
```

Genera automáticamente:
- 📋 Tabla de métricas: Accuracy, Sensibilidad, Especificidad, F1, MCC, AUC-ROC
- 📈 Curva ROC con AUC y umbral óptimo de Youden
- 🔲 Matriz de confusión (valores absolutos + normalizada)
- 📊 Tabla comparativa de **todos** los experimentos realizados (resaltando el mejor)

Las figuras se guardan en `experiments/figures/`.

---

## 📝 Log de experimentos

Cada prueba queda registrada automáticamente en dos formatos:

**`experiments_log.csv`** — tabla resumen rápida:

| run_id | model_name | dataset | fold | accuracy | auc_roc | f1 | notes |
|--------|-----------|---------|------|----------|---------|-----|-------|
| 20260507_143012_baseline_cnn1d_fold1 | baseline_cnn1d | neurovoz | 1 | 0.8421 | 0.9134 | 0.8512 | Prueba inicial |

**`<run_id>.json`** — detalle completo con hiperparámetros, historial de loss y métricas.

---

## 💡 Consejos para experimentar

> [!TIP]
> Para hacer pruebas rápidas, reduce `epochs` a 10-15 mientras configuras los hiperparámetros. Cuando estés conforme, sube a 50-100 con early stopping.

> [!IMPORTANT]
> Cambia siempre el campo `NOTES` antes de cada prueba. Te ayudará a entender el log de experimentos después.

> [!NOTE]
> El `FOLD_INDEX` permite entrenar el mismo modelo en 5 particiones distintas para obtener resultados estadísticamente robustos. Para una comparación justa entre modelos, usa siempre el mismo fold.

> [!WARNING]
> Los Transformers (Familia 3) son más pesados. Si tienes solo CPU, usa `batch_size=8` y `num_layers=2` para empezar.

---

## 🗑️ Archivos eliminados / ignorar

Los archivos anteriores en `src/models/` (`wavenet.py`, `transformer_1d.py`, `sincnet.py`, `mamba_audio.py`) han sido **reemplazados** por la nueva estructura. El notebook antiguo `src/models/01_baseline_cnn.ipynb` también queda obsoleto.
