"""
config.py — Hiperparámetros del experimento CDIL CNN
═════════════════════════════════════════════════════
Modifica este archivo para cada experimento.
Cada ejecución guarda una copia de este config junto a sus resultados,
y añade una fila al fichero results/experiments_log.csv para comparar.
"""

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET   = 'pc-gita'   # 'neurovoz' | 'pc-gita'

# ── Arquitectura ──────────────────────────────────────────────────────────────
NHID         = 32    # canales por capa (feature maps)
KERNEL_SIZE  = 3     # tamaño del kernel convolucional
DROPOUT      = 0.0   # dropout tras cada bloque (0.0 = desactivado)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
BATCH_SIZE   = 32    # muestras por paso de gradiente
N_EPOCHS     = 200   # épocas de entrenamiento
LR           = 1e-3  # learning rate (Adam)

# ── Validación ────────────────────────────────────────────────────────────────
VAL_RATIO    = 0.10  # fracción de sujetos de train reservados para validación
K_FOLDS      = 5     # número de folds (debe coincidir con data_splits.json)
SEED         = 42    # semilla para reproducibilidad

# ── Rutas ─────────────────────────────────────────────────────────────────────
# El script train.py sube automáticamente hasta encontrar la carpeta 'data/'.
# Solo necesitas cambiar esto si tu estructura de carpetas es diferente.
SPLITS_FILE   = 'data/data_splits.json'
PROCESSED_DIR = 'data/processed'

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_EVERY_N_EPOCHS = 10   # imprime métricas cada N épocas (y en la 1ª y última)

# ── Etiqueta del experimento (opcional) ───────────────────────────────────────
# Si la dejas vacía se genera automáticamente a partir de los hiperparámetros.
# Úsala para identificar el experimento en el log: ej. 'baseline', 'dropout02'
EXPERIMENT_NAME = 'pc-gita_dropout00'
