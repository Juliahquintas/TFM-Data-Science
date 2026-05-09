#!/bin/bash
#SBATCH --job-name=run_experiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard-gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# 1. Limpiar e inicializar módulos en CESVIMA
module purge
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1 \
            Jupyter-bundle/20230823-GCCcore-12.3.0 \
            SciPy-bundle/2023.07-gfbf-2023a \
            matplotlib/3.7.2-gfbf-2023a \
            rasterio/1.3.9-foss-2023a \
            scikit-learn/1.4.2-gfbf-2023a

# Instalamos las dependencias faltantes para el notebook en el entorno de usuario de CESVIMA (Python 3.11 del módulo)
python3 -m pip install --user librosa tqdm soundfile

# 2. Activar el entorno
export PATH=~/conda/envs/framework/bin:~/conda/bin:$PATH
source activate framework

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# 3. Ir a la raíz del TFM
cd /home/y222/y222732/DeepLearningChallenge/TFM-Data-Science

# El primer argumento es el path al notebook que quieres ejecutar
# Por defecto se ejecutará src/experiments/wavenet.ipynb si no indicas otro
NOTEBOOK=${1:-src/experiments/wavenet.ipynb}

if [ ! -f "$NOTEBOOK" ]; then
    echo "Error: No se encuentra el cuaderno $NOTEBOOK"
    exit 1
fi

BASENAME=$(basename "$NOTEBOOK" .ipynb)
OUT_DIR="src/experiments/results/${BASENAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

OUT_NOTEBOOK="$OUT_DIR/${BASENAME}_ejecutado.ipynb"
OUT_CSV="$OUT_DIR/log.csv"
OUT_TXT="$OUT_DIR/accuracy_y_resumen.txt"

echo "--------------------------------------------------------"
echo " Lanzando entrenamiento de $NOTEBOOK"
echo " Resultados se guardarán en: $OUT_DIR"
echo "--------------------------------------------------------"

# 4. Ejecutar el Cuaderno
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

# Se guardará el output en la carpeta de resultados usando --output-dir
jupyter nbconvert --execute --to notebook --output-dir "$OUT_DIR" --output "${BASENAME}_ejecutado.ipynb" "$NOTEBOOK"

# 5. Extracción de Métricas usando Python (incrustado)
echo "Procesando el cuaderno para extraer log.csv y el resumen/accuracy..."

python3 -c "
import json
import csv
import re
import sys

notebook_path = '$OUT_NOTEBOOK'
csv_path = '$OUT_CSV'
txt_path = '$OUT_TXT'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print('Error leyendo el notebook ejecutado:', e)
    sys.exit(1)

log_rows = []
model_summary = []
accuracy_text = []

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        for out in cell.get('outputs', []):
            text = ''
            if out.get('output_type') == 'stream':
                text = ''.join(out.get('text', []))
            elif out.get('output_type') in ['execute_result', 'display_data']:
                text_data = out.get('data', {}).get('text/plain', [])
                if isinstance(text_data, list):
                    text = ''.join(text_data)
                elif isinstance(text_data, str):
                    text = text_data
            
            if not text:
                continue

            lines = text.split('\n')
            
            # Extraer Logs de entrenamiento (loss, accuracy por epoch)
            for line in lines:
                # Filtrar lineas típicas de progreso de Keras (Ej: 1/50 [===] - 2s 15ms/step - loss: 0.5 - accuracy: 0.8)
                if 'loss:' in line or '- val_' in line or ('Epoch' in line and '/' in line):
                    # Limpiamos barras de progreso
                    clean_line = re.sub(r'+', '', line) # limpiar backspaces
                    log_rows.append([clean_line.strip()])
                    
            # Extraer Resumen del modelo
            if 'Model:' in text or 'Layer (type)' in text or 'Total params:' in text:
                model_summary.append(text)
                
            # Extraer Reportes de Precisión / Classification report
            if 'accuracy' in text.lower() or 'classification report' in text.lower() or 'roc auc' in text.lower() or 'f1-score' in text.lower() or 'confusion matrix' in text.lower():
                accuracy_text.append(text)

# Escribir log.csv
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Keras_Training_Logs'])
    writer.writerows(log_rows)

# Escribir el .txt con accuracy y summary
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write('===========================================\n')
    f.write('            RESUMEN DEL MODELO\n')
    f.write('===========================================\n\n')
    for summary in set(model_summary):  # set() para evitar duplicados si se imprime varias veces
        f.write(summary + '\n')
        
    f.write('\n\n')
    f.write('===========================================\n')
    f.write('           METRICAS Y ACCURACY\n')
    f.write('===========================================\n\n')
    for acc in set(accuracy_text):
        f.write(acc + '\n')

print('Extracción completada. Revisa:', csv_path, 'y', txt_path)
"

echo "Trabajo finalizado."
