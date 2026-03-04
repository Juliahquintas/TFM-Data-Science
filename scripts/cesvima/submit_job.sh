#!/bin/bash
#SBATCH --job-name=tfm_julia
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard-gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

# 1. Limpiamos cualquier módulo previo
module purge

# 2. Cargamos los módulos oficiales de Deep Learning
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1 \
            Jupyter-bundle/20230823-GCCcore-12.3.0 \
            SciPy-bundle/2023.07-gfbf-2023a \
            matplotlib/3.7.2-gfbf-2023a \
            rasterio/1.3.9-foss-2023a \
            scikit-learn/1.4.2-gfbf-2023a

# 3. Activamos tu entorno framework
export PATH=~/conda/envs/framework/bin:~/conda/bin:$PATH
source activate framework

# 4. Aseguramos que las librerías base (como TensorFlow / PyTorch) vean las librerías de CUDA
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX

# 5. Entrar en la carpeta del TFM
# ¡IMPORTANTE!: Esta ruta asume que has metido el repositorio aquí. Cambiala si hace falta.
cd /home/y222/y222732/DeepLearningChallenge/TFM-Data-Science

# 6. Ejecutamos tu Cuaderno General que abarca todo el proceso de entrenamiento y test
srun jupyter nbconvert --execute --to notebook --inplace train.ipynb
