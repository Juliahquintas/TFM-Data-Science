# TFM Parkinson - Detección de Parkinson por Voz

Este repositorio contiene el código fuente para el Trabajo de Fin para la detección de la enfermedad de Parkinson utilizando grabaciones de voz (por defecto pensado para datasets de voz como PC-GITA y NeuroVoz).

## Estructura del repositorio
- `src/preprocessing/`: Scripts para cargar y preprocesar audio (recorte de silencios, normalización).
- `src/models/`: Arquitecturas de redes neuronales (WaveNet, Transformer 1D, SincNet, y placeholder para Mamba/xLSTM).
- `src/training/`: Lógica de entrenamiento incluyendo un Trainer (bucle con optimizador) y el script para validación cruzada.
- `src/evaluation/`: Cálculo de métricas e informes.
- `configs/`: Archivos YAML con la configuración de hiperparámetros.
- `src/train.py`: Script principal para orquestar el entrenamiento.

## Uso
Ejemplo de ejecución del script de entrenamiento:
```bash
python src/train.py --config configs/wavenet_pcgita.yaml --model wavenet --dataset PC-GITA --vocal a
```

Puedes instalar las dependencias con:
```bash
pip install -r requirements.txt
```
