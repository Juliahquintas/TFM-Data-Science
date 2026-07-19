# TFM — Detección de Parkinson mediante análisis de voz

Este repositorio contiene el código fuente para el Trabajo de Fin de Máster:  End-to-End Deep Learning on Raw Speech for Parkinson's Disease Detection: A Comparative Study on PC-GTA and Neurovoz. Este TFM se centra en la detección de la enfermedad de Parkinson a partir de grabaciones de voz, utilizando redes neuronales que trabajan directamente sobre la señal de audio cruda (raw waveform), sin extracción manual de características acústicas.

Se trabaja sobre dos bases de datos públicas de voz patológica:

- **NeuroVoz**
- **PC-GITA**

y se comparan tres arquitecturas de red distintas para la tarea de clasificación binaria (Control vs. Parkinson).

---

## Estructura del repositorio

```
TFM-Data-Science/
├── preprocessing/           ← limpieza, normalización y partición de los datos
│   ├── data_exploration.ipynb   Análisis exploratorio de los audios (EDA)
│   ├── preprocessing.py         Limpieza y estandarización de los audios (silencios, duración, amplitud)
│   └── data_split.py            Generación de particiones K-Fold a nivel de sujeto
│
├── modelos/                 ← una carpeta independiente por arquitectura
│   ├── cdil_cnn/                 CDIL CNN (línea base)
│   ├── temp_attention/           CDIL CNN + Temporal Attention Pooling
│   └── wavenet/                  WaveNet-style CNN (gated activations)
│
└── requirements.txt          Dependencias generales del proyecto
```

Cada modelo en `modelos/` es autocontenido: tiene su propio `config.py`, `model.py`, `train.py`, `requirements.txt` y su propia carpeta `results/`. Esto permite lanzar y comparar experimentos de cada arquitectura de forma independiente.

---

## Modelos incluidos

| Modelo | Carpeta | Idea clave |
|---|---|---|
| **CDIL CNN** | [`modelos/cdil_cnn`](modelos/cdil_cnn) | Línea base: CNN 1D con convoluciones dilatadas y padding circular |
| **Temporal Attention** | [`modelos/temp_attention`](modelos/temp_attention) | Mismo backbone CDIL, sustituyendo el Global Average Pooling por un mecanismo de atención temporal aprendido |
| **WaveNet** | [`modelos/wavenet`](modelos/wavenet) | Backbone estilo WaveNet (Van den Oord et al.) con Gated Activation Units y skip connections acumulativas |

Los tres modelos comparten el mismo pipeline de entrenamiento (validación cruzada de 5 folds a nivel de sujeto, mismas métricas y mismo formato de resultados), lo que permite comparar arquitecturas en igualdad de condiciones.

---

## Pipeline de trabajo

1. **Preprocesar el audio** (`preprocessing/preprocessing.py`): recorta silencios, normaliza amplitud a ±1.0, remuestrea a 22050 Hz y ajusta todos los audios a la misma duración (la duración mínima del conjunto tras eliminar silencios).
2. **Generar las particiones** (`preprocessing/data_split.py`): crea una validación cruzada de 5 folds *a nivel de sujeto* (evitando que audios del mismo paciente aparezcan en train y test a la vez) y guarda el resultado en `data/data_splits.json`.
3. **Entrenar y Validar el modelo** (`modelos/<modelo>/train.py`): entrena y evalúa el modelo elegido sobre los 5 folds, guardando métricas, gráficas y un registro comparable en `modelos/<modelo>/results/`.

```bash
# 1. Preprocesado
python preprocessing/preprocessing.py

# 2. Partición en folds
python preprocessing/data_split.py

# 3. Entrenamiento (ejemplo con CDIL CNN)
cd modelos/cdil_cnn
pip install -r requirements.txt
python train.py
```

> **Nota sobre los datos:** la carpeta `data/` (audios originales, audios procesados y `data_splits.json`) no se incluye en el repositorio.
---

## Instalación

```bash
pip install -r requirements.txt
```

Cada modelo tiene además su propio `requirements.txt` con las dependencias mínimas para entrenarlo de forma aislada (útil si solo quieres reproducir un experimento concreto sin instalar todo el proyecto).

---

## Notebooks de experimentación

Dentro de `modelos/<modelo>/experimentos/` hay notebooks (`Neurovoz_EXPn.ipynb`, `PcGita_EXPn.ipynb`) usados para lanzar y documentar distintas configuraciones de hiperparámetros sobre cada dataset. Los resultados numéricos de referencia de cada ejecución quedan además registrados en `modelos/<modelo>/results/experiments_log.csv`.
