
# Planificación Completa del TFM

**Desarrollo y comparación de arquitecturas de redes neuronales
profundas en dominio temporal para el diagnóstico automático de
patologías de la voz**

Este documento contiene:\
- Explicación detallada de cada fase del trabajo\
- Objetivos semanales\
- Lista de tareas accionables para marcar en VS Code\
- Recomendaciones técnicas, modelos a implementar y flujo de trabajo

------------------------------------------------------------------------

# 🗓️ Vista General (1 -- 1.5 meses, 20--25 h/semana)

Tu TFM se organiza en **6 semanas**, cada una con una meta clara y
entregables concretos.\
El objetivo es **superar los dos TFM anteriores**, añadiendo modelos
modernos (Transformers temporales, WaveNet-like, ConvNeXt-1D) y un
generador GAN más reciente (BigVGAN v2 o DiffWave).

------------------------------------------------------------------------

# ⭐ Semana 1 --- Preparación y Setup

## 🎯 Objetivo: Entender completamente el problema y montar todo el entorno

En esta semana te familiarizas con el trabajo previo y defines qué vas a
mejorar. Además, preparas todo el ecosistema de trabajo: datasets,
entorno, repositorio, preprocesamiento básico.

### Explicación

Los TFMs anteriores usaron:\
- Marta: BigVSAN + ResNet, LSTM-FCN, InceptionTime, CDIL-CNN\
- Javier: QGAN + modelos similares + TimesNet

Tú vas a mejorar esto con:\
- Nuevas arquitecturas 2023--2025\
- Dos bases de datos (PC-GITA + Neurovoz)\
- Interpretabilidad\
- GAN más moderno

### ✔ Checklist Semana 1

-   [ ] Leer TFM Marta Rey y escribir resumen de limitaciones\
-   [ ] Leer TFM Javier Jardón y escribir resumen de limitaciones\
-   [✔] Crear repositorio del TFM en GitHub\
-   [ ] Crear entorno Python + PyTorch\
-   [✔] Solicitar acceso a Neurovoz (urgente)\
-   [✔] Descargar dataset PC-GITA\
-   [ ] Preprocesar: normalización y recorte de silencios\
-   [ ] Escribir documento "Mejoras propuestas sobre trabajos previos"

------------------------------------------------------------------------

# ⭐ Semana 2 --- Pipeline y Baselines

## 🎯 Objetivo: Construir un pipeline robusto e implementar los modelos base

Antes de probar modelos nuevos necesitas reproducir y mejorar el
baseline.

### Explicación

Los baselines sirven para comparar. Si tus modelos modernos no mejoran
estos resultados, sabrás que algo falla.\
El pipeline debe incluir: - DataLoader sujeto-wise\
- Validación cruzada\
- Logging (TensorBoard o WandB)\
- Métricas clínicas (accuracy, F1, MCC, AUC)

### ✔ Checklist Semana 2

-   [ ] Crear pipeline modular: loaders → modelos → entrenamiento →
    métricas\
-   [ ] Implementar validación subject-wise\
-   [ ] Implementar modelos baseline:
    -   [ ] ResNet-1D\
    -   [ ] LSTM-FCN\
    -   [ ] InceptionTime\
    -   [ ] CDIL-CNN\
    -   [ ] TimesNet (opcional)\
-   [ ] Ejecutar primeros experimentos base\
-   [ ] Comparar con TFMs previos y documentar

------------------------------------------------------------------------

# ⭐ Semana 3 --- Entrenamiento GAN Moderno

## 🎯 Objetivo: Crear un generador sintético mejor que BigVSAN y QGAN

### Explicación

Aquí introduces la parte generativa, pero sólo **una arquitectura**
(como pidió tu tutor).\
Recomendados:\
- **BigVGAN v2 (2023)** → calidad excelente\
- **DiffWave / FastDiff (2021--2023)** → alternativa moderna basada en
difusión

Necesitas entrenar el generador para producir nuevas vocales que
aumenten tus datos.

### ✔ Checklist Semana 3

-   [ ] Elegir arquitectura GAN\
-   [ ] Preparar entorno GPU (local o nube)\
-   [ ] Entrenar modelo con PC-GITA\
-   [ ] Generar audios sintéticos\
-   [ ] Evaluarlos con:
    -   [ ] MCD\
    -   [ ] PESQ\
    -   [ ] STOI\
    -   [ ] MS-STFT Loss\
-   [ ] Comparar visualmente waveform real vs sintética\
-   [ ] Crear dataset combinado real + sintético

------------------------------------------------------------------------

# ⭐ Semana 4 --- Arquitecturas Modernas

## 🎯 Objetivo: Implementar modelos avanzados (2022--2025) no usados en TFMs previos

### Explicación

Esta es la parte donde tu TFM destaca. Los modelos que deberías incluir:

### Modelos modernos sugeridos

-   **Temporal Convolutional Transformer (TCT)**\
-   **Transformers eficientes (FlashAttention / Linear Transformer)**\
-   **WaveNet-like classifier** (no generativo, solo discriminador)\
-   **ConvNeXt-1D**\
-   **WavEmbed + MLP/Vit-lite**

El propósito es evaluar si estos modelos superan a ResNet-1D, LSTM-FCN,
etc.

### ✔ Checklist Semana 4

-   [ ] Implementar Transformer temporal\
-   [ ] Implementar TCT\
-   [ ] Implementar WaveNet-like classifier\
-   [ ] Implementar ConvNeXt-1D\
-   [ ] Implementar WavEmbed + MLP o ViT-lite\
-   [ ] Entrenar cada modelo con PC-GITA\
-   [ ] Registrar métricas y comparar

------------------------------------------------------------------------

# ⭐ Semana 5 --- Experimentación y Optimización

## 🎯 Objetivo: Comparar modelos reales vs sintéticos y elegir los definitivos

### Explicación

Aquí realizas: - Comparación "sin sintéticos" vs "con sintéticos"\
- Búsqueda de hiperparámetros\
- Interpretabilidad\
- Tablas finales

### ✔ Checklist Semana 5

-   [ ] Ejecutar comparativa:
    -   [ ] Modelos sin datos sintéticos\
    -   [ ] Modelos con datos sintéticos\
-   [ ] Optimización de hiperparámetros (Grid/Bayes search)\
-   [ ] Evaluación por:
    -   [ ] sexo\
    -   [ ] vocal\
    -   [ ] duración\
-   [ ] Interpretabilidad:
    -   [ ] Grad-CAM 1D\
    -   [ ] Mapas de atención del Transformer\
    -   [ ] SHAP temporal\
-   [ ] Tabla comparativa final entre arquitecturas\
-   [ ] Selección de los 3 mejores modelos

------------------------------------------------------------------------

# ⭐ Semana 6 --- Escritura, Memoria y Defensa

## 🎯 Objetivo: Redactar, generar gráficos y preparar presentación

### Explicación

Última fase: documentación y defensa.\
Aquí incluyes:\
- Resultados\
- Gráficos\
- Conclusiones\
- Limitaciones\
- Futuro trabajo

### ✔ Checklist Semana 6

-   [ ] Redacción de metodología completa\
-   [ ] Redacción de experimentos y análisis\
-   [ ] Redacción del estado del arte actualizado\
-   [ ] Preparar conclusiones + líneas futuras\
-   [ ] Crear presentación de diapositivas\
-   [ ] Preparar demostraciones o audios de ejemplo

------------------------------------------------------------------------

# 🎯 Lista de Innovaciones Obligatorias para Superar TFMs Previos

-   [ ] Usar dos datasets (PC-GITA + Neurovoz)\
-   [ ] Implementar modelos modernos (Transformers, ConvNeXt-1D,
    WaveNet-like)\
-   [ ] GAN moderno (BigVGAN v2 / DiffWave)\
-   [ ] Interpretabilidad con Grad-CAM/SHAP\
-   [ ] Métrica MCC + subject-wise CV\
-   [ ] Comparación sistemática con y sin sintéticos

------------------------------------------------------------------------

# 🧩 Notas Personales

*(Aquí puedes añadir tus propias notas semanales, cambios, dudas, etc.)*

------------------------------------------------------------------------

Fin de la planificación.
