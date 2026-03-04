
# SUMMARY OF MODELS USED BY BOTH TFM AND THE ONLY PROYECT OUTSIDE WITH ROW AUDIOS
# Modelos de Deep Learning para Clasificación de Parkinson en Voz

---

## 1️⃣ CNN + MLP (Red Convolucional + Perceptrón Multicapa)

**Proyecto donde se usó:**  
Artículo científico sobre flujo glotal (Narendra et al.)

### 📥 Entrada de datos
- Segmentos fijos de **250 milisegundos**
- Señal:
  - Onda de voz cruda  
  - Onda de flujo glotal  

### 🧠 Arquitectura
Modelo profundo combinado:

1. **CNN 1D**
   - Múltiples capas convolucionales
   - Extraen patrones de la señal
2. **Max Pooling**
   - Reduce dimensionalidad
3. **Batch Normalization**
   - Estabiliza el entrenamiento
4. **Dropout**
   - Reduce sobreajuste
5. **MLP (Red Densa)**
   - Clasificación binaria:
     - Parkinson
     - Sano

---

## 2️⃣ ResNet-1D (Red Residual Unidimensional)

**Proyectos donde se usó:**  
- TFM 1 (vocal /a/)  
- TFM 2 (vocal /e/)

### 📥 Entrada de datos
- Forma de onda completa preprocesada

### 🧠 Arquitectura

Adaptación de ResNet de visión artificial a series temporales.

- 3 bloques residuales consecutivos
- Cada bloque contiene:
  - 3 capas convolucionales
    - Filtros de tamaño 8
    - Filtros de tamaño 5
    - Filtros de tamaño 3

### 🔑 Clave: Conexiones Residuales (Skip Connections)

- La entrada del bloque se suma a la salida.
- Permite:
  - Mejor flujo del gradiente
  - Redes más profundas
  - Evita pérdida de información

### 🔚 Finalización
- Global Average Pooling (GAP)
- Clasificador final

---

## 3️⃣ LSTM-FCN (Híbrido Convolucional-Recurrente)

**Proyectos donde se usó:**  
- TFM 1 (vocal /a/)  
- TFM 2 (vocal /e/)

### 📥 Entrada de datos
- Forma de onda completa

### 🧠 Arquitectura Híbrida (Dos Ramas en Paralelo)

#### 🔹 Rama FCN (Convolucional)
- 3 capas convolucionales
- Extrae patrones locales y rápidos

#### 🔹 Rama LSTM (Recurrente)
- 1 capa LSTM (memoria a corto-largo plazo)
- Se aplica **dimension shift**:
  - La secuencia larga se interpreta como
    - 1 solo paso temporal
    - Muchas variables

👉 Reduce sobreajuste en secuencias largas.

### 🔗 Fusión Final
- Concatenación de ambas ramas
- Clasificador final

---

## 4️⃣ InceptionTime

**Proyectos donde se usó:**  
- TFM 1 (vocal /a/)  
- TFM 2 (vocal /e/)

### 📥 Entrada de datos
- Forma de onda completa

### 🧠 Arquitectura

Basado en **módulos Inception**.

En lugar de usar un único tamaño de filtro:

- Aplica filtros en paralelo:
  - Tamaño 39
  - Tamaño 19
  - Tamaño 9

### 🎯 Ventajas
- Captura patrones:
  - Muy cortos
  - Intermedios
  - Largos
- Mayor robustez

### 🤝 Ensamble
- No es un solo modelo
- Son **5 redes InceptionTime independientes**
- La decisión final es el promedio

👉 Alta precisión y estabilidad.

---

## 5️⃣ CDIL-CNN (Red Convolucional Dilatada Circular)

**Proyectos donde se usó:**  
- TFM 1 (vocal /a/)  
- TFM 2 (vocal /e/)

### 📥 Entrada de datos
- Forma de onda completa
- Ideal para audios muy largos

### 🧠 Arquitectura

Diseñada para secuencias larguísimas.

- 14 capas convolucionales
- Convoluciones dilatadas simétricas
  - "Saltan" datos exponencialmente
  - Amplían el campo receptivo rápidamente

### 🔄 Circular Padding (Mezcla Circular)

- Conecta el final del audio con el principio
- Ventaja:
  - No depende de la posición exacta del fallo vocal

### 🔚 Final
- Ensamble interno de todas las posiciones
- Decisión final robusta

---

## 6️⃣ TimesNet

**Proyecto donde se usó:**  
- Exclusivamente en TFM 2 (Jardón Gómez)

### 📥 Entrada de datos
- Forma de onda completa

### 🧠 Arquitectura (Foundation Model estilo Transformer)
Modelo de última generación.

### 🔍 Enfoque Innovador

1. Usa FFT (Transformada Rápida de Fourier)
   - Detecta periodos dominantes
2. "Corta y apila" la señal
   - Convierte señal 1D en representación 2D
   - Genera parches tipo imagen
3. Aplica bloques Inception sobre 2D
   - Captura variaciones:
     - Dentro del mismo periodo
     - Entre distintos periodos

### ⚠️ Resultado en el estudio

Aunque es el más avanzado tecnológicamente:

- Fracasó en este estudio
- Requiere volúmenes masivos de datos
- No se adapta bien a datasets pequeños

---

## 📊 Resumen Comparativo Rápido

| Modelo | Tipo | Ideal para | Complejidad | Rendimiento |
|---------|-------|------------|-------------|-------------|
| CNN + MLP | Convolucional clásico | Segmentos cortos | Media | Bueno |
| ResNet-1D | Residual profundo | Señales completas | Media-Alta | Muy bueno |
| LSTM-FCN | Híbrido CNN + RNN | Patrones locales + temporales | Alta | Muy bueno |
| InceptionTime | Multi-escala + Ensamble | Señales completas | Alta | Excelente |
| CDIL-CNN | Dilatado circular | Audios muy largos | Alta | Excelente |
| TimesNet | Transformer-like | Grandes datasets | Muy alta | Bajo en este estudio |




# NUEVOS MODELOS QUE PODRÍA PROBAR QUE HAN SALIDO HACE POCO (CLAUDE)

---

## 1️⃣ 🔥 Mamba / SSM (State Space Models) — el más relevante ahora mismo

Los modelos de espacio de estados estructurados (SSMs), como:

- S4  
- S4D  
- S5  
- DSS  

han conseguido resultados notables en tareas de razonamiento de largo alcance y en clasificación de audio crudo, donde las RNNs tradicionales fallan.

### 🚀 Mamba

Mamba representa el punto álgido de esta familia.

- Resultados estado del arte en:
  - Lenguaje  
  - ADN  
  - Audio  
- Supera a bloques de atención y bloques S4 en condiciones controladas por parámetros.
- Reemplaza la atención cuadrática de los Transformers.
- Usa un núcleo SSM con:
  - Complejidad lineal real en tiempo
  - Complejidad lineal en memoria respecto a la longitud de la secuencia

👉 Especialmente adecuado para:
- Audio de larga duración  
- Tareas en tiempo real  

### 🎧 Variante específica

- **Audio Mamba (AuM)**  
  - Modelo puro de espacio de estados  
  - Sin atención  
  - Diseñado para clasificación de audio  
  - Aceptado en IEEE Signal Processing Letters (2024)

### 🎯 ¿Por qué es ideal para tu caso?

- Las vocales sostenidas son secuencias largas.
- Mamba está diseñado exactamente para:
  - Captar dependencias a muy largo plazo
  - Sin coste cuadrático como los Transformers

---

## 2️⃣ 🧠 SincNet + variantes — interpretable y fisiológicamente motivado

SincNet reemplaza la primera capa convolucional por:

- Un banco de filtros pasa-banda
- Parametrizados como funciones **sinc**
- Solo se aprenden:
  - Frecuencia de corte inferior
  - Frecuencia de corte superior

### 🔎 Ventajas

- Filtros más interpretables
- Menor número de parámetros

### 🎤 Relevancia para Parkinson

Las frecuencias aprendidas podrían coincidir directamente con:

- Rangos asociados al temblor vocal  
- Jitter / Shimmer patológico  

### 🔬 Variante ERB

SincNet con inicialización **ERB (Equivalent Rectangular Bandwidth)**:

- Mejora la extracción de:
  - Pitch  
  - Formantes  
- Asigna más filtros a la región de baja frecuencia del espectro

---

## 3️⃣ 🏗️ RawNet2 / RawNet3 — arquitecturas end-to-end para voz cruda

Arquitecturas diseñadas específicamente para procesar audio crudo.

### 🔹 RawNet2
- Emplea capas Sinc
- Extrae características directamente desde la forma de onda

### 🔹 RawNet3
- Usa bancos de filtros analíticos parametrizados
- Tres bloques con conexiones residuales

### 🧩 Componentes comunes

- Bloques residuales  
- GRU  
- Pooling estadístico  

👉 Transferirlas a clasificación de Parkinson sería una contribución novedosa y justificable.

---

## 4️⃣ ⚡ xLSTM aplicado a audio (Audio xLSTM)

Arquitectura de 2024 para:

- Aprender representaciones de audio auto-supervisadas

Es una evolución del LSTM clásico hacia:

- Arquitecturas de espacio de estados mejoradas

### 🔄 xLSTM

- Revisión profunda del LSTM
- Supera limitaciones de memoria del LSTM estándar

👉 Dado que LSTM-FCN ya dio buenos resultados, xLSTM sería la evolución natural a explorar.

---

## 5️⃣ 🔀 Conformer 1D / Temporal Conformer

El Conformer combina:

- Convoluciones locales  
- Atención global  

en un mismo bloque.

### 🧠 Adaptación a 1D

- Trabaja directamente sobre la onda
- Sin necesidad de espectrograma
- Captura:
  - Patrones locales
  - Dependencias de largo alcance

Es estado del arte en reconocimiento de voz y adaptable a esta tarea.

---

# 📊 Tabla comparativa rápida

| Modelo | Novedad | Ventaja para Parkinson / voz | Dificultad de implementación |
|--------|----------|------------------------------|------------------------------|
| Mamba / SSM | ⭐⭐⭐⭐⭐ | Ideal para secuencias muy largas en 1D | Media-alta |
| SincNet | ⭐⭐⭐ | Filtros interpretables, captura pitch/formantes | Baja |
| RawNet2/3 | ⭐⭐⭐⭐ | Diseñado para voz cruda end-to-end | Media |
| xLSTM | ⭐⭐⭐⭐ | Evolución directa del LSTM-FCN ya probado | Media |
| Conformer 1D | ⭐⭐⭐⭐ | Local + global sobre la onda cruda | Media-alta |






































# ResNet-1D



# LSTM-FCN



# InceptionTime



# CDIL-CNN



# TimesNet



# Transformer temporal



# TCT



# WaveNet-like classifier



# ConvNeXt-1D



# WavEmbed + MLP/Vit-lite