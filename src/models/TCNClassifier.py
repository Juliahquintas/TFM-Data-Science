"""
TCNClassifier.py
================
Clasificador basado en Temporal Convolutional Network (TCN) con activación
WaveNet (tanh × sigmoid) para audio raw.

Arquitectura (inspirada en https://github.com/philipperemy/keras-tcn):
  Input (samples,)
    → Reshape (samples, 1)
    → Conv1D inicial  [proyección de canales]
    → N bloques residuales con dilatación exponencial + skip connections
    → Suma de skip connections → ReLU
    → GlobalAveragePooling1D + GlobalMaxPooling1D
    → Concatenate → Dense(units) → Dropout → Dense(n_classes, softmax)

Diferencias respecto al TCN original del documento:
  - Sin capa Embedding (entrada es audio raw, no índices de texto).
  - Padding 'causal' por defecto (garantiza causalidad temporal).
  - Cabeza de clasificación con pooling dual (avg + max) igual al modelo
    del documento (GlobalAveragePooling1D + GlobalMaxPooling1D).
  - Compatible con TensorFlow/Keras 3.x (sin keras.engine.topology).

Uso:
    clf = TCNClassifier(
        input_shape   = (8000,),   # nº de muestras de audio
        n_classes     = 2,         # Control vs Patológicas
        nb_filters    = 64,
        kernel_size   = 3,
        dilations     = [1, 2, 4, 8, 16, 32],
        nb_stacks     = 1,
        dropout_rate  = 0.1,
    )
    clf.model.summary()
    clf.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input


# ---------------------------------------------------------------------------
# Bloque de activación WaveNet
# ---------------------------------------------------------------------------

def wavenet_activation(x_tanh: tf.Tensor, x_sigm: tf.Tensor) -> tf.Tensor:
    """
    Activación gated: tanh(x) ⊙ σ(x).

    Dos convoluciones separadas (una para tanh, otra para sigmoid) permiten
    al modelo controlar cuánta información "pasa" en cada timestep, de forma
    análoga a las puertas de una LSTM pero implementado con Conv1D.

    Args:
        x_tanh: Salida de la convolución dilatada para la rama tanh.
        x_sigm: Salida de la convolución dilatada para la rama sigmoid.
    Returns:
        Tensor gate: tanh(x_tanh) * sigmoid(x_sigm)
    """
    return layers.Multiply()([
        layers.Activation('tanh')(x_tanh),
        layers.Activation('sigmoid')(x_sigm),
    ])


# ---------------------------------------------------------------------------
# Bloque residual
# ---------------------------------------------------------------------------

def residual_block(
    x: tf.Tensor,
    nb_filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float = 0.0,
    padding: str = 'causal',
    name: str = '',
) -> tuple:
    """
    Bloque residual TCN con activación WaveNet.

    Estructura interna:
        x_in ──┬──► Conv1D(tanh)  ┐
               │   Conv1D(sigmoid) ├─► gate ─► SpatialDropout ─► Conv1D(1×1) ──► skip_out
               │                  ┘                                    │
               └────────────────────────────────────────────── add ───► res_out

    La convolución 1×1 (res_x) proyecta al mismo número de canales que la
    entrada para que la suma residual sea dimensionalmente compatible.

    Args:
        x:            Tensor de entrada, shape (batch, timesteps, nb_filters).
        nb_filters:   Nº de filtros de las convoluciones dilatadas.
        kernel_size:  Tamaño del kernel convolucional.
        dilation_rate: Factor de dilatación (potencia de 2).
        dropout_rate: Fracción de unidades a anular (SpatialDropout1D).
        padding:      'causal' (recomendado) o 'same'.
        name:         Prefijo para los nombres de las capas.
    Returns:
        res_out  (tf.Tensor): Salida residual = entrada + proyección gated.
        skip_out (tf.Tensor): Salida de skip connection (para sumar al final).
    """
    original_x = x

    # Dos convoluciones dilatadas paralelas: una para tanh, otra para sigmoid
    conv_tanh = layers.Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        name=f'{name}_dilated_conv_{dilation_rate}_tanh',
    )(x)

    conv_sigm = layers.Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        name=f'{name}_dilated_conv_{dilation_rate}_sigm',
    )(x)

    # Activación gated WaveNet
    gated = wavenet_activation(conv_tanh, conv_sigm)

    # Dropout espacial (anula canales enteros, no muestras individuales)
    gated = layers.SpatialDropout1D(
        dropout_rate,
        name=f'{name}_dropout_{dilation_rate}',
    )(gated)

    # Conv 1×1: proyecta la salida gated → nb_filters canales (skip out)
    skip_out = layers.Conv1D(
        nb_filters, kernel_size=1, padding='same',
        name=f'{name}_skip_{dilation_rate}',
    )(gated)

    # Conexión residual: suma la entrada original con skip_out
    res_out = layers.Add(name=f'{name}_residual_{dilation_rate}')(
        [original_x, skip_out]
    )

    return res_out, skip_out


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class TCNClassifier:
    """
    Clasificador TCN para audio raw.

    Parámetros
    ----------
    input_shape : tuple
        Shape del audio de entrada SIN batch dim, p. ej. (8000,) ó (16000,).
    n_classes : int
        Número de clases de salida (2 para clasificación binaria).
    nb_filters : int
        Número de filtros en cada capa convolucional (ancho del TCN).
    kernel_size : int
        Tamaño del kernel de las convoluciones dilatadas.
    dilations : list[int]
        Lista de tasas de dilatación. Por defecto [1, 2, 4, 8, 16, 32].
        Cada valor DEBE ser potencia de 2. La campo receptivo efectivo es:
            RF = 1 + (kernel_size - 1) * sum(dilations) * nb_stacks
    nb_stacks : int
        Nº de veces que se repite la secuencia de bloques de dilación.
        Más stacks = más campo receptivo, más parámetros.
    dropout_rate : float
        Fracción de dropout en SpatialDropout1D (0.0 = sin dropout).
    dense_units : int
        Nº de neuronas de la capa Dense intermedia antes de la salida.
    padding : str
        'causal' (garantiza causalidad) o 'same'.
    name : str
        Prefijo de nombre para todas las capas del modelo.
    """

    def __init__(
        self,
        input_shape: tuple = (8000,),
        n_classes: int = 2,
        nb_filters: int = 64,
        kernel_size: int = 3,
        dilations: list = None,
        nb_stacks: int = 1,
        dropout_rate: float = 0.1,
        dense_units: int = 32,
        padding: str = 'causal',
        name: str = 'tcn',
    ):
        if dilations is None:
            dilations = [1, 2, 4, 8, 16, 32]

        self.input_shape  = input_shape
        self.n_classes    = n_classes
        self.nb_filters   = nb_filters
        self.kernel_size  = kernel_size
        self.dilations    = dilations
        self.nb_stacks    = nb_stacks
        self.dropout_rate = dropout_rate
        self.dense_units  = dense_units
        self.padding      = padding
        self.name         = name

        # Campo receptivo (informativo)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations) * nb_stacks

        self.model = self._build()

    def _build(self) -> Model:
        """Construye y devuelve el modelo Keras."""

        # ── Entrada ──────────────────────────────────────────────────────────
        inp = Input(shape=self.input_shape, name='audio_input')

        # Reshape: (batch, samples) → (batch, samples, 1)
        # Necesario porque Conv1D espera 3D: (batch, timesteps, channels)
        x = layers.Reshape((*self.input_shape, 1), name='reshape_input')(inp)

        # ── Proyección inicial de canales ─────────────────────────────────────
        # Conv1D 1×1 lleva 1 canal → nb_filters, igualando dimensiones
        # para que la primera suma residual funcione correctamente.
        x = layers.Conv1D(
            self.nb_filters, kernel_size=1,
            padding=self.padding,
            name=f'{self.name}_input_projection',
        )(x)

        # ── Bloques residuales TCN ────────────────────────────────────────────
        skip_connections = []
        for stack_idx in range(self.nb_stacks):
            for dilation in self.dilations:
                x, skip = residual_block(
                    x,
                    nb_filters   = self.nb_filters,
                    kernel_size  = self.kernel_size,
                    dilation_rate= dilation,
                    dropout_rate = self.dropout_rate,
                    padding      = self.padding,
                    name         = f'{self.name}_s{stack_idx}',
                )
                skip_connections.append(skip)

        # ── Suma de todas las skip connections ───────────────────────────────
        # En lugar de usar sólo la última capa residual, se suman las
        # contribuciones de TODOS los bloques. Esto permite al modelo
        # integrar información a múltiples escalas temporales simultáneamente.
        if len(skip_connections) > 1:
            x = layers.Add(name=f'{self.name}_skip_sum')(skip_connections)
        else:
            x = skip_connections[0]

        x = layers.Activation('relu', name=f'{self.name}_post_relu')(x)

        # ── Cabeza de clasificación ───────────────────────────────────────────
        # Pooling dual: captura tanto la media (tendencia global) como
        # el máximo (activaciones más salientes) de la secuencia temporal.
        avg_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        max_pool = layers.GlobalMaxPooling1D(name='global_max_pool')(x)

        # Concatenar ambas representaciones → vector de 2*nb_filters
        x = layers.Concatenate(name='pool_concat')([avg_pool, max_pool])

        # Dense intermedia con activación ReLU
        x = layers.Dense(self.dense_units, activation='relu', name='dense_hidden')(x)
        x = layers.Dropout(self.dropout_rate, name='dense_dropout')(x)

        # Capa de salida: softmax para clasificación multi-clase
        out = layers.Dense(
            self.n_classes, activation='softmax', name='output'
        )(x)

        return Model(inputs=inp, outputs=out, name=self.name)

    def predict(self, X: 'np.ndarray', batch_size: int = 32) -> 'np.ndarray':
        """
        Inferencia sobre un array de formas de onda.

        Args:
            X: np.ndarray de shape (N, n_samples).
            batch_size: Tamaño de lote para inferencia.
        Returns:
            Probabilidades softmax, shape (N, n_classes).
        """
        return self.model.predict(X, batch_size=batch_size, verbose=0)

    def receptive_field_info(self) -> str:
        """Devuelve un resumen del campo receptivo efectivo."""
        rf_ms_16k = self.receptive_field / 16000 * 1000
        return (
            f"Campo receptivo: {self.receptive_field} muestras "
            f"≈ {rf_ms_16k:.1f} ms @ 16 kHz"
        )
