"""معمارية CRNN: التفافات لاستخلاص السمات ثم LSTM ثنائي الاتجاه مع خسارة CTC.

لماذا CTC؟ لأن الأسطر النصية متغيرة الطول ولا تتوفر محاذاة بين البكسلات والحروف،
فتتعلم الشبكة المحاذاة ضمنياً بدل تجهيزها يدوياً.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .charset import BLANK, VOCAB_SIZE
from .synth import IMG_HEIGHT, IMG_WIDTH

TIME_STEPS = IMG_WIDTH // 4      # نتيجة تجميعتين أفقيتين


def build_model(*, height: int = IMG_HEIGHT, width: int = IMG_WIDTH,
                vocab_size: int = VOCAB_SIZE, rnn_units: int = 128) -> keras.Model:
    inputs = keras.Input(shape=(height, width, 1), name="image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)                       # 16 x W/2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)                       # 8  x W/4
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 1))(x)                       # 4  x W/4
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 1))(x)                       # 2  x W/4

    # (دفعة، ارتفاع، عرض، قنوات) -> (دفعة، خطوات زمنية = العرض، سمات)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((width // 4, -1))(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    logits = layers.Dense(vocab_size, name="logits")(x)
    return keras.Model(inputs, logits, name="arabic_crnn_ctc")


def ctc_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """خسارة CTC على تسميات محشوّة بالأصفار (0 = blank = الحشو)."""
    y_true = tf.cast(y_true, tf.int32)
    label_length = tf.reduce_sum(tf.cast(y_true > 0, tf.int32), axis=-1)
    batch = tf.shape(y_pred)[0]
    logit_length = tf.fill([batch], tf.shape(y_pred)[1])
    loss = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=label_length,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=BLANK,
    )
    return tf.reduce_mean(loss)


def greedy_decode(logits) -> list[list[int]]:
    """فك ترميز جشِع: إزالة التكرارات ثم الرموز الفارغة."""
    logits = tf.convert_to_tensor(logits)
    time_major = tf.transpose(logits, (1, 0, 2))
    seq_len = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    decoded, _ = tf.nn.ctc_greedy_decoder(time_major, seq_len, blank_index=BLANK)
    dense = tf.sparse.to_dense(decoded[0], default_value=-1).numpy()
    return [[i for i in row if i >= 0] for row in dense]
