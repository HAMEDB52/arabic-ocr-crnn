"""تحضير بيانات التدريب: ترميز التسميات والحشو وبناء tf.data."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .charset import encode
from .synth import generate_dataset


def encode_labels(texts: list[str], max_len: int | None = None) -> np.ndarray:
    sequences = [encode(t) for t in texts]
    max_len = max_len or max((len(s) for s in sequences), default=1)
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)   # 0 = حشو = blank
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq[:max_len]
    return padded


def make_datasets(n_train: int = 3000, n_val: int = 400, *, batch_size: int = 32,
                  seed: int = 7, max_chars: int = 24):
    x_train, y_train_txt = generate_dataset(n_train, seed=seed, max_chars=max_chars)
    x_val, y_val_txt = generate_dataset(n_val, seed=seed + 1000, max_chars=max_chars)

    max_len = max(max(len(t) for t in y_train_txt), max(len(t) for t in y_val_txt))
    y_train = encode_labels(y_train_txt, max_len)
    y_val = encode_labels(y_val_txt, max_len)

    train = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
             .shuffle(min(2048, n_train), seed=seed)
             .batch(batch_size)
             .prefetch(tf.data.AUTOTUNE))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    return train, val, (x_val, y_val_txt)
