"""الاستدلال: من صورة إلى نص."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from .charset import decode
from .model import ctc_loss, greedy_decode
from .synth import IMG_HEIGHT, IMG_WIDTH, _fit


def load_model(path: str | Path) -> keras.Model:
    return keras.models.load_model(str(path), custom_objects={"ctc_loss": ctc_loss}, compile=False)


def preprocess(path: str | Path, *, width: int = IMG_WIDTH, height: int = IMG_HEIGHT) -> np.ndarray:
    """تهيئة صورة خارجية: تدرّج رمادي، قص الهوامش، تحجيم مع حفظ النسبة."""
    img = Image.open(str(path)).convert("L")
    inverted = Image.eval(img, lambda p: 255 - p)
    bbox = inverted.getbbox()
    if bbox:
        img = img.crop(bbox)
    arr = np.asarray(_fit(img, width, height), dtype=np.float32) / 255.0
    return arr[None, ..., None]


def predict_batch(model: keras.Model, images: np.ndarray) -> list[str]:
    logits = model.predict(images, verbose=0)
    return [decode(ids) for ids in greedy_decode(logits)]


def predict_image(model: keras.Model, path: str | Path) -> str:
    return predict_batch(model, preprocess(path))[0]
