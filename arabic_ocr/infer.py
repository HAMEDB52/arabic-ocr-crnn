"""الاستدلال: من صورة إلى نص."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

from .charset import decode
from .model import ctc_loss, greedy_decode
from .synth import IMG_HEIGHT, IMG_WIDTH


def load_model(path: str | Path) -> keras.Model:
    return keras.models.load_model(str(path), custom_objects={"ctc_loss": ctc_loss}, compile=False)


# ارتفاع الحبر والهامش الأيمن كما يظهران في صور التدريب (synth.render): سطر بخط 18–24 على لوح مضاعف ثم تصغير ×0.5
TEXT_HEIGHT = 12
RIGHT_MARGIN = 4


def preprocess(path: str | Path, *, width: int = IMG_WIDTH, height: int = IMG_HEIGHT) -> np.ndarray:
    """تهيئة صورة خارجية بحيث تطابق توزيع صور التدريب.

    صورة بمقاس النموذج (256×32) تُمرَّر كما هي. غير ذلك: قص على الحبر فقط (عتبة تتجاهل الضوضاء)،
    ثم تحجيم يجعل ارتفاع النص قريباً من ارتفاعه في التدريب، ولصق بمحاذاة يمين ومركز رأسي.
    (القص ثم التمديد لملء الارتفاع كاملاً يُضخّم النص أكبر بكثير مما رآه النموذج، فتنهار الدقة.)
    """
    img = Image.open(str(path)).convert("L")
    if img.size == (width, height):
        return (np.asarray(img, dtype=np.float32) / 255.0)[None, ..., None]
    ink = img.point(lambda p: 255 if p < 128 else 0)
    bbox = ink.getbbox()
    if bbox:
        img = img.crop(bbox)
    scale = min(TEXT_HEIGHT / max(img.height, 1), (width - 2 * RIGHT_MARGIN) / max(img.width, 1))
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    out = Image.new("L", (width, height), color=255)
    out.paste(img, (width - RIGHT_MARGIN - img.width, (height - img.height) // 2))
    return (np.asarray(out, dtype=np.float32) / 255.0)[None, ..., None]


def predict_batch(model: keras.Model, images: np.ndarray) -> list[str]:
    logits = model.predict(images, verbose=0)
    return [decode(ids) for ids in greedy_decode(logits)]


def predict_image(model: keras.Model, path: str | Path) -> str:
    return predict_batch(model, preprocess(path))[0]
