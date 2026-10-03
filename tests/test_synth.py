import numpy as np

from arabic_ocr.charset import is_supported
from arabic_ocr.synth import IMG_HEIGHT, IMG_WIDTH, generate_dataset, random_text


def test_dataset_shapes_and_range():
    images, texts = generate_dataset(6, seed=3)
    assert images.shape == (6, IMG_HEIGHT, IMG_WIDTH, 1)
    assert images.dtype == np.float32
    assert images.min() >= 0.0 and images.max() <= 1.0
    assert len(texts) == 6


def test_generated_text_is_within_charset_and_length():
    import random
    rng = random.Random(1)
    for _ in range(50):
        text = random_text(rng, max_chars=20)
        assert is_supported(text)
        assert 0 < len(text) <= 20


def test_images_are_not_blank():
    images, _ = generate_dataset(4, seed=5)
    for img in images:
        assert img.min() < 0.5, "يجب أن تحتوي الصورة على بكسلات نص داكنة"


def test_dataset_is_reproducible():
    a_img, a_txt = generate_dataset(5, seed=42)
    b_img, b_txt = generate_dataset(5, seed=42)
    assert a_txt == b_txt
    assert np.allclose(a_img, b_img)
