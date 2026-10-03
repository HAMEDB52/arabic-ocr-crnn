import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from arabic_ocr.charset import VOCAB_SIZE, decode  # noqa: E402
from arabic_ocr.data import encode_labels  # noqa: E402
from arabic_ocr.model import build_model, ctc_loss, greedy_decode  # noqa: E402
from arabic_ocr.synth import IMG_HEIGHT, IMG_WIDTH  # noqa: E402


def test_model_output_shape():
    model = build_model()
    out = model.predict(np.zeros((2, IMG_HEIGHT, IMG_WIDTH, 1), dtype="float32"), verbose=0)
    assert out.shape == (2, IMG_WIDTH // 4, VOCAB_SIZE)


def test_time_steps_exceed_label_length():
    assert IMG_WIDTH // 4 > 24, "عدد الخطوات الزمنية يجب أن يفوق أطول تسمية"


def test_ctc_loss_is_finite_and_positive():
    y_pred = tf.random.normal((2, IMG_WIDTH // 4, VOCAB_SIZE))
    y_true = tf.constant(encode_labels(["فاتورة", "رقم الحساب"], 12))
    loss = ctc_loss(y_true, y_pred)
    assert np.isfinite(loss.numpy()) and loss.numpy() > 0


def test_greedy_decode_removes_blanks_and_repeats():
    # بناء منطق مصطنع: الحرف 1 مكرر ثم فراغ ثم الحرف 2
    logits = np.full((1, 6, VOCAB_SIZE), -10.0, dtype="float32")
    for t, idx in enumerate([1, 1, 0, 2, 2, 0]):
        logits[0, t, idx] = 10.0
    assert greedy_decode(logits)[0] == [1, 2]


def test_labels_are_zero_padded():
    padded = encode_labels(["رقم", "فاتورة ضريبية"], 16)
    assert padded.shape == (2, 16)
    assert padded[0][-1] == 0
    assert decode(padded[1]).strip() == "فاتورة ضريبية"
