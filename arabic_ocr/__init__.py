"""arabic-ocr-crnn — تعرّف ضوئي على النصوص العربية بمعمارية CNN-LSTM وخسارة CTC."""

from .charset import CHARS, VOCAB_SIZE, decode, encode
from .metrics import cer, corpus_metrics, wer
from .synth import generate_dataset, render, save_samples

__version__ = "0.1.0"
__all__ = ["CHARS", "VOCAB_SIZE", "cer", "corpus_metrics", "decode", "encode",
           "generate_dataset", "render", "save_samples", "wer"]
