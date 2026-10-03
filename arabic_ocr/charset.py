"""مجموعة المحارف وترميزها — الفهرس 0 محجوز لرمز CTC الفارغ (blank)."""
from __future__ import annotations

ARABIC_LETTERS = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
EXTRA_FORMS = "أإآءئؤىة"
DIGITS = "0123456789"
PUNCT = " ،.:/-"

CHARS = ARABIC_LETTERS + EXTRA_FORMS + DIGITS + PUNCT
BLANK = 0                       # فهرس CTC الفارغ
VOCAB_SIZE = len(CHARS) + 1     # +1 للفراغ

_CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARS)}
_ID_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}


def encode(text: str) -> list[int]:
    """تحويل النص إلى أرقام — تُتجاهل المحارف خارج المجموعة."""
    return [_CHAR_TO_ID[c] for c in text if c in _CHAR_TO_ID]


def decode(ids) -> str:
    """تحويل الأرقام إلى نص، مع تجاهل الفراغ والحشو."""
    return "".join(_ID_TO_CHAR.get(int(i), "") for i in ids if int(i) > 0)


def is_supported(text: str) -> bool:
    return all(c in _CHAR_TO_ID for c in text)
