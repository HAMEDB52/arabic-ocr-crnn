"""مقاييس التعرف: معدل خطأ المحرف (CER) ومعدل خطأ الكلمة (WER)."""
from __future__ import annotations


def levenshtein(a, b) -> int:
    """مسافة التحرير — أساس كلا المقياسين."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def corpus_metrics(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """تجميع على مستوى المجموعة — بالتجميع لا بمتوسط النسب."""
    char_errors = sum(levenshtein(r, h) for r, h in zip(references, hypotheses))
    char_total = sum(len(r) for r in references) or 1
    word_errors = sum(levenshtein(r.split(), h.split()) for r, h in zip(references, hypotheses))
    word_total = sum(len(r.split()) for r in references) or 1
    exact = sum(r == h for r, h in zip(references, hypotheses))
    return {
        "cer": char_errors / char_total,
        "wer": word_errors / word_total,
        "exact_match": exact / (len(references) or 1),
        "samples": len(references),
    }
