from arabic_ocr.metrics import cer, corpus_metrics, levenshtein, wer


def test_levenshtein_basic():
    assert levenshtein("كتاب", "كتاب") == 0
    assert levenshtein("كتاب", "كتب") == 1


def test_cer_is_zero_for_identical():
    assert cer("فاتورة ضريبية", "فاتورة ضريبية") == 0.0


def test_cer_counts_character_edits():
    assert cer("رقم", "رقن") == 1 / 3


def test_wer_counts_word_edits():
    assert wer("فاتورة ضريبية جديدة", "فاتورة ضريبية") == 1 / 3


def test_corpus_metrics_aggregate():
    refs = ["فاتورة", "رقم الحساب"]
    hyps = ["فاتورة", "رقم الحسب"]
    m = corpus_metrics(refs, hyps)
    assert m["samples"] == 2
    assert m["exact_match"] == 0.5
    assert 0 < m["cer"] < 0.2
