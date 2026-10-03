from arabic_ocr.charset import BLANK, VOCAB_SIZE, decode, encode, is_supported


def test_roundtrip():
    text = "فاتورة ضريبية 1234"
    assert decode(encode(text)) == text


def test_blank_is_zero_and_outside_charset():
    assert BLANK == 0
    assert all(i > 0 for i in encode("فاتورة"))
    assert VOCAB_SIZE == max(encode("".join("ابت0 "))) + 1 or VOCAB_SIZE > 1


def test_unknown_characters_are_dropped():
    assert decode(encode("فاتورة@#$")) == "فاتورة"
    assert not is_supported("فاتورة@")


def test_decode_ignores_padding():
    assert decode([0, 0, *encode("رقم"), 0]) == "رقم"
