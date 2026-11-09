import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "persian"))
from leakage_check import ngrams, tokens  # noqa: E402


def test_tokens_normalise_persian_variants():
    # Arabic yeh/kaf, ZWNJ and Persian digits must not hide a copied question
    a = tokens("كدام داروي ضد‌انعقاد در ۲ ساعت")
    b = tokens("کدام داروی ضد انعقاد در 2 ساعت")
    assert a == b


def test_planted_question_is_found_by_ngrams():
    q = tokens("بیمار ۴۵ ساله با درد قفسه سینه و تنگی نفس به اورژانس مراجعه کرده است")
    doc = tokens("مقدمه متن. بيمار 45 ساله با درد قفسه سينه و تنگي نفس به اورژانس مراجعه کرده است. ادامه")
    qg = ngrams(q, 8)
    assert qg and qg <= ngrams(doc, 8)
