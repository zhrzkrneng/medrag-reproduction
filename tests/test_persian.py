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


def test_clean_text_removes_bidi_marks_without_merging_words():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from medrag_repro.persian import clean_text
    raw = "‫در‬‫میاستنی‬ ‫گراویس‪،‬‬ ‫اختلال‬"
    assert clean_text(raw) == "در میاستنی گراویس ، اختلال"


def test_gaokerena_prompt_and_parser():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "persian"))
    from medrag_repro.templates import build_messages_gk
    from gk_parse import parse
    opts = {"A": "الف", "B": "ب", "C": "ج", "D": "د"}
    m = build_messages_gk("سؤال؟", opts, None)
    assert len(m) == 1 and m[0]["role"] == "user"
    assert m[0]["content"] == ("\nبه پرسش زیر به پاسخ بدهید و تنها از زبان فارسی استفاده کنید!\n## پرسش\nسؤال؟ \n\n\n"
                               "## گزینه ها\n[A] : الف\n[B] : ب\n[C] : ج\n[D] : د \n\n\n\n\n")
    assert "## متن\nCTX" in build_messages_gk("q", opts, "CTX")[0]["content"]
    assert parse("گزینه [B] درست است.\n\nگزینه [A] نادرست است") == "B"
    assert parse("**پاسخ صحیح:** C") == "C"
    assert parse("بررسی گزینه‌ها: [A] ... [B] ...\nبنابراین، پاسخ صحیح [D] است.") == "D"
    assert parse("برای پاسخ به این پرسش، باید دقت کنیم. گزینه ۲ مناسب است") == "B"
    assert parse("هیچ‌کدام") is None


def test_holm_and_bh_corrections():
    from stats_all import bh, holm
    ps = [0.01, 0.04, 0.03, 0.5]
    # Holm: sorted 0.01*4, 0.03*3, 0.04*2, 0.5*1, made monotone
    assert [round(x, 4) for x in holm(ps)] == [0.04, 0.09, 0.09, 0.5]
    # BH: 0.01*4/1, 0.03*4/2, 0.04*4/3, 0.5*4/4, made monotone from the top
    assert [round(x, 4) for x in bh(ps)] == [0.04, 0.0533, 0.0533, 0.5]
