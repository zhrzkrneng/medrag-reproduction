import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medrag_repro.evaluate import official_prediction, strict_prediction, score
from medrag_repro.retrieval import rrf
from medrag_repro.templates import build_messages, format_snippet


def test_official_parser_matches_mirage():
    assert official_prediction('{"step_by_step_thinking": "x", "answer_choice": "C"}') == "C"
    assert official_prediction('{"answer_choice": "B. Insulin"}') == "B"
    # official fallback: unparseable -> "A"
    assert official_prediction("I am not sure") == "A"


def test_strict_parser_flags_failures():
    assert strict_prediction('{"answer_choice": "D"}') == "D"
    assert strict_prediction("I am not sure") is None


def test_score_counts():
    recs = [{"answer": "A", "response": "no idea"}, {"answer": "B", "response": '"answer_choice": "B"'}]
    s = score(recs)
    assert s["acc_official"] == 1.0      # fallback "A" happens to be right
    assert s["acc_strict"] == 0.5 and s["parse_failures"] == 1


def test_rrf_matches_medrag_formula():
    ids, sc = rrf([["a", "b"], ["b", "c"]], k=3, rrf_k=100)
    assert ids[0] == "b" and abs(sc[0] - (1 / 102 + 1 / 101)) < 1e-12
    assert ids[1:] == ["a", "c"]          # tie: first-seen order kept


def test_prompt_format():
    m = build_messages("Q?", {"B": "y", "A": "x"}, None)
    assert "A. x\nB. y" in m[1]["content"]
    assert format_snippet(0, {"title": "T", "content": "C"}) == "Document [0] (Title: T) C"
