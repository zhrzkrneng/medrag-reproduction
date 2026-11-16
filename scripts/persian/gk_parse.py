"""Answer extraction for Gaokerena's own free-form prompt (no JSON).

Gaokerena's evaluation (github.com/Mehrdadghassabi/Gaokerena-V, evaluation/multiple_choice_qa)
labels the chosen option by hand ("append AI_chosen_answer column manually"). We need an
automatic rule; it is validated against their hand labels (scripts/persian/gk_parse_check.py).
Returns "A"-"D" or None (= their "invalid").
"""
import re

L = "ABCD"
_FA_DIG = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_OPT = r"(?:\[\s*([A-D])\s*\]|\(\s*([A-D])\s*\)|\b([A-D])\b)"
_POS = r"(?:صحیح|درست|صحيح|مناسب|بهترین|نهایی)"
_PATTERNS = [
    # "پاسخ صحیح/درست/نهایی ... [B]" / "جواب ... گزینه B"
    rf"(?:پاسخ|جواب)\s*(?:{_POS}|:)[^\n\[\]A-D]{{0,30}}?{_OPT}",
    # "گزینه [B] صحیح/درست است"
    rf"گزینه\s*{_OPT}[^\n\[]{{0,60}}?(?:{_POS})\s*(?:است|می\s?باشد|هست)",
    # "[B] ... صحیح است" on the same line
    rf"{_OPT}[^\n]{{0,80}}?(?:{_POS})\s*(?:است|می\s?باشد)",
]


FA_LET = {"الف": "A", "ب": "B", "ج": "C", "د": "D"}
_ANY = re.compile(r"\[\s*([A-D])\s*\]|\(\s*([A-D])\s*\)|گزینه\s*([A-D])\b|(?<![A-Za-z])([A-D])(?![A-Za-z])")
_FAOPT = re.compile(r"(?:گزینه|پاسخ\s*(?:صحیح|درست|نهایی)?\s*:?)\s*\(?\s*(الف|ب|ج|د)(?![؀-ۿ])")


def _letter(m):
    return next(g for g in m.groups() if g)


def _norm(s):
    s = str(s).replace("ي", "ی").replace("ك", "ک").replace("‌", "")
    return re.sub(r"[\s\.،,:؛;!?«»\"'()\[\]]+", "", s)


def parse(answer, options=None):
    if not answer:
        return None
    t = str(answer).translate(_FA_DIG).replace("**", "").replace("__", "")
    neg = re.compile(r"(نادرست|غلط|اشتباه|صحیح نیست|درست نیست)")
    pos = []
    for p in _PATTERNS:
        for m in re.finditer(p, t):
            if not neg.search(t[m.start(): m.end() + 15]):
                pos.append((m.start(), _letter(m)))
    if pos:  # an explicit "the answer is X" statement; the conclusion (last one) wins
        return max(pos)[1]
    cands = []
    m = _ANY.search(t)
    if m:
        cands.append((m.start(), _letter(m)))
    m = _FAOPT.search(t)
    if m:
        cands.append((m.start(), FA_LET[m.group(1)]))
    m = re.search(r"گزینه\s*([1-4])(?!\d)", t)
    if m:
        cands.append((m.start(), L[int(m.group(1)) - 1]))
    if options:
        nt = _norm(t)
        for k, v in options.items():
            nv = _norm(v)
            if len(nv) >= 3 and nv in nt:
                cands.append((None, k, nt.index(nv)))
    lett = [c for c in cands if c[0] is not None]
    if lett:
        return min(lett)[1]
    txt = sorted(cands, key=lambda c: c[2])
    if txt:
        return txt[0][1]
    return None
