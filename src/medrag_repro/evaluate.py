"""Answer extraction and accuracy, following the official MIRAGE evaluator.

Source of ``locate_answer`` / ``locate_answer4pub_llama``:
https://github.com/Teddy-XiongGZ/MIRAGE/blob/392943af99cd94cafd50a0de2e7fca24bbf65494/src/utils.py

Important quirk of the official parser, kept here for comparability: when no
option letter can be found it returns "A". We therefore ALSO report a strict
parse-failure count so this fallback cannot silently inflate or deflate
accuracy.
"""
import json
import math
import re

ANSWERS = ["A", "B", "C", "D"]


def locate_answer(sentence: str) -> str:
    for pat in [r"^\s*(A|B|C|D)$", r"^\s*(A|B|C|D) or", r"^\s*(A|B|C|D) and",
                r"^\s*(A|B|C|D)/", r"^\s*(A|B|C|D),"]:
        ans = re.findall(pat, sentence)
        if ans:
            return ans[0].upper()
    ans = re.findall(r"[Oo]ption (A|B|C|D)", sentence)
    if ans:
        return ans[0]
    for pat in [r":\s*(A|B|C|D)", r"^\s*(A|B|C|D)\.", r"^\s*(A|B|C|D)\"", r"^\s*(A|B|C|D):"]:
        ans = re.findall(pat, sentence)
        if ans:
            return ans[0].upper()
    return "A"


def locate_answer4pub_llama(sentence: str) -> str:
    sentence = sentence.split("Answer:")[-1]
    for pat in [r"[Oo]ption (A|B|C|D)", r"OPTION (A|B|C|D)"]:
        ans = re.findall(pat, sentence)
        if ans:
            return ans[0]
    for pat in [r"^\s*(A|B|C|D)\"", r"^\s*(A|B|C|D):"]:
        ans = re.findall(pat, sentence)
        if ans:
            return ans[0].upper()
    return "A"


def official_prediction(response: str, pmc_llama: bool = False) -> str:
    """Exactly what MIRAGE/src/evaluate.py does with one response string."""
    fn = locate_answer4pub_llama if pmc_llama else locate_answer
    return fn(response.split('"answer_choice": "')[-1].strip())


def strict_prediction(response: str) -> str | None:
    """Letter only if the model clearly produced an answer_choice; else None."""
    m = re.search(r'"answer_choice"\s*:\s*"\s*([A-D])\b', response)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", response, flags=re.S)
    if m:
        try:
            val = str(json.loads(m.group(0)).get("answer_choice", "")).strip()
            if val[:1] in ANSWERS:
                return val[:1]
        except (json.JSONDecodeError, AttributeError):
            pass
    return None


def score(records: list[dict], pmc_llama: bool = False) -> dict:
    """records: [{'answer': gold, 'response': text}, ...]."""
    n = len(records)
    correct = strict_correct = failures = 0
    for r in records:
        if official_prediction(r["response"], pmc_llama) == r["answer"]:
            correct += 1
        s = strict_prediction(r["response"])
        if s is None:
            failures += 1
        elif s == r["answer"]:
            strict_correct += 1
    acc = correct / n if n else float("nan")
    return {
        "n": n,
        "acc_official": acc,
        # same binomial std as the official evaluator: sqrt(p(1-p)/n)
        "std": math.sqrt(acc * (1 - acc) / n) if n else float("nan"),
        "acc_strict": strict_correct / n if n else float("nan"),
        "parse_failures": failures,
    }
