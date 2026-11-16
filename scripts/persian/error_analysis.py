"""Error analysis of the Persian runs (no GPU): language gap by MMLU subject, and the language the
model reasons in (Qwen2.5 often reasons in Chinese when the question is Persian).

The reasoning language is taken from the text before "answer_choice": Chinese if CJK characters are
more than 20% of the letters, else Persian or English by majority. Writes counts and accuracies only.

Usage: python scripts/persian/error_analysis.py --run_fa <run_fa results> --xling_rag <xling_rag results> \
           --out results/persian/error_analysis.json
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from medrag_repro.evaluate import official_prediction  # noqa: E402

M = "Qwen__Qwen2.5-7B-Instruct"


def reasoning_language(response):
    t = response.split('"answer_choice"')[0]
    zh = sum("一" <= c <= "鿿" for c in t)
    fa = sum("؀" <= c <= "ۿ" for c in t)
    la = sum(c.isascii() and c.isalpha() for c in t)
    return "zh" if zh > 0.2 * (zh + fa + la) else ("fa" if fa > la else "en")


def load(path):
    return {r["qid"]: r for r in map(json.loads, open(path))}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run_fa", required=True)
    a.add_argument("--xling_rag", required=True)
    a.add_argument("--out", required=True)
    a = a.parse_args()
    ok = lambda r: official_prediction(r["response"]) == r["answer"]  # noqa: E731

    en = load(f"{ROOT}/results/qwen_full/{M}/mmlu/cot.jsonl")
    fa = load(f"{a.run_fa}/{M}/fa_mmlu/cot.jsonl")
    sub = collections.defaultdict(lambda: [0, 0, 0])
    for q in fa:
        s = sub[q.rsplit("-", 1)[0]]
        s[0] += 1; s[1] += ok(en[q]); s[2] += ok(fa[q])  # noqa: E702
    report = {"cot_gap_by_subject": {k: {"n": n, "en_%": round(100 * e / n, 1), "fa_%": round(100 * f / n, 1),
                                         "gap": round(100 * (e - f) / n, 1)} for k, (n, e, f) in sorted(sub.items())}}

    files = {"fa_mmlu cot": f"{a.run_fa}/{M}/fa_mmlu/cot.jsonl",
             "fa_mmlu rrf2_k4 (Persian corpus)": f"{a.run_fa}/{M}/fa_mmlu/rrf2_k4.jsonl",
             "kopp cot": f"{a.run_fa}/{M}/kopp/cot.jsonl",
             "kopp rrf2_k4 (Persian corpus)": f"{a.run_fa}/{M}/kopp/rrf2_k4.jsonl",
             "fa_mmlu bge_k4 Textbooks": f"{a.xling_rag}/{M}/fa_mmlu/bge_tb_k4.jsonl",
             "fa_mmlu bge_k4 Persian corpus": f"{a.xling_rag}/{M}/fa_mmlu/bge_fa_k4.jsonl",
             "mmlu cot": f"{ROOT}/results/qwen_full/{M}/mmlu/cot.jsonl",
             "mmlu bge_k4 Textbooks": f"{a.xling_rag}/{M}/mmlu/bge_tb_k4.jsonl",
             "mmlu bge_k4 Persian corpus": f"{a.xling_rag}/{M}/mmlu/bge_fa_k4.jsonl"}
    report["reasoning_language"] = {}
    for name, path in files.items():
        g = collections.defaultdict(list)
        for r in load(path).values():
            g[reasoning_language(r["response"])].append(ok(r))
        report["reasoning_language"][name] = {k: {"n": len(v), "acc_%": round(100 * sum(v) / len(v), 1)}
                                              for k, v in sorted(g.items())}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
