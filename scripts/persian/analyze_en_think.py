"""The "think in English" ablation (kaggle/en_think): the Persian FA_MED_MMLU questions with one extra
sentence in the system prompt asking the model to write its step-by-step thinking in English.
Compared, question by question, with the same configurations without the sentence (run_fa CoT and
xling_rag Textbooks@4), with the translate-first runs and with the English original.

Writes ids, letters, booleans and the detected reasoning language only (no model text: the
responses discuss FA_MED_MMLU questions, which state no licence).

Usage: python scripts/persian/analyze_en_think.py --run <kaggle output dir> --out_dir results/persian/en_think
"""
import argparse
import collections
import gzip
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from error_analysis import reasoning_language  # noqa: E402
from medrag_repro.evaluate import official_prediction, strict_prediction  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

M = "Qwen__Qwen2.5-7B-Instruct"
P = f"{ROOT}/results/persian"


def gz(path):
    return {r["qid"]: r for r in map(json.loads, gzip.open(path, "rt"))}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", required=True)
    a.add_argument("--out_dir", required=True)
    a = a.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    new = {c: {r["qid"]: r for r in map(json.loads, open(f"{a.run}/results/{M}/fa_mmlu/{c}_en_think.jsonl"))}
           for c in ["cot", "bge_tb_k4"]}
    qids = sorted(new["cot"])
    assert sorted(new["bge_tb_k4"]) == qids
    run_fa, tb = gz(f"{P}/run_fa/fa_mmlu_predictions.jsonl.gz"), gz(f"{P}/xling_rag/2_fa_q__textbooks_predictions.jsonl.gz")
    tr, en_tb = gz(f"{P}/translate/predictions.jsonl.gz"), gz(f"{P}/xling_rag/1_en_q__textbooks_predictions.jsonl.gz")
    assert sorted(run_fa) == qids == sorted(tb) == sorted(tr)
    gold = {q: run_fa[q]["gold"] for q in qids}
    assert all(new["cot"][q]["answer"] == gold[q] for q in qids)
    en_cot = {r["qid"]: official_prediction(r["response"]) == r["answer"]
              for r in map(json.loads, open(f"{ROOT}/results/qwen_full/{M}/mmlu/cot.jsonl"))}

    pred = {c: {q: official_prediction(new[c][q]["response"]) for q in qids} for c in new}
    lang = {c: {q: reasoning_language(new[c][q]["response"]) for q in qids} for c in new}
    runs = {"fa_cot": [run_fa[q]["cot"] == gold[q] for q in qids],
            "fa_cot_en_think": [pred["cot"][q] == gold[q] for q in qids],
            "fa_textbooks_k4": [tb[q]["correct"] for q in qids],
            "fa_textbooks_k4_en_think": [pred["bge_tb_k4"][q] == gold[q] for q in qids],
            "mt_cot": [tr[q]["mt_cot_correct"] for q in qids],
            "mt_textbooks_k4": [tr[q]["mt_textbooks_k4_correct"] for q in qids],
            "en_cot": [en_cot[q] for q in qids],
            "en_textbooks_k4": [en_tb[q]["correct"] for q in qids]}
    rep = {"n": len(qids), "system_suffix": " Write your step-by-step thinking in English, even if the question is in another language.",
           "accuracy_%": {k: round(100 * sum(v) / len(v), 2) for k, v in runs.items()},
           "unparsed_official": {c: sum(pred[c][q] is None for q in qids) for c in new},
           "unparsed_strict": {c: sum(strict_prediction(new[c][q]["response"]) is None for q in qids) for c in new},
           "paired_tests": {}, "reasoning_language": {}}
    for x, y in [("fa_cot", "fa_cot_en_think"), ("fa_textbooks_k4", "fa_textbooks_k4_en_think"),
                 ("fa_cot_en_think", "mt_cot"), ("fa_textbooks_k4_en_think", "mt_textbooks_k4"),
                 ("fa_cot_en_think", "fa_textbooks_k4_en_think"), ("fa_cot_en_think", "en_cot")]:
        rep["paired_tests"][f"{x} vs {y}"] = mcnemar(runs[x], runs[y])
    key = {"cot": "fa_cot_en_think", "bge_tb_k4": "fa_textbooks_k4_en_think"}
    for c in new:
        g = collections.defaultdict(list)
        for i, q in enumerate(qids):
            g[lang[c][q]].append(runs[key[c]][i])
        rep["reasoning_language"][key[c]] = {k: {"n": len(v), "acc_%": round(100 * sum(v) / len(v), 1)} for k, v in sorted(g.items())}
    # does retrieval help when the correct option's text is in the 4 Textbooks snippets? (same split as xling_rag)
    for lab, v in [("present", True), ("absent", False)]:
        sel = [i for i, q in enumerate(qids) if tb[q]["gold_in_top4"] is v]
        rep[f"textbooks_k4_gold_{lab}"] = {"n": len(sel), **{k: round(100 * sum(runs[k][i] for i in sel) / len(sel), 1) for k in
                                                           ["fa_cot", "fa_cot_en_think", "fa_textbooks_k4", "fa_textbooks_k4_en_think"]}}
    with gzip.open(f"{a.out_dir}/predictions.jsonl.gz", "wt") as f:
        for q in qids:
            f.write(json.dumps({"qid": q, "gold": gold[q], "cot_en_think": pred["cot"][q], "cot_lang": lang["cot"][q],
                                "textbooks_k4_en_think": pred["bge_tb_k4"][q], "textbooks_lang": lang["bge_tb_k4"][q]}) + "\n")
    json.dump(rep, open(f"{a.out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
