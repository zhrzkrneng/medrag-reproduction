"""2x2 RAG accuracy (question language x corpus, BGE-M3, k = 4): paired tests and the effect of
having the correct option's text in the 4 snippets (scored in the corpus language, as in
xling_coverage.py). Baselines: CoT on the same items (English: results/qwen_full; Persian: run_fa).

Writes ids, letters and booleans only.

Usage: python scripts/persian/analyze_xling_rag.py --runs <results dir of xling_rag> \
           --textbooks_dir <MedRAG textbooks/chunk> --fa_chunks data_local/fa_corpus_chunks.jsonl.gz \
           --fa_cot results/persian/run_fa/fa_mmlu_predictions.jsonl.gz --out_dir results/persian/xling_rag
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from answer_coverage import norm  # noqa: E402
from medrag_repro.evaluate import official_prediction  # noqa: E402
from medrag_repro.retrieval import load_corpus  # noqa: E402
from xling_coverage import mcnemar, present_gold  # noqa: E402

M = "Qwen__Qwen2.5-7B-Instruct"
K = 4


def preds(path):
    return {r["qid"]: official_prediction(r["response"]) for r in map(json.loads, open(path))}


def main():
    a = argparse.ArgumentParser()
    for x in ["runs", "textbooks_dir", "fa_chunks", "fa_cot", "out_dir"]:
        a.add_argument("--" + x, required=True)
    a = a.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    en = json.load(open(f"{ROOT}/data/mirage_benchmark.json"))["mmlu"]
    fa = json.load(open(f"{ROOT}/data_local/fa_mmlu_benchmark.json"))["fa_mmlu"]
    qids = sorted(fa)
    assert set(qids) <= set(en) and all(en[q]["answer"] == fa[q]["answer"] for q in qids)
    gold = {q: en[q]["answer"] for q in qids}
    ret = {}
    for r in map(json.loads, gzip.open(f"{ROOT}/results/persian/xling_rag_top32.jsonl.gz", "rt")):
        ret[(r["dataset"], "tb", r["qid"])] = r["bge_tb"][:K]
        ret[(r["dataset"], "fa", r["qid"])] = r["bge_fa"][:K]
    need = {i for v in ret.values() for i in v}
    text = {s["id"]: norm(s["content"]) for s in load_corpus(a.textbooks_dir) if s["id"] in need}
    for line in gzip.open(a.fa_chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            text[s["id"]] = norm(s["content"])

    cot = {"mmlu": preds(f"{ROOT}/results/qwen_full/{M}/mmlu/cot.jsonl"),
           "fa_mmlu": {r["qid"]: r["cot"] for r in map(json.loads, gzip.open(a.fa_cot, "rt"))}}
    cells = {("mmlu", "tb"): "1_en_q__textbooks", ("fa_mmlu", "tb"): "2_fa_q__textbooks",
             ("fa_mmlu", "fa"): "3_fa_q__fa_web", ("mmlu", "fa"): "4_en_q__fa_web"}
    corr, rep = {}, {"questions": len(qids), "k": K, "cells": {}, "paired_tests": {}}
    for (ds, c), name in cells.items():
        p = preds(f"{a.runs}/{M}/{ds}/bge_{c}_k4.jsonl")
        ok = [p[q] == gold[q] for q in qids]
        base = [cot[ds][q] == gold[q] for q in qids]
        opts = en if c == "tb" else fa  # coverage in the corpus language
        cov = [present_gold(opts[q], [text[i] for i in ret[(ds, c, q)]]) for q in qids]
        corr[name] = ok
        d = {"acc_%": round(100 * sum(ok) / len(ok), 2), "cot_same_language_%": round(100 * sum(base) / len(base), 2),
             "vs_cot_p": mcnemar(base, ok), "gold_in_top4": sum(cov)}
        for lab, v in [("present", True), ("absent", False)]:
            sel = [i for i in range(len(qids)) if cov[i] is v]
            d[f"{lab}: n, cot_%, rag_%"] = [len(sel), round(100 * sum(base[i] for i in sel) / len(sel), 1),
                                           round(100 * sum(ok[i] for i in sel) / len(sel), 1)]
        rep["cells"][name] = d
        with gzip.open(f"{a.out_dir}/{name}_predictions.jsonl.gz", "wt") as f:
            for q, x, y in zip(qids, ok, cov):
                f.write(json.dumps({"qid": q, "gold": gold[q], "pred": p[q], "correct": x, "gold_in_top4": y}) + "\n")
    for x, y in [("1_en_q__textbooks", "2_fa_q__textbooks"), ("4_en_q__fa_web", "3_fa_q__fa_web"),
                 ("1_en_q__textbooks", "4_en_q__fa_web"), ("2_fa_q__textbooks", "3_fa_q__fa_web")]:
        rep["paired_tests"][f"{x} vs {y}"] = mcnemar(corr[x], corr[y])
    json.dump(rep, open(f"{a.out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
