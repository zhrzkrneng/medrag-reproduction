"""RQ3, "lost in the middle": at k = 32, does it matter WHERE the correct option's text first appears
among the retrieved snippets? English (MedQA, MMLU-Med, Textbooks) vs Persian (FA_MED_MMLU, web corpus).

For each question whose correct option's text occurs in the top 32 RRF-2 snippets (same lexical proxy
as answer_coverage.py, generic options excluded), the rank of the first such snippet is binned
(1, 2-4, 5-8, 9-16, 17-32). Per bin we report accuracy at k = 32 and, on the same questions, without
retrieval (CoT), because questions whose answer ranks first are also easier: RAG minus CoT is the
position effect with question difficulty held fixed (paired, exact McNemar). No GPU; counts only.

Usage: python scripts/persian/gold_position.py --en_chunk_dir <MedRAG textbooks/chunk> \
           --fa_chunks data_local/fa_corpus_chunks.jsonl.gz --out results/persian/gold_position.json
"""
import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from answer_coverage import GENERIC, norm  # noqa: E402
from medrag_repro.evaluate import official_prediction  # noqa: E402
from medrag_repro.retrieval import load_corpus  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

BINS = [(1, 1), (2, 4), (5, 8), (9, 16), (17, 32)]


def first_gold(item, ids, text):
    t = norm(item["options"][item["answer"]])
    if t.strip() in GENERIC or len(t.strip()) < 3:
        return None
    for r, i in enumerate(ids[:32], 1):
        if t in text[i]:
            return r
    return 0


def jl(path):
    op = gzip.open if path.endswith(".gz") else open
    return [json.loads(line) for line in op(path, "rt", encoding="utf-8") if line.strip()]


def summarise(bench, ret, text, rag, cot):
    out = {"n": len(bench), "gold_in_top32": 0, "bins": []}
    pos = {}
    for q, it in bench.items():
        r = first_gold(it, ret[q], text)
        if r:
            pos[q] = r
    out["gold_in_top32"] = len(pos)
    for lo, hi in BINS:
        qs = [q for q, r in pos.items() if lo <= r <= hi]
        a, b = [cot[q] for q in qs], [rag[q] for q in qs]
        out["bins"].append({"rank": f"{lo}" if lo == hi else f"{lo}-{hi}", "n": len(qs),
                            "acc_k32_%": round(100 * sum(b) / len(qs), 1) if qs else None,
                            "acc_cot_%": round(100 * sum(a) / len(qs), 1) if qs else None,
                            "rag_minus_cot": round(100 * (sum(b) - sum(a)) / len(qs), 1) if qs else None,
                            "p": mcnemar(a, b) if qs else None})
    absent = [q for q in bench if q not in pos]
    a, b = [cot[q] for q in absent], [rag[q] for q in absent]
    out["absent_or_generic"] = {"n": len(absent), "acc_k32_%": round(100 * sum(b) / len(absent), 1),
                                "acc_cot_%": round(100 * sum(a) / len(absent), 1),
                                "rag_minus_cot": round(100 * (sum(b) - sum(a)) / len(absent), 1), "p": mcnemar(a, b)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--en_chunk_dir", required=True)
    p.add_argument("--fa_chunks", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rep = {"method": __doc__.split("Usage")[0].strip()}

    en_ret = {(r["dataset"], r["qid"]): r["rrf2"] for r in jl(os.path.join(ROOT, "results", "retrieval", "textbooks_top32.jsonl.gz"))}
    need = {i for v in en_ret.values() for i in v}
    en_text = {s["id"]: norm(s["title"] + " " + s["content"]) for s in load_corpus(a.en_chunk_dir) if s["id"] in need}
    en_bench = json.load(open(os.path.join(ROOT, "data", "mirage_benchmark.json")))
    base = os.path.join(ROOT, "results", "qwen_full", "Qwen__Qwen2.5-7B-Instruct")
    for ds in ("medqa", "mmlu"):
        def load(conf):
            return {r["qid"]: official_prediction(r["response"]) == r["answer"] for r in jl(os.path.join(base, ds, f"{conf}.jsonl"))}
        bench = en_bench[ds]
        rep[f"en_{ds}"] = summarise(bench, {q: en_ret[(ds, q)] for q in bench}, en_text, load("rrf2_k32"), load("cot"))

    fa_ret = {r["qid"]: r["rrf2"] for r in jl(os.path.join(ROOT, "results", "persian", "fa_retrieval_top32.jsonl.gz"))
              if r["dataset"] == "fa_mmlu"}
    need = {i for v in fa_ret.values() for i in v}
    fa_text = {}
    for line in gzip.open(a.fa_chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            fa_text[s["id"]] = norm(s["content"])
    fa_bench = json.load(open(os.path.join(ROOT, "data_local", "fa_mmlu_benchmark.json")))["fa_mmlu"]
    k32 = {r["qid"]: r["rrf2_k32"] == r["gold"] for r in jl(os.path.join(ROOT, "results", "persian", "run_fa32", "fa_mmlu_predictions.jsonl.gz"))}
    cot = {r["qid"]: r["cot"] == r["gold"] for r in jl(os.path.join(ROOT, "results", "persian", "run_fa", "fa_mmlu_predictions.jsonl.gz"))}
    rep["fa_mmlu"] = summarise(fa_bench, fa_ret, fa_text, k32, cot)
    json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)
    for k, v in rep.items():
        if k != "method":
            print(k, v["n"], "gold in top 32:", v["gold_in_top32"])
            for b in v["bins"]:
                print("  ", b)
            print("   absent", v["absent_or_generic"])


if __name__ == "__main__":
    main()
