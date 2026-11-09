"""Persian MedRAG runs (run_fa kernel): per-question correctness, paired tests, and RAG effect split
by whether the correct option's text is in the k retrieved snippets.

Writes only ids, gold letters, predictions and booleans (no question, corpus or model text, because
the Persian datasets state no licence).

Usage: python scripts/persian/analyze_fa_runs.py --runs <results dir of run_fa> \
           --fa_chunks data_local/fa_corpus_chunks.jsonl.gz --out_dir results/persian/run_fa
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from answer_coverage import GENERIC, norm  # noqa: E402
from medrag_repro.evaluate import official_prediction, strict_prediction  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

M = "Qwen__Qwen2.5-7B-Instruct"
K = 4


def load_preds(path):
    out = {}
    for r in map(json.loads, open(path)):
        p = official_prediction(r["response"])
        out[r["qid"]] = {"gold": r["answer"], "pred": p, "strict": strict_prediction(r["response"]),
                         "correct": p == r["answer"]}
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", required=True)
    a.add_argument("--fa_chunks", required=True)
    a.add_argument("--out_dir", required=True)
    a = a.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    R = os.path.join(ROOT, "results")
    bench = {**json.load(open(f"{ROOT}/data_local/fa_mmlu_benchmark.json")),
             **json.load(open(f"{ROOT}/data_local/kopp_benchmark.json"))}
    ret = {(r["dataset"], r["qid"]): r["rrf2"][:K]
           for r in map(json.loads, gzip.open(f"{R}/persian/fa_retrieval_top32.jsonl.gz", "rt"))}
    need = {i for v in ret.values() for i in v}
    text = {}
    for line in gzip.open(a.fa_chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            text[s["id"]] = norm(s["content"])

    def covered(ds, q):
        t = norm(bench[ds][q]["options"][bench[ds][q]["answer"]])
        if t.strip() in GENERIC or len(t.strip()) < 3:
            return None  # cannot be measured
        return any(t in text[i] for i in ret[(ds, q)])

    report = {}
    for ds in ["fa_mmlu", "kopp"]:
        cot = load_preds(f"{a.runs}/{M}/{ds}/cot.jsonl")
        rag = load_preds(f"{a.runs}/{M}/{ds}/rrf2_k4.jsonl")
        qids = sorted(cot)
        assert sorted(rag) == qids
        with gzip.open(f"{a.out_dir}/{ds}_predictions.jsonl.gz", "wt") as f:
            for q in qids:
                f.write(json.dumps({"qid": q, "gold": cot[q]["gold"], "cot": cot[q]["pred"],
                                    "rrf2_k4": rag[q]["pred"], "gold_in_top4": covered(ds, q)}) + "\n")
        c, r = [cot[q]["correct"] for q in qids], [rag[q]["correct"] for q in qids]
        d = {"n": len(qids), "cot_%": round(100 * sum(c) / len(c), 2), "rrf2_k4_%": round(100 * sum(r) / len(r), 2),
             "rag_fixes": sum(y and not x for x, y in zip(c, r)), "rag_breaks": sum(x and not y for x, y in zip(c, r)),
             "mcnemar_p": mcnemar(c, r), "by_gold_in_top4": {}}
        for label, val in [("present", True), ("absent", False), ("not_measurable", None)]:
            sel = [i for i, q in enumerate(qids) if covered(ds, q) is val]
            if sel:
                d["by_gold_in_top4"][label] = {
                    "n": len(sel), "cot_%": round(100 * sum(c[i] for i in sel) / len(sel), 1),
                    "rrf2_k4_%": round(100 * sum(r[i] for i in sel) / len(sel), 1),
                    "mcnemar_p": mcnemar([c[i] for i in sel], [r[i] for i in sel])}
        report[ds] = d

    # language gap: the same 1,089 MMLU-Med items, English (results/qwen_full) vs Persian
    en_dir = f"{R}/qwen_full/{M}/mmlu"
    fa_q = sorted(json.load(open(f"{ROOT}/data_local/fa_mmlu_benchmark.json"))["fa_mmlu"])
    gap = {}
    for conf in ["cot", "rrf2_k4"]:
        en = load_preds(f"{en_dir}/{conf}.jsonl")
        fa = load_preds(f"{a.runs}/{M}/fa_mmlu/{conf}.jsonl")
        assert set(fa) == set(fa_q) and set(fa) <= set(en)
        e, f_ = [en[q]["correct"] for q in fa_q], [fa[q]["correct"] for q in fa_q]
        gap[conf] = {"en_%": round(100 * sum(e) / len(e), 2), "fa_%": round(100 * sum(f_) / len(f_), 2),
                     "only_en_correct": sum(x and not y for x, y in zip(e, f_)),
                     "only_fa_correct": sum(y and not x for x, y in zip(e, f_)), "mcnemar_p": mcnemar(e, f_)}
    report["language_gap_mmlu"] = gap
    json.dump(report, open(f"{a.out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
