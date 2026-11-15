"""Persian MedRAG at k = 32 (kaggle/run_fa32) on FA_MED_MMLU: accuracy, paired tests against Persian CoT
and k = 4 (run_fa) and against the English RRF-2 runs on the same items (results/qwen_full), and the
RAG effect split by whether the correct option's text is in the 32 snippets.

Writes ids, letters and booleans only (the Persian data state no licence).

Usage: python scripts/persian/analyze_fa32.py --run <kaggle output dir> \
           --fa_chunks data_local/fa_corpus_chunks.jsonl.gz --out_dir results/persian/run_fa32
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


def main():
    a = argparse.ArgumentParser()
    for x in ["run", "fa_chunks", "out_dir"]:
        a.add_argument("--" + x, required=True)
    a = a.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = {r["qid"]: r for r in map(json.loads, open(f"{a.run}/results/{M}/fa_mmlu/rrf2_k32.jsonl"))}
    fa = json.load(open(f"{ROOT}/data_local/fa_mmlu_benchmark.json"))["fa_mmlu"]
    qids = sorted(fa)
    done = [q for q in qids if q in rows]  # all of them unless the time budget stopped the run
    base = {r["qid"]: r for r in map(json.loads, gzip.open(f"{ROOT}/results/persian/run_fa/fa_mmlu_predictions.jsonl.gz", "rt"))}
    gold = {q: base[q]["gold"] for q in qids}
    assert all(rows[q]["answer"] == gold[q] for q in done)
    en = {}
    for c in ["cot", "rrf2_k4", "rrf2_k32"]:
        en[c] = {r["qid"]: official_prediction(r["response"]) == r["answer"]
                 for r in map(json.loads, open(f"{ROOT}/results/qwen_full/{M}/mmlu/{c}.jsonl"))}
    pred = {q: official_prediction(rows[q]["response"]) for q in done}
    runs = {"fa_cot": [base[q]["cot"] == gold[q] for q in done],
            "fa_rrf2_k4": [base[q]["rrf2_k4"] == gold[q] for q in done],
            "fa_rrf2_k32": [pred[q] == gold[q] for q in done],
            "en_cot": [en["cot"][q] for q in done], "en_rrf2_k4": [en["rrf2_k4"][q] for q in done],
            "en_rrf2_k32": [en["rrf2_k32"][q] for q in done]}

    ret = {r["qid"]: r["rrf2"][:32] for r in map(json.loads, gzip.open(f"{ROOT}/results/persian/fa_retrieval_top32.jsonl.gz", "rt"))
           if r["dataset"] == "fa_mmlu"}
    need = {i for q in done for i in ret[q]}
    text = {}
    for line in gzip.open(a.fa_chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            text[s["id"]] = norm(s["content"])

    def covered(q):
        t = norm(fa[q]["options"][gold[q]])
        if t.strip() in GENERIC or len(t.strip()) < 3:
            return None
        return any(t in text[i] for i in ret[q])

    cov = [covered(q) for q in done]
    rep = {"questions": len(qids), "answered": len(done),
           "unparsed_official": sum(pred[q] is None for q in done),
           "unparsed_strict": sum(strict_prediction(rows[q]["response"]) is None for q in done),
           "prompt_tokens_mean": round(sum(rows[q]["prompt_tokens"] for q in done) / len(done)),
           "accuracy_%": {k: round(100 * sum(v) / len(v), 2) for k, v in runs.items()}, "paired_tests": {}}
    for x, y in [("fa_cot", "fa_rrf2_k32"), ("fa_rrf2_k4", "fa_rrf2_k32"), ("fa_rrf2_k32", "en_rrf2_k32"),
                 ("en_cot", "en_rrf2_k32")]:
        rep["paired_tests"][f"{x} vs {y}"] = mcnemar(runs[x], runs[y])
    rep["language_gap_points"] = {"cot": round(rep["accuracy_%"]["en_cot"] - rep["accuracy_%"]["fa_cot"], 2),
                                  "rrf2_k4": round(rep["accuracy_%"]["en_rrf2_k4"] - rep["accuracy_%"]["fa_rrf2_k4"], 2),
                                  "rrf2_k32": round(rep["accuracy_%"]["en_rrf2_k32"] - rep["accuracy_%"]["fa_rrf2_k32"], 2)}
    rep["by_gold_in_top32"] = {}
    for lab, v in [("present", True), ("absent", False), ("not_measurable", None)]:
        sel = [i for i in range(len(done)) if cov[i] is v]
        if sel:
            rep["by_gold_in_top32"][lab] = {"n": len(sel), **{k: round(100 * sum(runs[k][i] for i in sel) / len(sel), 1)
                                                              for k in ["fa_cot", "fa_rrf2_k4", "fa_rrf2_k32"]},
                                            "cot_vs_k32_p": mcnemar([runs["fa_cot"][i] for i in sel], [runs["fa_rrf2_k32"][i] for i in sel])}
    # MedRAG's length budget: context cut at 7,168 tokens and prompt + answer <= 8,192, so a long question leaves
    # little room for the answer; Persian needs more tokens than English, so more answers hit that cap.
    def capped(r):
        return r["output_tokens"] >= min(1024, 8192 - r["prompt_tokens"])
    en_rows = {r["qid"]: r for r in map(json.loads, open(f"{ROOT}/results/qwen_full/{M}/mmlu/rrf2_k32.jsonl"))}
    cap = [capped(rows[q]) for q in done]
    rep["answer_hit_length_cap"] = {"fa_rrf2_k32": sum(cap), "en_rrf2_k32": sum(capped(en_rows[q]) for q in done),
                                    "fa_context_trimmed_further": sum(bool(rows[q].get("context_trimmed_tokens")) for q in done)}
    rep["by_answer_capped"] = {}
    for lab, v in [("capped", True), ("not_capped", False)]:
        sel = [i for i in range(len(done)) if cap[i] is v]
        rep["by_answer_capped"][lab] = {"n": len(sel), **{k: round(100 * sum(runs[k][i] for i in sel) / len(sel), 1)
                                                         for k in ["fa_cot", "fa_rrf2_k4", "fa_rrf2_k32"]},
                                        "unparsed_strict_k32": sum(strict_prediction(rows[done[i]]["response"]) is None for i in sel),
                                        "cot_vs_k32_p": mcnemar([runs["fa_cot"][i] for i in sel], [runs["fa_rrf2_k32"][i] for i in sel])}
    with gzip.open(f"{a.out_dir}/fa_mmlu_predictions.jsonl.gz", "wt") as f:
        for q, c in zip(done, cov):
            f.write(json.dumps({"qid": q, "gold": gold[q], "rrf2_k32": pred[q], "gold_in_top32": c,
                                "answer_capped": capped(rows[q])}) + "\n")
    json.dump(rep, open(f"{a.out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
