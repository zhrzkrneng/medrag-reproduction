"""RQ2: RAG vs a Persian medical fine-tuned model (kaggle/rq2). Gaokerena-V (LoRA on Aya-Expanse-8B) and its base
Aya-Expanse-8B, each with CoT and RRF-2@4 over the Persian corpus, on FA_MED_MMLU and KOPP; Qwen2.5-7B-Instruct
(run_fa) on the same questions as a reference. Paired exact McNemar tests on the same questions.

  fine-tuning effect: gaokerena cot vs aya cot
  RAG effect:         aya rrf2_k4 vs aya cot (and for gaokerena, and qwen)
  RAG vs fine-tuning: aya rrf2_k4 vs gaokerena cot
  both:               gaokerena rrf2_k4 vs aya cot

Writes ids, letters and booleans only (the Persian data state no licence).

Usage: python scripts/persian/analyze_rq2.py --run <kaggle output dir> --out_dir results/persian/rq2
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from medrag_repro.evaluate import official_prediction, strict_prediction  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

MODELS = {"gaokerena": "gaokerena__gaokerena-v1.0", "aya": "CohereLabs__aya-expanse-8b"}
CONFIGS = ["cot", "rrf2_k4"]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run", required=True)
    a.add_argument("--out_dir", required=True)
    a = a.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rep = {"model_revisions": json.load(open(f"{a.run}/results/model_revisions.json"))}
    for ds in ["fa_mmlu", "kopp"]:
        base = {r["qid"]: r for r in map(json.loads, gzip.open(f"{ROOT}/results/persian/run_fa/{ds}_predictions.jsonl.gz", "rt"))}
        rows = {(m, c): {r["qid"]: r for r in map(json.loads, open(f"{a.run}/results/{slug}/{ds}/{c}.jsonl"))}
                for m, slug in MODELS.items() for c in CONFIGS}
        qids = sorted(set.intersection(*[set(v) for v in rows.values()]))
        gold = {q: base[q]["gold"] for q in qids}
        assert all(rows[k][q]["answer"] == gold[q] for k in rows for q in qids)
        pred = {k: {q: official_prediction(rows[k][q]["response"]) for q in qids} for k in rows}
        runs = {f"{m}_{c}": [pred[(m, c)][q] == gold[q] for q in qids] for m, c in rows}
        runs.update({f"qwen_{c}": [base[q][c] == gold[q] for q in qids] for c in CONFIGS})

        def capped(r):
            return r["output_tokens"] >= min(1024, 8192 - r["prompt_tokens"])
        d = {"n": len(qids), "of": len(base),
             "accuracy_%": {k: round(100 * sum(v) / len(v), 2) for k, v in runs.items()},
             "unparsed_official": {f"{m}_{c}": sum(pred[(m, c)][q] is None for q in qids) for m, c in rows},
             "unparsed_strict": {f"{m}_{c}": sum(strict_prediction(rows[(m, c)][q]["response"]) is None for q in qids) for m, c in rows},
             "answer_hit_length_cap": {f"{m}_{c}": sum(capped(rows[(m, c)][q]) for q in qids) for m, c in rows},
             "paired_tests": {}}
        for x, y in [("aya_cot", "gaokerena_cot"), ("aya_cot", "aya_rrf2_k4"), ("gaokerena_cot", "gaokerena_rrf2_k4"),
                     ("gaokerena_cot", "aya_rrf2_k4"), ("aya_cot", "gaokerena_rrf2_k4"), ("qwen_cot", "gaokerena_cot"),
                     ("qwen_rrf2_k4", "gaokerena_rrf2_k4"), ("qwen_cot", "aya_cot")]:
            d["paired_tests"][f"{x} vs {y}"] = {"diff_points": round(d["accuracy_%"][y] - d["accuracy_%"][x], 2),
                                                "p": mcnemar(runs[x], runs[y])}
        d["by_gold_in_top4"] = {}
        for lab, v in [("present", True), ("absent", False)]:
            sel = [i for i, q in enumerate(qids) if base[q].get("gold_in_top4") is v]
            if sel:
                d["by_gold_in_top4"][lab] = {"n": len(sel), **{k: round(100 * sum(runs[k][i] for i in sel) / len(sel), 1) for k in runs}}
        rep[ds] = d
        with gzip.open(f"{a.out_dir}/{ds}_predictions.jsonl.gz", "wt") as f:
            for q in qids:
                f.write(json.dumps({"qid": q, "gold": gold[q], **{f"{m}_{c}": pred[(m, c)][q] for m, c in rows}}) + "\n")
    json.dump(rep, open(f"{a.out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
