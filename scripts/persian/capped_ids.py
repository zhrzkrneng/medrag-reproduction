"""Which answers hit the answer-length cap, per run (ids and booleans only, so the result can be committed).

An answer hits the cap when output_tokens >= min(1024, 8192 - prompt_tokens) (MedRAG's budget as used in
generate.py). Persian needs about twice the tokens of English for the same text, so the cap cuts more
Persian answers; stats_study.py uses this file to repeat the key comparisons on uncapped answers only.

Needs the raw kernel outputs (local, not committed: they contain model responses to dataset questions).
Usage: python scripts/persian/capped_ids.py --scratch <dir with out_* kernel outputs>
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
Q, L = "Qwen__Qwen2.5-7B-Instruct", "meta-llama__Llama-3.1-8B-Instruct"

# run name -> path relative to --scratch (or to the repo for committed English runs)
RUNS = {
    "qwen/mmlu/cot": ("repo", f"results/qwen_full/{Q}/mmlu/cot.jsonl"),
    "qwen/mmlu/rrf2_k4": ("repo", f"results/qwen_full/{Q}/mmlu/rrf2_k4.jsonl"),
    "qwen/mmlu/rrf2_k32": ("repo", f"results/qwen_full/{Q}/mmlu/rrf2_k32.jsonl"),
    "qwen/mmlu/bge_tb_k4": ("scratch", f"out_xling_rag/results/{Q}/mmlu/bge_tb_k4.jsonl"),
    "qwen/fa_mmlu/bge_tb_k4": ("scratch", f"out_xling_rag/results/{Q}/fa_mmlu/bge_tb_k4.jsonl"),
    "qwen/fa_mmlu/bge_m3_k4": ("scratch", f"out_xling_rag/results/{Q}/fa_mmlu/bge_fa_k4.jsonl"),
    "qwen/fa_mmlu/cot": ("scratch", f"out_run_fa/results/{Q}/fa_mmlu/cot.jsonl"),
    "qwen/fa_mmlu/rrf2_k4": ("scratch", f"out_run_fa/results/{Q}/fa_mmlu/rrf2_k4.jsonl"),
    "qwen/kopp/cot": ("scratch", f"out_run_fa/results/{Q}/kopp/cot.jsonl"),
    "qwen/kopp/rrf2_k4": ("scratch", f"out_run_fa/results/{Q}/kopp/rrf2_k4.jsonl"),
    "qwen/fa_mmlu/rrf2_k32": ("scratch", f"out_fa32b/results/{Q}/fa_mmlu/rrf2_k32.jsonl"),
    "qwen/fa_mmlu_mt/cot": ("scratch", f"out_translate/results/{Q}/fa_mmlu_mt/cot.jsonl"),
    "qwen/fa_mmlu_mt/bge_tb_k4": ("scratch", f"out_translate/results/{Q}/fa_mmlu_mt/bge_tb_k4.jsonl"),
    "llama/mmlu/cot": ("scratch", f"out_model2/results/{L}/mmlu/cot.jsonl"),
    "llama/mmlu/rrf2_k4": ("scratch", f"out_model2en/results/{L}/mmlu/rrf2_k4.jsonl"),
    "llama/fa_mmlu/cot": ("scratch", f"out_model2/results/{L}/fa_mmlu/cot.jsonl"),
    "llama/fa_mmlu/rrf2_k4": ("scratch", f"out_model2/results/{L}/fa_mmlu/rrf2_k4.jsonl"),
    "llama/kopp/cot": ("scratch", f"out_model2/results/{L}/kopp/cot.jsonl"),
    "llama/kopp/rrf2_k4": ("scratch", f"out_model2/results/{L}/kopp/rrf2_k4.jsonl"),
}
for ds, confs in (("fa_mmlu", ["bm25_k4", "rrf2_k1", "rrf2_k2", "rrf2_k8", "rrf2_k16"]), ("kopp", ["bm25_k4", "bge_m3_k4"])):
    for c in confs:
        RUNS[f"qwen/{ds}/{c}"] = ("scratch", f"out_rq3/results/{Q}/{ds}/{c}.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", required=True)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "persian", "capped_ids.json"))
    a = p.parse_args()
    out = {}
    for name, (base, rel) in RUNS.items():
        path = os.path.join(ROOT if base == "repo" else a.scratch, rel)
        if not os.path.exists(path):
            print("missing", name, path)
            continue
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        capped = sorted(r["qid"] for r in rows if r["output_tokens"] >= min(1024, 8192 - r["prompt_tokens"]))
        out[name] = {"n": len(rows), "n_capped": len(capped), "capped_qids": capped}
        print(f"{name:28s} n={len(rows):5d} capped={len(capped)}")
    json.dump({"rule": "output_tokens >= min(1024, 8192 - prompt_tokens)", "runs": out}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
