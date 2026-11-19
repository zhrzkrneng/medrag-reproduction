"""Second model family (kaggle/model2): Llama-3.1-8B-Instruct with MedRAG's prompts, compared with
Qwen2.5-7B-Instruct on the same questions.

  * English MMLU-Med CoT vs Persian FA_MED_MMLU CoT (language gap, same 1,089 items)
  * Persian CoT vs RRF-2@4 on FA_MED_MMLU and KOPP (effect of retrieval)
  * Llama vs Qwen in every cell
Official MIRAGE parser (with its fallback "A") and the strict parser; exact McNemar; Holm and BH within
this family. Writes ids / letters / booleans only.

Usage: python scripts/persian/analyze_model2.py --run <model2 output dir> [<model2_en output dir>] --out_dir results/persian/model2
"""
import argparse
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from medrag_repro.evaluate import official_prediction, strict_prediction  # noqa: E402
from stats_all import bh, holm  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

SLUG = "meta-llama__Llama-3.1-8B-Instruct"


def jl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, nargs="+", help="kernel output dirs (model2, model2_en)")
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "persian", "model2"))
    a = p.parse_args()
    res_dirs = [os.path.join(r, "results") for r in a.run]
    llama, info = {}, {}
    for ds, conf in [("mmlu", "cot"), ("mmlu", "rrf2_k4"), ("fa_mmlu", "cot"), ("fa_mmlu", "rrf2_k4"), ("kopp", "cot"), ("kopp", "rrf2_k4")]:
        found = [os.path.join(r, SLUG, ds, f"{conf}.jsonl") for r in res_dirs if os.path.exists(os.path.join(r, SLUG, ds, f"{conf}.jsonl"))]
        path = found[0] if found else os.path.join(res_dirs[0], SLUG, ds, f"{conf}.jsonl")
        if not os.path.exists(path):
            print("missing", path)
            continue
        rows = jl(path)
        d = {}
        unp = capped = fa_reason = looping = 0
        for r in rows:
            t = r["response"]
            nfa, nen, nzh = (len(re.findall(x, t)) for x in ("[\u0600-\u06FF]", "[A-Za-z]", "[\u4e00-\u9fff]"))
            fa_reason += nfa / (nfa + nen + nzh + 1) > 0.5
            off = official_prediction(r["response"])
            st = strict_prediction(r["response"])
            d[r["qid"]] = {"gold": r["answer"], "pred": off, "strict": st}
            unp += st is None
            budget = min(1024, 8192 - r["prompt_tokens"])
            hit = r["output_tokens"] >= budget
            capped += hit
            looping += hit and len(set(t[-400:].split())) < 30  # the last 400 characters use < 30 distinct words
        llama[(ds, conf)] = d
        info[f"{ds}/{conf}"] = {"n": len(d), "acc_%": round(100 * sum(v["pred"] == v["gold"] for v in d.values()) / len(d), 2),
                                "strict_acc_%": round(100 * sum(v["strict"] == v["gold"] for v in d.values()) / len(d), 2),
                                "strict_unparsed": unp, "answer_hit_length_cap": capped,
                                "capped_and_repeating": looping, "answers_mostly_in_persian": fa_reason}
    # Qwen on the same questions
    qwen = {}
    base = os.path.join(ROOT, "results", "qwen_full", "Qwen__Qwen2.5-7B-Instruct", "mmlu", "cot.jsonl")
    qwen[("mmlu", "cot")] = {r["qid"]: official_prediction(r["response"]) == r["answer"] for r in jl(base)}
    qwen[("mmlu", "rrf2_k4")] = {r["qid"]: official_prediction(r["response"]) == r["answer"]
                                 for r in jl(base.replace("cot.jsonl", "rrf2_k4.jsonl"))}
    for ds in ("fa_mmlu", "kopp"):
        rows = jl(os.path.join(ROOT, "results", "persian", "run_fa", f"{ds}_predictions.jsonl.gz"))
        qwen[(ds, "cot")] = {r["qid"]: r["cot"] == r["gold"] for r in rows}
        qwen[(ds, "rrf2_k4")] = {r["qid"]: r["rrf2_k4"] == r["gold"] for r in rows}
    ok = {k: {q: v["pred"] == v["gold"] for q, v in d.items()} for k, d in llama.items()}

    tests = []

    def test(name, A, B):
        qs = sorted(set(A) & set(B))
        x, y = [A[q] for q in qs], [B[q] for q in qs]
        tests.append({"test": name, "n": len(qs), "acc_a": round(100 * sum(x) / len(qs), 2),
                      "acc_b": round(100 * sum(y) / len(qs), 2), "diff_b_minus_a": round(100 * (sum(y) - sum(x)) / len(qs), 2),
                      "p_raw": mcnemar(x, y)})

    if ("mmlu", "cot") in ok and ("fa_mmlu", "cot") in ok:
        test("Llama language gap: EN CoT -> FA CoT (FA_MED_MMLU items)", ok[("mmlu", "cot")], ok[("fa_mmlu", "cot")])
    for ds in ("mmlu", "fa_mmlu", "kopp"):
        if (ds, "cot") in ok and (ds, "rrf2_k4") in ok:
            test(f"Llama RAG effect {ds}: CoT -> RRF-2@4", ok[(ds, "cot")], ok[(ds, "rrf2_k4")])
    for k in ok:
        test(f"Qwen -> Llama, {k[0]}/{k[1]}", qwen[k], ok[k])
    robust = {}
    for ds in ("fa_mmlu", "kopp"):
        if (ds, "cot") in llama and (ds, "rrf2_k4") in llama:
            A, B = llama[(ds, "cot")], llama[(ds, "rrf2_k4")]
            qs = [q for q in A if A[q]["strict"] and q in B and B[q]["strict"]]
            x = [A[q]["strict"] == A[q]["gold"] for q in qs]
            y = [B[q]["strict"] == B[q]["gold"] for q in qs]
            robust[ds] = {"n_both_strictly_parsed": len(qs), "cot_%": round(100 * sum(x) / len(qs), 2),
                          "rrf2_k4_%": round(100 * sum(y) / len(qs), 2), "p": mcnemar(x, y)}
    ps = [t["p_raw"] for t in tests]
    for t, h, q in zip(tests, holm(ps), bh(ps)):
        t["p_holm"], t["q_bh"] = round(h, 4), round(q, 4)
    src = os.path.join(res_dirs[0], "model_source.json")
    out = {"model": "meta-llama/Llama-3.1-8B-Instruct", "weights": json.load(open(src)) if os.path.exists(src) else None,
           "prompt": "MedRAG (English instructions, JSON CoT); Persian question, options and snippets",
           "accuracy": info, "tests": tests, "rag_effect_on_strictly_parsed_both": robust,
           "note": "official MIRAGE parser (falls back to 'A'); strict parser counts unparsable answers as wrong"}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    with gzip.open(os.path.join(a.out_dir, "predictions.jsonl.gz"), "wt") as f:
        for (ds, conf), d in sorted(llama.items()):
            for q, v in sorted(d.items()):
                f.write(json.dumps({"dataset": ds, "config": conf, "qid": q, "gold": v["gold"], "pred": v["pred"],
                                    "strict": v["strict"]}) + "\n")
    print(json.dumps(info, indent=1))
    for t in tests:
        print(t)


if __name__ == "__main__":
    main()
