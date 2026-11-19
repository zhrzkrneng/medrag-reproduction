"""Checks requested by the self-review (kaggle/review), Qwen2.5-7B-Instruct:

  1. translate first on KOPP (no English original): kopp_mt CoT vs Persian CoT on KOPP
  2. option rotation on English MMLU-Med: mmlu_rot CoT vs English CoT (recall of MMLU items?)
  3. oracle@4 (snippets containing the correct option's text first) vs CoT and RRF-2@4 on the same questions, Persian
     (FA_MED_MMLU, KOPP; Persian corpus) and English (MMLU-Med; Textbooks), and the oracle effect FA vs EN on the
     items in both subsets
  4. MedRAG's prompts in Persian: CoT and RRF-2@4 on FA_MED_MMLU vs the English-instruction prompt
Official MIRAGE parser (fallback 'A') and the strict parser; exact McNemar; Holm / BH within this family (they also
enter the whole-study family in stats_study.py). Writes ids / letters / counts only.
Usage: python scripts/persian/analyze_review.py --run <kernel output dir>
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
from stats_study import sign_test  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

Q = "Qwen__Qwen2.5-7B-Instruct"
P = os.path.join(ROOT, "results", "persian")


def jl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out_dir", default=os.path.join(P, "review"))
    a = p.parse_args()
    R = os.path.join(a.run, "results", Q)
    new, info = {}, {}
    for ds, conf in [("kopp_mt", "cot"), ("mmlu_rot", "cot"), ("fa_mmlu", "oracle_k4"), ("kopp", "oracle_k4"),
                     ("mmlu", "oracle_k4"), ("fa_mmlu", "cot_fa_prompt"), ("fa_mmlu", "rrf2_k4_fa_prompt")]:
        rows = jl(os.path.join(R, ds, f"{conf}.jsonl"))
        d = {}
        capped = fa_reason = zh_reason = 0
        for r in rows:
            t = r["response"]
            nfa, nen, nzh = (len(re.findall(x, t)) for x in ("[؀-ۿ]", "[A-Za-z]", "[一-鿿]"))
            fa_reason += nfa / (nfa + nen + nzh + 1) > 0.5
            zh_reason += nzh / (nfa + nen + nzh + 1) > 0.5
            capped += r["output_tokens"] >= min(1024, 8192 - r["prompt_tokens"])
            d[r["qid"]] = {"gold": r["answer"], "pred": official_prediction(t), "strict": strict_prediction(t)}
        new[(ds, conf)] = d
        info[f"{ds}/{conf}"] = {"n": len(d), "acc_%": round(100 * sum(v["pred"] == v["gold"] for v in d.values()) / len(d), 2),
                                "strict_acc_%": round(100 * sum(v["strict"] == v["gold"] for v in d.values()) / len(d), 2),
                                "strict_unparsed": sum(v["strict"] is None for v in d.values()), "hit_length_cap": capped,
                                "answers_mostly_persian": fa_reason, "answers_mostly_chinese": zh_reason}
    ok = {k: {q: v["pred"] == v["gold"] for q, v in d.items()} for k, d in new.items()}
    st = {k: {q: v["strict"] == v["gold"] for q, v in d.items()} for k, d in new.items()}
    # earlier runs
    fa_runs = {ds: jl(os.path.join(P, "run_fa", f"{ds}_predictions.jsonl.gz")) for ds in ("fa_mmlu", "kopp")}
    old = {}
    for ds, rows in fa_runs.items():
        old[(ds, "cot")] = {r["qid"]: r["cot"] == r["gold"] for r in rows}
        old[(ds, "rrf2_k4")] = {r["qid"]: r["rrf2_k4"] == r["gold"] for r in rows}
    base = os.path.join(ROOT, "results", "qwen_full", Q, "mmlu")
    for conf in ("cot", "rrf2_k4"):
        old[("mmlu", conf)] = {r["qid"]: official_prediction(r["response"]) == r["answer"] for r in jl(os.path.join(base, f"{conf}.jsonl"))}
    en_strict = {r["qid"]: strict_prediction(r["response"]) == r["answer"] for r in jl(os.path.join(base, "cot.jsonl"))}

    tests = []

    def test(name, A, B, extra=None):
        qs = sorted(set(A) & set(B))
        x, y = [A[q] for q in qs], [B[q] for q in qs]
        tests.append({"test": name, "n": len(qs), "acc_a": round(100 * sum(x) / len(qs), 2), "acc_b": round(100 * sum(y) / len(qs), 2),
                      "diff_b_minus_a": round(100 * (sum(y) - sum(x)) / len(qs), 2), "p_raw": mcnemar(x, y), **(extra or {})})

    test("KOPP: Persian CoT -> translate-first CoT", old[("kopp", "cot")], ok[("kopp_mt", "cot")])
    test("MMLU-Med EN: CoT -> CoT with options rotated", old[("mmlu", "cot")], ok[("mmlu_rot", "cot")])
    test("MMLU-Med EN (strict parser): CoT -> rotated", en_strict, st[("mmlu_rot", "cot")])
    for ds in ("fa_mmlu", "kopp", "mmlu"):
        test(f"{ds}: CoT -> oracle@4 (oracle items)", old[(ds, "cot")], ok[(ds, "oracle_k4")])
        test(f"{ds}: RRF-2@4 -> oracle@4 (oracle items)", old[(ds, "rrf2_k4")], ok[(ds, "oracle_k4")])
    test("FA_MED_MMLU: English-instruction CoT -> Persian-instruction CoT", old[("fa_mmlu", "cot")], ok[("fa_mmlu", "cot_fa_prompt")])
    test("FA_MED_MMLU: English-instruction RRF-2@4 -> Persian-instruction RRF-2@4", old[("fa_mmlu", "rrf2_k4")], ok[("fa_mmlu", "rrf2_k4_fa_prompt")])
    test("FA_MED_MMLU, Persian instructions: CoT -> RRF-2@4", ok[("fa_mmlu", "cot_fa_prompt")], ok[("fa_mmlu", "rrf2_k4_fa_prompt")])
    # oracle effect, Persian vs English, on the aligned items in both oracle subsets
    both = sorted(set(ok[("fa_mmlu", "oracle_k4")]) & set(ok[("mmlu", "oracle_k4")]))
    de = [int(ok[("mmlu", "oracle_k4")][q]) - int(old[("mmlu", "cot")][q]) for q in both]
    df = [int(ok[("fa_mmlu", "oracle_k4")][q]) - int(old[("fa_mmlu", "cot")][q]) for q in both]
    inter = {"test": "oracle effect FA vs EN (items in both oracle subsets)", "n": len(both),
             "oracle_effect_en": round(100 * sum(de) / max(1, len(both)), 2), "oracle_effect_fa": round(100 * sum(df) / max(1, len(both)), 2),
             "diff_b_minus_a": round(100 * (sum(df) - sum(de)) / max(1, len(both)), 2),
             "p_raw": sign_test([y - x for x, y in zip(de, df)]), "method": "exact sign test"}
    tests.append(inter)
    ps = [t["p_raw"] for t in tests]
    for t, h, q in zip(tests, holm(ps), bh(ps)):
        t["p_holm"], t["q_bh"] = round(h, 4), round(q, 4)
    mt = os.path.join(a.run, "results", "kopp_mt_report.json")
    mt = json.load(open(mt)) if os.path.exists(mt) else {}
    oracle_en = os.path.join(a.run, "results", "oracle_en.jsonl.gz")
    out = {"model": "Qwen/Qwen2.5-7B-Instruct", "accuracy": info, "tests": tests,
           "kopp_translation": {"n": mt.get("n"), "parse_failures": mt.get("parse_failures")},
           "oracle_sizes": {"fa_mmlu": len(ok[("fa_mmlu", "oracle_k4")]), "kopp": len(ok[("kopp", "oracle_k4")]),
                            "mmlu": len(ok[("mmlu", "oracle_k4")]), "fa_and_en": len(both)},
           "timing": jl(os.path.join(a.run, "results", "timing.jsonl")),
           "note": "official MIRAGE parser unless stated; strict parser counts unparsable answers as wrong"}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    with gzip.open(os.path.join(a.out_dir, "predictions.jsonl.gz"), "wt") as f:
        for (ds, conf), d in sorted(new.items()):
            for q, v in sorted(d.items()):
                f.write(json.dumps({"dataset": ds, "config": conf, "qid": q, "gold": v["gold"], "pred": v["pred"], "strict": v["strict"]}) + "\n")
    if os.path.exists(oracle_en):
        import shutil
        shutil.copy(oracle_en, os.path.join(P, "oracle_retrieval_en.jsonl.gz"))
    print(json.dumps(info, indent=1))
    for t in tests:
        print(t)


if __name__ == "__main__":
    main()
