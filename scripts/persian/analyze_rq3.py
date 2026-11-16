"""RQ3: do MedRAG's findings on retrievers and on k hold in Persian? (kaggle/rq3 + earlier runs)

Qwen2.5-7B-Instruct, MedRAG prompts, Persian web corpus.
  * retrievers at k = 4: BM25, BGE-M3, RRF-2 on FA_MED_MMLU and KOPP
    (FA_MED_MMLU BGE-M3@4 = the 2x2 run, whose top-4 lists are identical for all 1,089 questions)
  * k sweep on FA_MED_MMLU: RRF-2 at k = 1, 2, 4, 8, 16, 32, with the English sweep on the same items
Official MIRAGE parser; exact McNemar; Holm and BH within this family; answer coverage (correct option's
text in the k snippets, lexical proxy) per setting. Writes ids / letters / booleans / counts only, and
a figure of accuracy against k (English vs Persian).

Usage: python scripts/persian/analyze_rq3.py --run <kernel output dir> --fa_chunks data_local/fa_corpus_chunks.jsonl.gz
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
from stats_all import bh, holm  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

SLUG = "Qwen__Qwen2.5-7B-Instruct"
KS = [1, 2, 4, 8, 16, 32]


def jl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--fa_chunks", default=os.path.join(ROOT, "data_local", "fa_corpus_chunks.jsonl.gz"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "persian", "rq3"))
    p.add_argument("--figure", default=os.path.join(ROOT, "docs", "k_sweep_fa.png"))
    a = p.parse_args()
    P = os.path.join(ROOT, "results", "persian")
    ok, pred = {}, {}
    # new runs
    for ds in ("fa_mmlu", "kopp"):
        d = os.path.join(a.run, "results", SLUG, ds)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            conf = fn[:-6]
            rows = jl(os.path.join(d, fn))
            pred[(ds, conf)] = {r["qid"]: (r["answer"], official_prediction(r["response"])) for r in rows}
            ok[(ds, conf)] = {q: g == x for q, (g, x) in pred[(ds, conf)].items()}
    # earlier runs on the same lists and prompts
    for ds in ("fa_mmlu", "kopp"):
        for r in jl(os.path.join(P, "run_fa", f"{ds}_predictions.jsonl.gz")):
            ok.setdefault((ds, "cot"), {})[r["qid"]] = r["cot"] == r["gold"]
            ok.setdefault((ds, "rrf2_k4"), {})[r["qid"]] = r["rrf2_k4"] == r["gold"]
    for r in jl(os.path.join(P, "run_fa32", "fa_mmlu_predictions.jsonl.gz")):
        ok.setdefault(("fa_mmlu", "rrf2_k32"), {})[r["qid"]] = r["rrf2_k32"] == r["gold"]
    if ("fa_mmlu", "bge_m3_k4") not in ok:
        ok[("fa_mmlu", "bge_m3_k4")] = {r["qid"]: bool(r["correct"]) for r in
                                         jl(os.path.join(P, "xling_rag", "3_fa_q__fa_web_predictions.jsonl.gz"))}
    en = {}
    base = os.path.join(ROOT, "results", "qwen_full", SLUG, "mmlu")
    for conf in ["cot"] + [f"rrf2_k{k}" for k in KS]:
        en[conf] = {r["qid"]: official_prediction(r["response"]) == r["answer"] for r in jl(os.path.join(base, f"{conf}.jsonl"))}

    # coverage (correct option's text in the top k), from the committed retrieval lists and the local corpus
    ret = {(r["dataset"], r["qid"]): r for r in jl(os.path.join(P, "fa_retrieval_top32.jsonl.gz"))}
    need = {i for r in ret.values() for key in ("bm25", "bge_m3", "rrf2") for i in r[key][:32]}
    text = {}
    for line in gzip.open(a.fa_chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            text[s["id"]] = norm(s["content"])
    bench = {**json.load(open(os.path.join(ROOT, "data_local", "fa_mmlu_benchmark.json"))),
             **json.load(open(os.path.join(ROOT, "data_local", "kopp_benchmark.json")))}

    def coverage(ds, retr, k):
        hit = n = 0
        for q, it in bench[ds].items():
            t = norm(it["options"][it["answer"]])
            if t.strip() in GENERIC or len(t.strip()) < 3:
                continue
            n += 1
            hit += any(t in text[i] for i in ret[(ds, q)][retr][:k])
        return round(100 * hit / n, 1)

    acc = lambda d: round(100 * sum(d.values()) / len(d), 2)  # noqa: E731
    table = {"retrievers_k4": {}, "k_sweep_fa_mmlu": {}, "k_sweep_en_mmlu": {}}
    for ds in ("fa_mmlu", "kopp"):
        for retr in ("bm25", "bge_m3", "rrf2"):
            c = (ds, f"{retr}_k4")
            if c in ok:
                table["retrievers_k4"][f"{ds}/{retr}"] = {"acc_%": acc(ok[c]), "n": len(ok[c]),
                                                          "coverage_top4_%": coverage(ds, retr, 4)}
        table["retrievers_k4"][f"{ds}/cot"] = {"acc_%": acc(ok[(ds, "cot")]), "n": len(ok[(ds, "cot")])}
    for k in KS:
        c = ("fa_mmlu", f"rrf2_k{k}")
        if c in ok:
            table["k_sweep_fa_mmlu"][k] = {"acc_%": acc(ok[c]), "n": len(ok[c]), "coverage_%": coverage("fa_mmlu", "rrf2", k)}
        table["k_sweep_en_mmlu"][k] = {"acc_%": acc(en[f"rrf2_k{k}"])}
    table["k_sweep_fa_mmlu"]["cot"] = acc(ok[("fa_mmlu", "cot")])
    table["k_sweep_en_mmlu"]["cot"] = acc(en["cot"])

    tests = []

    def test(name, A, B):
        qs = sorted(set(A) & set(B))
        x, y = [A[q] for q in qs], [B[q] for q in qs]
        tests.append({"test": name, "n": len(qs), "acc_a": round(100 * sum(x) / len(qs), 2), "acc_b": round(100 * sum(y) / len(qs), 2),
                      "diff_b_minus_a": round(100 * (sum(y) - sum(x)) / len(qs), 2), "p_raw": mcnemar(x, y)})

    for ds in ("fa_mmlu", "kopp"):
        have = [r for r in ("bm25", "bge_m3", "rrf2") if (ds, f"{r}_k4") in ok]
        for r in have:
            test(f"{ds}: CoT -> {r}@4", ok[(ds, "cot")], ok[(ds, f"{r}_k4")])
        for i, r1 in enumerate(have):
            for r2 in have[i + 1:]:
                test(f"{ds}: {r1}@4 -> {r2}@4", ok[(ds, f"{r1}_k4")], ok[(ds, f"{r2}_k4")])
    for k in KS:
        c = ("fa_mmlu", f"rrf2_k{k}")
        if c in ok and k != 4:  # k = 4 is already tested above (RRF-2 in the retriever comparison)
            test(f"fa_mmlu: CoT -> RRF-2@{k}", ok[("fa_mmlu", "cot")], ok[c])
    if ("fa_mmlu", "rrf2_k1") in ok and ("fa_mmlu", "rrf2_k16") in ok:
        test("fa_mmlu: RRF-2@1 -> RRF-2@16", ok[("fa_mmlu", "rrf2_k1")], ok[("fa_mmlu", "rrf2_k16")])
    ps = [t["p_raw"] for t in tests]
    for t, h, q in zip(tests, holm(ps), bh(ps)):
        t["p_holm"], t["q_bh"] = round(h, 4), round(q, 4)
    timing = os.path.join(a.run, "results", "timing.jsonl")
    out = {"model": "Qwen/Qwen2.5-7B-Instruct", "prompt": "MedRAG", "tables": table, "tests": tests,
           "timing": jl(timing) if os.path.exists(timing) else None}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    with gzip.open(os.path.join(a.out_dir, "predictions.jsonl.gz"), "wt") as f:
        for (ds, conf), d in sorted(pred.items()):
            for q, (g, x) in sorted(d.items()):
                f.write(json.dumps({"dataset": ds, "config": conf, "qid": q, "gold": g, "pred": x}) + "\n")
    print(json.dumps(table, indent=1))
    for t in tests:
        print(t)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=200)
    ks_fa = [k for k in KS if k in table["k_sweep_fa_mmlu"]]
    ax.plot(KS, [table["k_sweep_en_mmlu"][k]["acc_%"] for k in KS], "o-", color="#1f5fa8", label="English (Textbooks)")
    ax.plot(ks_fa, [table["k_sweep_fa_mmlu"][k]["acc_%"] for k in ks_fa], "s-", color="#c0392b", label="Persian (web corpus)")
    ax.axhline(table["k_sweep_en_mmlu"]["cot"], color="#1f5fa8", ls="--", lw=1)
    ax.axhline(table["k_sweep_fa_mmlu"]["cot"], color="#c0392b", ls="--", lw=1)
    ax.text(1.05, table["k_sweep_en_mmlu"]["cot"] + 0.6, "CoT (EN)", color="#1f5fa8", fontsize=7)
    ax.text(1.05, table["k_sweep_fa_mmlu"]["cot"] + 0.6, "CoT (FA)", color="#c0392b", fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("Number of retrieved snippets (k)")
    ax.set_ylabel("Accuracy on MMLU-Med items (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.figure), exist_ok=True)
    fig.savefig(a.figure)
    print("figure", a.figure)


if __name__ == "__main__":
    main()
