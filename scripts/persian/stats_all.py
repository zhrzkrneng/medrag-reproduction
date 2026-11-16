"""All headline paired comparisons of the Persian extension in one family, with
multiple-comparison correction and bootstrap confidence intervals.

Inputs: only the committed files under results/ (ids, letters, booleans), so this needs no GPU and
no gaokerena data. Output: results/persian/stats_all.json and a Markdown table (stats_all.md).

  * p_raw: exact two-sided McNemar test (same function as every other analysis here).
  * p_holm: Holm (1979) step-down correction, controls the family-wise error rate.
  * q_bh: Benjamini-Hochberg (1995) adjusted p, controls the false discovery rate.
  * 95% CIs: paired percentile bootstrap over questions (10,000 resamples, seed 0) for each
    accuracy and for each accuracy difference (b - a).

Exploratory subgroup analyses (gold in the top k, strictly parsed subsets, per subject) are not in
the family; they stay labelled as exploratory in the README.

Usage: python scripts/persian/stats_all.py
"""
import gzip
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from medrag_repro.evaluate import official_prediction  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

R = os.path.join(ROOT, "results")
P = os.path.join(R, "persian")
B, SEED = 10_000, 0


def rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def letters(path, col):
    return {r["qid"]: r[col] == r["gold"] for r in rows(path)}


def booleans(path, col):
    return {r["qid"]: bool(r[col]) for r in rows(path)}


def english(conf):
    return {r["qid"]: official_prediction(r["response"]) == r["answer"]
            for r in rows(os.path.join(R, "qwen_full", "Qwen__Qwen2.5-7B-Instruct", "mmlu", f"{conf}.jsonl"))}


def load():
    c = {"fa_mmlu": {}, "kopp": {}}
    m = c["fa_mmlu"]
    m["Qwen EN CoT"] = english("cot")
    m["Qwen EN Textbooks@32"] = english("rrf2_k32")
    for ds in ("fa_mmlu", "kopp"):
        d = c[ds]
        f = os.path.join(P, "run_fa", f"{ds}_predictions.jsonl.gz")
        d["Qwen FA CoT"] = letters(f, "cot")
        d["Qwen FA web@4"] = letters(f, "rrf2_k4")
        for tag, sub in (("MedRAG prompt", "rq2"), ("GK prompt", "rq2_gk")):
            f = os.path.join(P, sub, f"{ds}_predictions.jsonl.gz")
            for col, name in (("aya_cot", "Aya CoT"), ("aya_rrf2_k4", "Aya web@4"),
                              ("gaokerena_cot", "Gaokerena CoT"), ("gaokerena_rrf2_k4", "Gaokerena web@4"),
                              ("qwen_cot", "Qwen FA CoT")):
                if col == "qwen_cot" and sub == "rq2":
                    continue
                d[f"{name} [{tag}]"] = letters(f, col)
    m["Qwen FA web@32"] = letters(os.path.join(P, "run_fa32", "fa_mmlu_predictions.jsonl.gz"), "rrf2_k32")
    x = os.path.join(P, "xling_rag")
    m["2x2 EN q + Textbooks@4"] = booleans(os.path.join(x, "1_en_q__textbooks_predictions.jsonl.gz"), "correct")
    m["2x2 FA q + Textbooks@4"] = booleans(os.path.join(x, "2_fa_q__textbooks_predictions.jsonl.gz"), "correct")
    m["2x2 FA q + FA web@4 (BGE-M3)"] = booleans(os.path.join(x, "3_fa_q__fa_web_predictions.jsonl.gz"), "correct")
    m["2x2 EN q + FA web@4 (BGE-M3)"] = booleans(os.path.join(x, "4_en_q__fa_web_predictions.jsonl.gz"), "correct")
    t = os.path.join(P, "translate", "predictions.jsonl.gz")
    m["Translate-first CoT"] = booleans(t, "mt_cot_correct")
    m["Translate-first Textbooks@4"] = booleans(t, "mt_textbooks_k4_correct")
    e = os.path.join(P, "en_think", "predictions.jsonl.gz")
    m["Think-in-English CoT"] = letters(e, "cot_en_think")
    m["Think-in-English Textbooks@4"] = letters(e, "textbooks_k4_en_think")
    return c


# (research question, dataset, condition a, condition b); diff = acc(b) - acc(a)
def family():
    F = [
        ("RQ1 language gap", "fa_mmlu", "Qwen EN CoT", "Qwen FA CoT"),
        ("RQ1 English RAG (reproduction)", "fa_mmlu", "Qwen EN CoT", "Qwen EN Textbooks@32"),
        ("RQ1 Persian RAG k=4", "fa_mmlu", "Qwen FA CoT", "Qwen FA web@4"),
        ("RQ1 Persian RAG k=4", "kopp", "Qwen FA CoT", "Qwen FA web@4"),
        ("RQ1 Persian RAG k=32", "fa_mmlu", "Qwen FA CoT", "Qwen FA web@32"),
        ("RQ1 k=32 vs k=4", "fa_mmlu", "Qwen FA web@4", "Qwen FA web@32"),
        ("RQ1 2x2: question language (Textbooks)", "fa_mmlu", "2x2 FA q + Textbooks@4", "2x2 EN q + Textbooks@4"),
        ("RQ1 2x2: question language (FA web)", "fa_mmlu", "2x2 FA q + FA web@4 (BGE-M3)", "2x2 EN q + FA web@4 (BGE-M3)"),
        ("RQ1 2x2: corpus (FA question)", "fa_mmlu", "2x2 FA q + Textbooks@4", "2x2 FA q + FA web@4 (BGE-M3)"),
        ("RQ1 2x2: corpus (EN question)", "fa_mmlu", "2x2 EN q + FA web@4 (BGE-M3)", "2x2 EN q + Textbooks@4"),
        ("RQ1 translate-first", "fa_mmlu", "Qwen FA CoT", "Translate-first CoT"),
        ("RQ1 translate-first + RAG vs same RAG in Persian", "fa_mmlu", "2x2 FA q + Textbooks@4", "Translate-first Textbooks@4"),
        ("RQ1 translate-first: RAG effect", "fa_mmlu", "Translate-first CoT", "Translate-first Textbooks@4"),
        ("RQ1 translate-first vs English original", "fa_mmlu", "Translate-first CoT", "Qwen EN CoT"),
        ("RQ1 think in English (CoT)", "fa_mmlu", "Qwen FA CoT", "Think-in-English CoT"),
        ("RQ1 think in English (Textbooks@4)", "fa_mmlu", "2x2 FA q + Textbooks@4", "Think-in-English Textbooks@4"),
    ]
    for ds in ("fa_mmlu", "kopp"):
        for tag in ("MedRAG prompt", "GK prompt"):
            F += [
                (f"RQ2 fine-tuning effect [{tag}]", ds, f"Aya CoT [{tag}]", f"Gaokerena CoT [{tag}]"),
                (f"RQ2 RAG effect, Aya [{tag}]", ds, f"Aya CoT [{tag}]", f"Aya web@4 [{tag}]"),
                (f"RQ2 RAG effect, Gaokerena [{tag}]", ds, f"Gaokerena CoT [{tag}]", f"Gaokerena web@4 [{tag}]"),
                (f"RQ2 head-to-head: Aya+RAG vs Gaokerena [{tag}]", ds, f"Gaokerena CoT [{tag}]", f"Aya web@4 [{tag}]"),
            ]
        F.append(("RQ2 Qwen vs Aya [MedRAG prompt]", ds, "Aya CoT [MedRAG prompt]", "Qwen FA CoT"))
        for name in ("Aya CoT", "Aya web@4", "Gaokerena CoT", "Gaokerena web@4"):
            F.append((f"RQ2 prompt effect, {name}", ds, f"{name} [MedRAG prompt]", f"{name} [GK prompt]"))
        F.append(("RQ2 prompt effect, Qwen CoT", ds, "Qwen FA CoT", "Qwen FA CoT [GK prompt]"))
    return F


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    out, run = [0.0] * m, 1.0
    for k, i in enumerate(order):
        rank = m - k
        run = min(run, ps[i] * m / rank)
        out[i] = run
    return out


def ci(vals):
    vals = sorted(vals)
    return [round(100 * vals[int(0.025 * len(vals))], 2), round(100 * vals[int(0.975 * len(vals)) - 1], 2)]


def boot(a, b, rng):
    n = len(a)
    acc_a, acc_b, diff = [], [], []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sum(a[i] for i in idx) / n
        sb = sum(b[i] for i in idx) / n
        acc_a.append(sa), acc_b.append(sb), diff.append(sb - sa)
    return ci(acc_a), ci(acc_b), ci(diff)


def main():
    c = load()
    rng = random.Random(SEED)
    res = []
    for rq, ds, ca, cb in family():
        A, Bd = c[ds][ca], c[ds][cb]
        q = sorted(set(A) & set(Bd))
        assert len(q) == len(A) == len(Bd), (rq, ds, ca, cb, len(A), len(Bd), len(q))
        a, b = [A[i] for i in q], [Bd[i] for i in q]
        cia, cib, cid = boot(a, b, rng)
        res.append({"question": rq, "dataset": ds, "a": ca, "b": cb, "n": len(q),
                    "acc_a": round(100 * sum(a) / len(q), 2), "acc_a_ci95": cia,
                    "acc_b": round(100 * sum(b) / len(q), 2), "acc_b_ci95": cib,
                    "diff_b_minus_a": round(100 * (sum(b) - sum(a)) / len(q), 2), "diff_ci95": cid,
                    "b_only_correct": sum(y and not x for x, y in zip(a, b)),
                    "a_only_correct": sum(x and not y for x, y in zip(a, b)),
                    "p_raw": mcnemar(a, b)})
    ps = [r["p_raw"] for r in res]
    for r, h, q in zip(res, holm(ps), bh(ps)):
        r["p_holm"], r["q_bh"] = round(h, 4), round(q, 4)
        r["significant_holm_0.05"] = h < 0.05
        r["significant_bh_0.05"] = q < 0.05
    out = {"method": __doc__.split("Usage")[0].strip(), "family_size": len(res), "bootstrap_resamples": B,
           "seed": SEED, "comparisons": res}
    json.dump(out, open(os.path.join(P, "stats_all.json"), "w"), indent=1)
    lines = ["| Question | Dataset | a → b | Acc a | Acc b | Δ (95% CI) | p raw | p Holm | q BH |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        star = " **" if r["significant_holm_0.05"] else ""
        lines.append(f"| {r['question']} | {r['dataset']} | {r['a']} → {r['b']} | {r['acc_a']} | {r['acc_b']} | "
                     f"{r['diff_b_minus_a']:+.2f} [{r['diff_ci95'][0]:+.2f}, {r['diff_ci95'][1]:+.2f}] | "
                     f"{r['p_raw']:.4f} | {r['p_holm']:.4f}{star} | {r['q_bh']:.4f} |")
    open(os.path.join(P, "stats_all.md"), "w").write(
        f"# All headline comparisons ({len(res)} tests, one family)\n\n"
        "Exact McNemar p; Holm (FWER) and Benjamini-Hochberg (FDR) over all tests; paired bootstrap 95% CI "
        f"({B:,} resamples, seed {SEED}). ** = significant after Holm at 0.05. Generated by "
        "`scripts/persian/stats_all.py`.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{len(res)} tests; Holm-significant: {sum(r['significant_holm_0.05'] for r in res)}; "
          f"BH-significant: {sum(r['significant_bh_0.05'] for r in res)}; raw p<0.05: {sum(r['p_raw'] < 0.05 for r in res)}")


if __name__ == "__main__":
    main()
