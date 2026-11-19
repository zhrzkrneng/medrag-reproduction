"""One Holm / BH family for the whole study, with each test labelled confirmatory or exploratory,
the language x retrieval interaction, and every comparison repeated on answers that did not hit the
answer-length cap.

  * family: the 44 tests of stats_all.py + the second model family (Llama) + RQ3 (retrievers, k sweep)
    + three interaction tests; duplicates removed. Holm (FWER) and BH (FDR) over all of them.
  * confirmatory = planned before the corresponding results were seen (RQ1 language gap and Persian
    RAG, RQ2 fine-tuning vs RAG under MedRAG's prompt, RQ3, the second model). exploratory = added after
    seeing earlier results (translate-first, "think in English", Gaokerena's prompt, prompt effects,
    the 2x2, k = 32, the interaction tests).
  * interaction: is the effect of retrieval different in Persian than in English on the same 1,089
    aligned items? Per item d = correct(RAG) - correct(CoT) in each language; exact two-sided sign test
    on d_FA - d_EN (items where it is non-zero).
  * uncapped: the same comparison on items where neither run hit the length cap (capped_ids.json).

Inputs: committed files only (ids, letters, booleans). Output: results/persian/stats_study.json and .md.
Usage: python scripts/persian/stats_study.py
"""
import gzip
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
import stats_all as SA  # noqa: E402
from xling_coverage import mcnemar  # noqa: E402

P = SA.P


def preds(path):
    out = {}
    for r in SA.rows(path):
        out.setdefault((r["dataset"], r["config"]), {})[r["qid"]] = r["pred"] == r["gold"]
    return out


def sign_test(diffs):
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


EXPLORATORY = ("translate", "think in English", "GK prompt", "prompt effect", "2x2", "k=32", "interaction", "review")


def main():
    c = SA.load()
    m, k = c["fa_mmlu"], c["kopp"]
    m["Qwen EN RRF-2@4"] = SA.english("rrf2_k4")
    ll = preds(os.path.join(P, "model2", "predictions.jsonl.gz"))
    m["Llama EN CoT"], m["Llama EN RRF-2@4"] = ll[("mmlu", "cot")], ll[("mmlu", "rrf2_k4")]
    for ds, d in (("fa_mmlu", m), ("kopp", k)):
        d["Llama FA CoT"], d["Llama FA web@4"] = ll[(ds, "cot")], ll[(ds, "rrf2_k4")]
    r3 = preds(os.path.join(P, "rq3", "predictions.jsonl.gz"))
    m["Qwen FA BM25@4"], k["Qwen FA BM25@4"], k["Qwen FA BGE-M3@4"] = r3[("fa_mmlu", "bm25_k4")], r3[("kopp", "bm25_k4")], r3[("kopp", "bge_m3_k4")]
    m["Qwen FA BGE-M3@4"] = m["2x2 FA q + FA web@4 (BGE-M3)"]
    for kk in (1, 2, 8, 16):
        m[f"Qwen FA web@{kk}"] = r3[("fa_mmlu", f"rrf2_k{kk}")]

    # checks added after the self-review (kaggle/review): all exploratory
    rv = preds(os.path.join(P, "review", "predictions.jsonl.gz"))
    k["Translate-first CoT"] = rv[("kopp_mt", "cot")]
    m["Qwen EN CoT rotated"] = rv[("mmlu_rot", "cot")]
    for ds, d in (("fa_mmlu", m), ("kopp", k)):
        sub = rv[(ds, "oracle_k4")]
        d["Qwen FA oracle@4"] = sub
        d["Qwen FA CoT [oracle items]"] = {q: d["Qwen FA CoT"][q] for q in sub}
        d["Qwen FA web@4 [oracle items]"] = {q: d["Qwen FA web@4"][q] for q in sub}
    sub = rv[("mmlu", "oracle_k4")]
    m["Qwen EN oracle@4"] = sub
    m["Qwen EN CoT [oracle items]"] = {q: m["Qwen EN CoT"][q] for q in sub}
    m["Qwen EN RRF-2@4 [oracle items]"] = {q: m["Qwen EN RRF-2@4"][q] for q in sub}
    m["Qwen FA CoT [FA prompt]"], m["Qwen FA web@4 [FA prompt]"] = rv[("fa_mmlu", "cot_fa_prompt")], rv[("fa_mmlu", "rrf2_k4_fa_prompt")]

    # condition -> run in capped_ids.json
    CAP = {"Qwen EN CoT": "qwen/mmlu/cot", "Qwen EN Textbooks@32": "qwen/mmlu/rrf2_k32", "Qwen EN RRF-2@4": "qwen/mmlu/rrf2_k4",
           "Qwen FA CoT": "qwen/{ds}/cot", "Qwen FA web@4": "qwen/{ds}/rrf2_k4", "Qwen FA web@32": "qwen/fa_mmlu/rrf2_k32",
           "2x2 EN q + Textbooks@4": "qwen/mmlu/bge_tb_k4", "2x2 FA q + Textbooks@4": "qwen/fa_mmlu/bge_tb_k4",
           "2x2 FA q + FA web@4 (BGE-M3)": "qwen/fa_mmlu/bge_m3_k4", "Qwen FA BGE-M3@4": "qwen/{ds}/bge_m3_k4",
           "Translate-first CoT": "qwen/{ds}_mt/cot", "Translate-first Textbooks@4": "qwen/fa_mmlu_mt/bge_tb_k4",
           "Llama EN CoT": "llama/mmlu/cot", "Llama EN RRF-2@4": "llama/mmlu/rrf2_k4",
           "Llama FA CoT": "llama/{ds}/cot", "Llama FA web@4": "llama/{ds}/rrf2_k4", "Qwen FA BM25@4": "qwen/{ds}/bm25_k4"}
    for kk in (1, 2, 8, 16):
        CAP[f"Qwen FA web@{kk}"] = f"qwen/fa_mmlu/rrf2_k{kk}"
    CAP.update({"Qwen EN CoT rotated": "qwen/mmlu_rot/cot", "Qwen FA oracle@4": "qwen/{ds}/oracle_k4",
                "Qwen FA CoT [oracle items]": "qwen/{ds}/cot", "Qwen FA web@4 [oracle items]": "qwen/{ds}/rrf2_k4",
                "Qwen EN oracle@4": "qwen/mmlu/oracle_k4", "Qwen EN CoT [oracle items]": "qwen/mmlu/cot",
                "Qwen EN RRF-2@4 [oracle items]": "qwen/mmlu/rrf2_k4", "Qwen FA CoT [FA prompt]": "qwen/fa_mmlu/cot_fa_prompt",
                "Qwen FA web@4 [FA prompt]": "qwen/fa_mmlu/rrf2_k4_fa_prompt"})
    capped = {n: set(v["capped_qids"]) for n, v in json.load(open(os.path.join(P, "capped_ids.json")))["runs"].items()}

    def cap_set(cond, ds):
        run = CAP.get(cond)
        return None if run is None else capped.get(run.format(ds=ds))

    F = list(SA.family())
    F += [("RQ1 language gap (Llama)", "fa_mmlu", "Llama EN CoT", "Llama FA CoT"),
          ("RQ1 English RAG (Llama)", "fa_mmlu", "Llama EN CoT", "Llama EN RRF-2@4")]
    for ds in ("fa_mmlu", "kopp"):
        F.append(("RQ1 Persian RAG k=4 (Llama)", ds, "Llama FA CoT", "Llama FA web@4"))
        F += [("Second model: Qwen -> Llama, FA CoT", ds, "Qwen FA CoT", "Llama FA CoT"),
              ("Second model: Qwen -> Llama, FA RAG@4", ds, "Qwen FA web@4", "Llama FA web@4")]
    F += [("Second model: Qwen -> Llama, EN CoT", "fa_mmlu", "Qwen EN CoT", "Llama EN CoT"),
          ("Second model: Qwen -> Llama, EN RAG@4", "fa_mmlu", "Qwen EN RRF-2@4", "Llama EN RRF-2@4")]
    for ds in ("fa_mmlu", "kopp"):
        have = ["Qwen FA BM25@4", "Qwen FA BGE-M3@4", "Qwen FA web@4"]  # CoT -> RRF-2@4 is "RQ1 Persian RAG k=4"
        show = {"Qwen FA BM25@4": "BM25@4", "Qwen FA BGE-M3@4": "BGE-M3@4", "Qwen FA web@4": "RRF-2@4"}
        for h in have[:2]:
            F.append((f"RQ3 retriever: CoT -> {show[h]}", ds, "Qwen FA CoT", h))
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                F.append((f"RQ3 retriever: {show[a]} -> {show[b]}", ds, a, b))
    for kk in (1, 2, 8, 16):
        F.append((f"RQ3 k sweep: CoT -> RRF-2@{kk}", "fa_mmlu", "Qwen FA CoT", f"Qwen FA web@{kk}"))
    F.append(("RQ3 k sweep: RRF-2@1 -> RRF-2@16", "fa_mmlu", "Qwen FA web@1", "Qwen FA web@16"))
    F += [("review: translate-first on KOPP (no English original)", "kopp", "Qwen FA CoT", "Translate-first CoT"),
          ("review: options rotated (MMLU recall probe)", "fa_mmlu", "Qwen EN CoT", "Qwen EN CoT rotated")]
    for ds in ("fa_mmlu", "kopp"):
        F += [("review: oracle, CoT -> oracle@4", ds, "Qwen FA CoT [oracle items]", "Qwen FA oracle@4"),
              ("review: oracle, RRF-2@4 -> oracle@4", ds, "Qwen FA web@4 [oracle items]", "Qwen FA oracle@4")]
    F += [("review: oracle (English), CoT -> oracle@4", "fa_mmlu", "Qwen EN CoT [oracle items]", "Qwen EN oracle@4"),
          ("review: oracle (English), RRF-2@4 -> oracle@4", "fa_mmlu", "Qwen EN RRF-2@4 [oracle items]", "Qwen EN oracle@4"),
          ("review: Persian instructions, CoT", "fa_mmlu", "Qwen FA CoT", "Qwen FA CoT [FA prompt]"),
          ("review: Persian instructions, RRF-2@4", "fa_mmlu", "Qwen FA web@4", "Qwen FA web@4 [FA prompt]"),
          ("review: Persian instructions, RAG effect", "fa_mmlu", "Qwen FA CoT [FA prompt]", "Qwen FA web@4 [FA prompt]")]
    keys = [(ds, a, b) for _, ds, a, b in F]
    assert len(keys) == len(set(keys)), "duplicate test"

    res = []
    for q, ds, ca, cb in F:
        A, B = c[ds][ca], c[ds][cb]
        qs = sorted(set(A) & set(B))
        assert len(qs) == min(len(A), len(B)), (q, ds, ca, cb)  # equal sets, or a subset (oracle items)
        a, b = [A[i] for i in qs], [B[i] for i in qs]
        r = {"question": q, "type": "exploratory" if any(e in q or e in ca + cb for e in EXPLORATORY) else "confirmatory",
             "dataset": ds, "a": ca, "b": cb, "n": len(qs), "acc_a": round(100 * sum(a) / len(qs), 2),
             "acc_b": round(100 * sum(b) / len(qs), 2), "diff_b_minus_a": round(100 * (sum(b) - sum(a)) / len(qs), 2),
             "p_raw": mcnemar(a, b)}
        sa, sb = cap_set(ca, ds), cap_set(cb, ds)
        if sa is not None and sb is not None:
            keep = [i for i in qs if i not in sa and i not in sb]
            x, y = [A[i] for i in keep], [B[i] for i in keep]
            r["uncapped"] = {"n": len(keep), "acc_a": round(100 * sum(x) / len(keep), 2), "acc_b": round(100 * sum(y) / len(keep), 2),
                             "diff_b_minus_a": round(100 * (sum(y) - sum(x)) / len(keep), 2), "p": mcnemar(x, y)}
        res.append(r)

    # language x retrieval interaction on the aligned items
    inter = [("interaction: RAG effect FA vs EN, Qwen, same Textbooks corpus (BGE-M3@4)",
              ("Qwen EN CoT", "2x2 EN q + Textbooks@4"), ("Qwen FA CoT", "2x2 FA q + Textbooks@4")),
             ("interaction: RAG effect FA vs EN, Qwen, own-language corpus (RRF-2@4)",
              ("Qwen EN CoT", "Qwen EN RRF-2@4"), ("Qwen FA CoT", "Qwen FA web@4")),
             ("interaction: RAG effect FA vs EN, Llama, own-language corpus (RRF-2@4)",
              ("Llama EN CoT", "Llama EN RRF-2@4"), ("Llama FA CoT", "Llama FA web@4")),
             ("interaction (review): oracle effect FA vs EN, items in both oracle subsets",
              ("Qwen EN CoT", "Qwen EN oracle@4"), ("Qwen FA CoT", "Qwen FA oracle@4"))]
    for name, (e0, e1), (f0, f1) in inter:
        qs = sorted(set(m[e0]) & set(m[e1]) & set(m[f0]) & set(m[f1]))
        de = [int(m[e1][i]) - int(m[e0][i]) for i in qs]
        df = [int(m[f1][i]) - int(m[f0][i]) for i in qs]
        dd = [y - x for x, y in zip(de, df)]
        r = {"question": name, "type": "exploratory", "dataset": "fa_mmlu", "a": f"{e0} -> {e1}", "b": f"{f0} -> {f1}",
             "n": len(qs), "rag_effect_en": round(100 * sum(de) / len(qs), 2), "rag_effect_fa": round(100 * sum(df) / len(qs), 2),
             "diff_b_minus_a": round(100 * sum(dd) / len(qs), 2), "p_raw": sign_test(dd), "test": "exact sign test on d_FA - d_EN"}
        sets = [cap_set(x, "fa_mmlu") for x in (e0, e1, f0, f1)]
        if all(s is not None for s in sets):
            keep = [j for j, i in enumerate(qs) if not any(i in s for s in sets)]
            d2 = [dd[j] for j in keep]
            r["uncapped"] = {"n": len(keep), "rag_effect_en": round(100 * sum(de[j] for j in keep) / len(keep), 2),
                             "rag_effect_fa": round(100 * sum(df[j] for j in keep) / len(keep), 2),
                             "diff_b_minus_a": round(100 * sum(d2) / len(keep), 2), "p": sign_test(d2)}
        res.append(r)

    ps = [r["p_raw"] for r in res]
    for r, h, q in zip(res, SA.holm(ps), SA.bh(ps)):
        r["p_holm"], r["q_bh"] = round(h, 4), round(q, 4)
    conf = [i for i, r in enumerate(res) if r["type"] == "confirmatory"]
    for i, h in zip(conf, SA.holm([res[i]["p_raw"] for i in conf])):
        res[i]["p_holm_confirmatory_only"] = round(h, 4)
    out = {"method": __doc__.split("Usage")[0].strip(), "family_size": len(res),
           "n_confirmatory": len(conf), "comparisons": res}
    json.dump(out, open(os.path.join(P, "stats_study.json"), "w"), indent=1)
    L = ["| Type | Question | Dataset | Acc a → b | Δ | p raw | p Holm (all) | q BH | uncapped: n, Δ, p |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        u = r.get("uncapped")
        us = f"{u['n']}, {u['diff_b_minus_a']:+.2f}, {u['p']:.4f}" if u else "–"
        acc = f"{r['acc_a']} → {r['acc_b']}" if "acc_a" in r else f"RAG EN {r['rag_effect_en']:+.2f} / FA {r['rag_effect_fa']:+.2f}"
        L.append(f"| {r['type'][:4]} | {r['question']} | {r['dataset']} | {acc} | {r['diff_b_minus_a']:+.2f} | {r['p_raw']:.4f} | "
                 f"{r['p_holm']:.4f}{' **' if r['p_holm'] < 0.05 else ''} | {r['q_bh']:.4f} | {us} |")
    open(os.path.join(P, "stats_study.md"), "w").write(
        f"# Whole-study family ({len(res)} tests; {len(conf)} confirmatory)\n\nGenerated by `scripts/persian/stats_study.py`. "
        "Exact McNemar (sign test for interactions); Holm and BH over all tests; ** = Holm < 0.05. "
        "'uncapped' repeats the test on items where neither run hit the answer-length cap.\n\n" + "\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n{len(res)} tests, Holm-significant {sum(r['p_holm'] < 0.05 for r in res)}, BH {sum(r['q_bh'] < 0.05 for r in res)}")


if __name__ == "__main__":
    main()
