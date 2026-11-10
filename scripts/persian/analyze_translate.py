"""Translate-first baseline (kaggle/translate): accuracy, paired tests against the English original
and the Persian runs, translation quality against the original English MMLU text (FA_MED_MMLU is a
translation of it, so the round trip English -> Persian -> English can be scored with chrF), and
retrieval coverage at k = 4 (correct option's text, English options, as in xling_coverage.py).

Writes ids, letters, scores and booleans only (no translated text: it derives from FA_MED_MMLU).

Usage: python scripts/persian/analyze_translate.py --run <kaggle output dir> \
           --textbooks_dir <MedRAG textbooks/chunk> --out_dir results/persian/translate
(needs sacrebleu)
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from answer_coverage import norm  # noqa: E402
from medrag_repro.evaluate import official_prediction  # noqa: E402
from medrag_repro.retrieval import load_corpus  # noqa: E402
from xling_coverage import mcnemar, present_gold  # noqa: E402

M = "Qwen__Qwen2.5-7B-Instruct"


def preds(path):
    return {r["qid"]: official_prediction(r["response"]) for r in map(json.loads, open(path))}


def main():
    a = argparse.ArgumentParser()
    for x in ["run", "textbooks_dir", "out_dir"]:
        a.add_argument("--" + x, required=True)
    a = a.parse_args()
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    os.makedirs(a.out_dir, exist_ok=True)
    en = json.load(open(f"{ROOT}/data/mirage_benchmark.json"))["mmlu"]
    mt = json.load(open(f"{a.run}/fa_mmlu_mt.json"))["fa_mmlu_mt"]
    rep_mt = json.load(open(f"{a.run}/results/mt_report.json"))
    failed = set(rep_mt["failed_qids"])
    qids = sorted(mt)
    gold = {q: en[q]["answer"] for q in qids}
    assert all(mt[q]["answer"] == gold[q] for q in qids)

    ret = {r["qid"]: r["bge_tb"][:4] for r in map(json.loads, open(f"{a.run}/results/mt_retrieved.jsonl"))}
    need = {i for v in ret.values() for i in v}
    text = {s["id"]: norm(s["content"]) for s in load_corpus(a.textbooks_dir) if s["id"] in need}
    cov = [present_gold(en[q], [text[i] for i in ret[q]]) for q in qids]

    ok = lambda p: [p[q] == gold[q] for q in qids]  # noqa: E731
    runs = {
        "mt_cot": ok(preds(f"{a.run}/results/{M}/fa_mmlu_mt/cot.jsonl")),
        "mt_textbooks_k4": ok(preds(f"{a.run}/results/{M}/fa_mmlu_mt/bge_tb_k4.jsonl")),
        "en_cot": ok(preds(f"{ROOT}/results/qwen_full/{M}/mmlu/cot.jsonl")),
    }
    fa_cot = {r["qid"]: r["cot"] for r in map(json.loads, gzip.open(f"{ROOT}/results/persian/run_fa/fa_mmlu_predictions.jsonl.gz", "rt"))}
    runs["fa_cot"] = ok(fa_cot)
    for name in ["1_en_q__textbooks", "2_fa_q__textbooks", "3_fa_q__fa_web"]:
        p = {r["qid"]: r["pred"] for r in map(json.loads, gzip.open(f"{ROOT}/results/persian/xling_rag/{name}_predictions.jsonl.gz", "rt"))}
        runs[name] = ok(p)

    q_chrf = [chrf.sentence_score(mt[q]["question"], [en[q]["question"]]).score for q in qids]
    o_chrf = [sum(chrf.sentence_score(mt[q]["options"][k], [en[q]["options"][k]]).score for k in en[q]["options"])
              / len(en[q]["options"]) for q in qids]
    rep = {"n": len(qids), "translation": {
        "parse_failures_kept_in_persian": len(failed), "sec_per_q": rep_mt["sec_per_q"],
        "corpus_chrF_question": round(chrf.corpus_score([mt[q]["question"] for q in qids], [[en[q]["question"] for q in qids]]).score, 1),
        "mean_sentence_chrF_options": round(sum(o_chrf) / len(o_chrf), 1)},
        "accuracy_%": {k: round(100 * sum(v) / len(v), 2) for k, v in runs.items()},
        "paired_tests": {}, "gold_in_top4": {"mt_question": sum(cov)}}
    for x, y in [("fa_cot", "mt_cot"), ("mt_cot", "en_cot"), ("2_fa_q__textbooks", "mt_textbooks_k4"),
                 ("3_fa_q__fa_web", "mt_textbooks_k4"), ("mt_textbooks_k4", "1_en_q__textbooks"), ("mt_cot", "mt_textbooks_k4")]:
        rep["paired_tests"][f"{x} vs {y}"] = mcnemar(runs[x], runs[y])
    # does translation quality explain the remaining gap to English? (CoT, by question chrF tercile)
    order = sorted(range(len(qids)), key=lambda i: q_chrf[i])
    t = len(order) // 3
    rep["cot_by_question_chrF_tercile"] = {}
    for name, sel in [("low", order[:t]), ("mid", order[t:2 * t]), ("high", order[2 * t:])]:
        rep["cot_by_question_chrF_tercile"][name] = {
            "chrF_range": [round(q_chrf[sel[0]], 1), round(q_chrf[sel[-1]], 1)],
            "mt_cot_%": round(100 * sum(runs["mt_cot"][i] for i in sel) / len(sel), 1),
            "en_cot_%": round(100 * sum(runs["en_cot"][i] for i in sel) / len(sel), 1)}
    sel = [i for i in range(len(qids)) if cov[i]]
    rep["mt_textbooks_k4_when_gold_in_top4"] = {"n": len(sel), "mt_cot_%": round(100 * sum(runs["mt_cot"][i] for i in sel) / len(sel), 1),
                                                "mt_textbooks_k4_%": round(100 * sum(runs["mt_textbooks_k4"][i] for i in sel) / len(sel), 1)}
    with gzip.open(f"{a.out_dir}/predictions.jsonl.gz", "wt") as f:
        for i, q in enumerate(qids):
            f.write(json.dumps({"qid": q, "gold": gold[q], "mt_cot_correct": runs["mt_cot"][i],
                                "mt_textbooks_k4_correct": runs["mt_textbooks_k4"][i], "gold_in_top4": cov[i],
                                "question_chrF": round(q_chrf[i], 1), "translation_failed": q in failed}) + "\n")
    json.dump(rep, open(f"{a.out_dir}/summary.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
