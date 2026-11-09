"""Does the retrieved context contain the answer, and does the model follow it?

MIRAGE has no gold evidence for MedQA / MMLU-Med, so this uses a lexical proxy:
an answer option counts as "present" in the context when its normalised text occurs
(as whole words) in one of the top-k snippets given to the model.

For every question and k the context is put in one of four groups:
  gold_only        only the correct option appears
  distractor_only  only wrong options appear      (potentially misleading context)
  both             correct and wrong options appear
  neither          no option appears              (proxy cannot judge)
and CoT vs RAG accuracy is compared inside each group. "Follows distractor" is the share of
answers in distractor_only that pick a wrong option that appears in the context.

Limitations: lexical presence is not evidence (a snippet can mention a drug to rule it out),
and long sentence-like options (common in MMLU-Med) rarely occur verbatim, so most of those
questions fall in "neither". At k=32 the context given to the model was cut to 7,168 tokens;
this script checks all 32 snippets.

Usage:
  python scripts/retrieval_quality.py --chunk_dir textbooks/chunk \
      --retrieved results/retrieval/textbooks_top32.jsonl.gz \
      --results results/qwen_full/Qwen__Qwen2.5-7B-Instruct --out results/retrieval_quality
"""
import argparse
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medrag_repro.evaluate import official_prediction  # noqa: E402
from medrag_repro.retrieval import load_corpus  # noqa: E402

KS = [1, 2, 4, 8, 16, 32]
GROUPS = ["gold_only", "distractor_only", "both", "neither"]
# options this generic say nothing about whether the context supports them
GENERIC = {"none", "all", "both", "neither", "none of the above", "all of the above", "yes", "no"}


def norm(text):
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def present(option, snippet_norms):
    o = norm(option)
    if o.strip() in GENERIC or len(o.strip()) < 3:
        return False
    return any(o in s for s in snippet_norms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk_dir", required=True)
    p.add_argument("--retrieved", required=True)
    p.add_argument("--results", required=True, help="folder with <dataset>/{cot,rrf2_k<k>}.jsonl")
    p.add_argument("--benchmark", default=os.path.join(os.path.dirname(__file__), "..", "data", "mirage_benchmark.json"))
    p.add_argument("--retriever", default="rrf2")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    bench = json.load(open(a.benchmark))
    corpus = {s["id"]: norm(s["title"] + " " + s["content"]) for s in load_corpus(a.chunk_dir)}
    retrieved = {}
    with gzip.open(a.retrieved, "rt") as f:
        for line in f:
            r = json.loads(line)
            retrieved[(r["dataset"], r["qid"])] = r[a.retriever]

    def preds(ds, conf):
        path = os.path.join(a.results, ds, f"{conf}.jsonl")
        return {r["qid"]: official_prediction(r["response"]) for r in map(json.loads, open(path))}

    report = {}
    for ds in ["medqa", "mmlu"]:
        cot = preds(ds, "cot")
        report[ds] = {}
        for k in KS:
            rag = preds(ds, f"{a.retriever}_k{k}")
            stats = {g: {"n": 0, "cot_correct": 0, "rag_correct": 0, "rag_follows": 0, "cot_follows": 0} for g in GROUPS}
            for qid, item in bench[ds].items():
                snips = [corpus[i] for i in retrieved[(ds, qid)][:k]]
                found = {o for o, text in item["options"].items() if present(text, snips)}
                gold = item["answer"] in found
                wrong = found - {item["answer"]}
                g = "both" if gold and wrong else "gold_only" if gold else "distractor_only" if wrong else "neither"
                s = stats[g]
                s["n"] += 1
                s["cot_correct"] += cot[qid] == item["answer"]
                s["rag_correct"] += rag[qid] == item["answer"]
                s["rag_follows"] += rag[qid] in wrong
                s["cot_follows"] += cot[qid] in wrong
            report[ds][k] = stats

    os.makedirs(a.out, exist_ok=True)
    json.dump(report, open(os.path.join(a.out, "retrieval_quality.json"), "w"), indent=1)

    lines = ["# Retrieval quality vs. answer accuracy (lexical proxy)", "",
             "Accuracy in % inside each group; Δ = RAG − CoT. 'follows' = share of answers that pick a wrong",
             "option present in the context (only meaningful for distractor_only and both).", ""]
    for ds in report:
        lines += [f"## {ds}", "", "| k | group | n | share | CoT acc | RAG acc | Δ | CoT follows | RAG follows |",
                  "|---|---|---|---|---|---|---|---|---|"]
        total = len(bench[ds])
        for k in KS:
            for g in GROUPS:
                s = report[ds][k][g]
                if not s["n"]:
                    continue
                pc = lambda x: 100 * x / s["n"]
                lines.append(f"| {k} | {g} | {s['n']} | {100 * s['n'] / total:.1f}% | {pc(s['cot_correct']):.1f} | "
                             f"{pc(s['rag_correct']):.1f} | {pc(s['rag_correct']) - pc(s['cot_correct']):+.1f} | "
                             f"{pc(s['cot_follows']):.1f} | {pc(s['rag_follows']):.1f} |")
        lines.append("")
    open(os.path.join(a.out, "retrieval_quality.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
