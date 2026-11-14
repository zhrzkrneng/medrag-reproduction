"""How often is the correct option's text in the top k Persian snippets (RRF-2, Persian web corpus),
for k = 4, 8, 16, 32? Same lexical proxy as analyze_fa_runs.py; no GPU. Writes counts only.

Usage: python scripts/persian/fa_coverage_by_k.py --fa_chunks data_local/fa_corpus_chunks.jsonl.gz \
           --out results/persian/fa_coverage_by_k.json
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.dirname(__file__))
from answer_coverage import GENERIC, norm  # noqa: E402

KS = [4, 8, 16, 32]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--fa_chunks", required=True)
    a.add_argument("--out", required=True)
    a = a.parse_args()
    bench = {**json.load(open(f"{ROOT}/data_local/fa_mmlu_benchmark.json")),
             **json.load(open(f"{ROOT}/data_local/kopp_benchmark.json"))}
    ret = {(r["dataset"], r["qid"]): r["rrf2"]
           for r in map(json.loads, gzip.open(f"{ROOT}/results/persian/fa_retrieval_top32.jsonl.gz", "rt"))}
    need = {i for v in ret.values() for i in v}
    text = {}
    for line in gzip.open(a.fa_chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            text[s["id"]] = norm(s["content"])
    rep = {}
    for ds in ["fa_mmlu", "kopp"]:
        c, m = {k: 0 for k in KS}, 0
        for q, it in bench[ds].items():
            t = norm(it["options"][it["answer"]])
            if t.strip() in GENERIC or len(t.strip()) < 3:
                continue  # not measurable
            m += 1
            for k in KS:
                c[k] += any(t in text[i] for i in ret[(ds, q)][:k])
        rep[ds] = {"questions": len(bench[ds]), "measurable": m,
                   "gold_in_topk": {str(k): v for k, v in c.items()},
                   "gold_in_topk_%": {str(k): round(100 * v / m, 1) for k, v in c.items()}}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
