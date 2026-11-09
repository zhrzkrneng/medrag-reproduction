"""Persian retrieval: BM25 vs BGE-M3 vs RRF-2 (BM25 + BGE-M3), answer-coverage proxy.

RRF-2 is MedRAG's fusion (1 / (100 + rank + 1) over the top-100 of each retriever), applied to
BM25 (Lucene PersianAnalyzer) and BGE-M3 instead of BM25 and MedCPT (MedCPT is English-only).
For each benchmark and k, reports how often the correct option's text and a wrong option's text
occur in the top-k snippets (the lexical proxy of scripts/persian/answer_coverage.py), and writes
the fused top-32 ids for the generation step.

Usage: python scripts/persian/compare_retrievers.py --chunks data_local/fa_corpus_chunks.jsonl.gz \
           --out results/persian/retriever_comparison.json \
           --fused_out results/persian/fa_retrieval_top32.jsonl.gz
"""
import argparse
import gzip
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from answer_coverage import coverage, norm  # noqa: E402
from medrag_repro.retrieval import rrf  # noqa: E402

KS = [1, 4, 8, 16, 32]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True)
    p.add_argument("--dense", default=os.path.join(ROOT, "results", "persian", "bge_m3_top100.jsonl.gz"))
    p.add_argument("--out", required=True)
    p.add_argument("--fused_out", required=True)
    a = p.parse_args()

    benches = {"fa_mmlu": json.load(open(os.path.join(ROOT, "data_local", "fa_mmlu_benchmark.json")))["fa_mmlu"],
               "kopp": json.load(open(os.path.join(ROOT, "data_local", "kopp_benchmark.json")))["kopp"]}
    lists = {ds: {} for ds in benches}
    for ds in benches:
        for r in map(json.loads, gzip.open(os.path.join(ROOT, "results", "persian", f"{ds}_bm25_top100.jsonl.gz"), "rt")):
            lists[ds].setdefault(r["qid"], {})["bm25"] = r["bm25"]
    for r in map(json.loads, gzip.open(a.dense, "rt")):
        lists[r["dataset"]].setdefault(r["qid"], {})["bge_m3"] = r["bge_m3"]
    for ds in lists:
        for q, d in lists[ds].items():
            d["rrf2"] = rrf([d["bm25"], d["bge_m3"]], 32)[0]

    need = {i for ds in lists for d in lists[ds].values() for lst in d.values() for i in lst[:32]}
    corpus = {}
    for line in gzip.open(a.chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            corpus[s["id"]] = norm(s["content"])

    report = {}
    for ds, bench in benches.items():
        report[ds] = {}
        for retr in ["bm25", "bge_m3", "rrf2"]:
            ret = {q: d[retr] for q, d in lists[ds].items()}
            report[ds][retr] = {k: coverage(bench, ret, corpus, k) for k in KS}
            # overlap between the two retrievers' top-32 (how complementary they are)
        report[ds]["bm25_bge_m3_top32_overlap_%"] = round(100 * sum(
            len(set(d["bm25"][:32]) & set(d["bge_m3"][:32])) for d in lists[ds].values()) / (32 * len(lists[ds])), 1)
    json.dump(report, open(a.out, "w"), indent=1)
    with gzip.open(a.fused_out, "wt") as f:
        for ds in lists:
            for q in sorted(lists[ds]):
                d = lists[ds][q]
                f.write(json.dumps({"dataset": ds, "qid": q, "bm25": d["bm25"][:32], "bge_m3": d["bge_m3"][:32],
                                    "rrf2": d["rrf2"]}) + "\n")
    for ds in report:
        print(ds, "top-32 overlap BM25/BGE-M3:", report[ds]["bm25_bge_m3_top32_overlap_%"], "%")
        for retr in ["bm25", "bge_m3", "rrf2"]:
            print(f"  {retr:7s}", {k: v["gold_present_%"] for k, v in report[ds][retr].items()})


if __name__ == "__main__":
    main()
