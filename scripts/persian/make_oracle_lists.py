"""Oracle retrieval lists for the Persian corpus (review check: does retrieval help in Persian when the answer is there?).

For each FA_MED_MMLU / KOPP question, the snippets among the union of the committed top-32 lists (RRF-2, then BM25,
then BGE-M3) that contain the correct option's text (same normalisation and exclusions as answer_coverage.py) are
put first, and the list is filled up to 4 with the RRF-2 order. Questions with no such snippet are left out.
Compared on the same questions with CoT and RRF-2@4, this asks whether the model uses Persian context when it
holds the answer. The match is lexical: a snippet can mention the option without supporting it, so this is an
"answer-mentioning" oracle, an upper bound on what this corpus can give rather than gold evidence.

Writes ids only: results/persian/oracle_retrieval.jsonl.gz ({dataset, qid, oracle: [snippet ids], n_answer}).
The same construction runs on English MMLU-Med with Textbooks inside the review kernel (the corpus is downloaded there),
so the oracle effect can be compared between languages on the same items.
Usage: python scripts/persian/make_oracle_lists.py
       python scripts/persian/make_oracle_lists.py --retrieved results/retrieval/textbooks_top32.jsonl.gz \
           --keys rrf2 bm25 medcpt --corpus tb.jsonl.gz --benchmarks data/mirage_benchmark.json --datasets mmlu \
           --out oracle_en.jsonl.gz
"""
import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
from answer_coverage import GENERIC, norm  # noqa: E402

P = os.path.join(ROOT, "results", "persian")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retrieved", default=os.path.join(P, "fa_retrieval_top32.jsonl.gz"))
    p.add_argument("--keys", nargs="+", default=["rrf2", "bm25", "bge_m3"], help="first key gives the fill order")
    p.add_argument("--corpus", default=os.path.join(ROOT, "data_local", "fa_corpus_chunks.jsonl.gz"))
    p.add_argument("--benchmarks", nargs="+", default=[os.path.join(ROOT, "data_local", "fa_mmlu_benchmark.json"),
                                                        os.path.join(ROOT, "data_local", "kopp_benchmark.json")])
    p.add_argument("--datasets", nargs="+", default=["fa_mmlu", "kopp"])
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", default=os.path.join(P, "oracle_retrieval.jsonl.gz"))
    a = p.parse_args()
    ret = {(r["dataset"], r["qid"]): r for r in map(json.loads, gzip.open(a.retrieved, "rt"))}
    need = {i for (ds, _), r in ret.items() if ds in a.datasets for key in a.keys for i in r[key][:32]}
    text = {}
    for line in gzip.open(a.corpus, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            text[s["id"]] = norm(s["content"])
    bench = {}
    for b in a.benchmarks:
        bench.update(json.load(open(b)))
    fill = a.keys[0]
    stats, rows = {}, []
    for ds in a.datasets:
        n_meas = 0
        for q, it in sorted(bench[ds].items()):
            t = norm(it["options"][it["answer"]])
            if t.strip() in GENERIC or len(t.strip()) < 3:
                continue
            n_meas += 1
            r = ret[(ds, q)]
            seen, hits = set(), []
            for key in a.keys:
                for i in r[key][:32]:
                    if i not in seen:
                        seen.add(i)
                        if t in text[i]:
                            hits.append(i)
            if not hits:
                continue
            lst = hits[: a.k]
            lst += [i for i in r[fill] if i not in lst][: a.k - len(lst)]
            rows.append({"dataset": ds, "qid": q, "oracle": lst, "n_answer": min(len(hits), a.k),
                         "in_top4": any(t in text[i] for i in r[fill][:4])})
        sel = [x for x in rows if x["dataset"] == ds]
        stats[ds] = {"measurable": n_meas, "with_answer_snippet": len(sel), "already_in_top4": sum(x["in_top4"] for x in sel)}
    with gzip.open(a.out, "wt") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
