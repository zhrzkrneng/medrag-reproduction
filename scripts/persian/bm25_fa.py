"""BM25 over the Persian corpus with Lucene's PersianAnalyzer, and question-only retrieval
for a Persian benchmark in MIRAGE format (e.g. FA_MED_MMLU from prepare_fa_mmlu.py).

Same settings as the English BM25 (Pyserini defaults k1=0.9, b=0.4, as in MedRAG), except the
analyzer: "--language fa" at indexing and set_language("fa") at search time (without the latter
a Persian index returns no hits). Indexed text is the snippet content only: the "title" of these
web snippets is the source site, which carries no medical content.

Output: one JSON line per question with the top-k snippet ids and scores (ids only, no corpus text).

Usage:
  python scripts/persian/bm25_fa.py --chunks data_local/fa_corpus_chunks.jsonl.gz \
      --work_dir work_fa --benchmark data_local/fa_mmlu_benchmark.json --dataset fa_mmlu \
      --out results/persian/fa_mmlu_bm25_top32.jsonl.gz
"""
import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from medrag_repro.retrieval import BM25, build_bm25_index  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True)
    p.add_argument("--work_dir", default="work_fa")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", default="fa_mmlu")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    coll = os.path.join(a.work_dir, "collection")
    index = os.path.join(a.work_dir, "bm25_fa_index")
    t0 = time.time()
    if not os.path.exists(index):
        os.makedirs(coll, exist_ok=True)
        part, n = None, 0
        with gzip.open(a.chunks, "rt", encoding="utf-8") as f:
            for line in f:
                if n % 100_000 == 0:  # several files so Pyserini can index with several threads
                    part and part.close()
                    part = open(os.path.join(coll, f"part{n // 100_000:02d}.jsonl"), "w", encoding="utf-8")
                s = json.loads(line)
                part.write(json.dumps({"id": s["id"], "contents": s["content"]}, ensure_ascii=False) + "\n")
                n += 1
        part.close()
        build_bm25_index(coll, index, threads=a.threads, language="fa")
        print(f"indexed {n} snippets in {time.time() - t0:.0f}s", flush=True)

    bm25 = BM25(index, language="fa")
    bench = json.load(open(a.benchmark))[a.dataset]
    k_pool = max(2 * a.k, 100)  # as in MedRAG, so RRF can be added later from the same lists
    t1 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    empty = 0
    with gzip.open(a.out, "wt", encoding="utf-8") as out:
        for qid in sorted(bench):
            ids, scores = bm25.search(bench[qid]["question"], k_pool)
            empty += not ids
            out.write(json.dumps({"dataset": a.dataset, "qid": qid, "bm25": ids[: a.k],
                                  "bm25_scores": [round(x, 4) for x in scores[: a.k]]}) + "\n")
    print(f"searched {len(bench)} questions in {time.time() - t1:.0f}s; questions with no hit: {empty}")


if __name__ == "__main__":
    main()
