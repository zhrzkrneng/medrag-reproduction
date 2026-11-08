"""Run BM25, MedCPT and RRF-2 retrieval for MIRAGE questions on the Textbooks corpus.

Question-Only Retrieval (QOR): the query is the question text only, never the options.

Output: one JSON line per question with the top-``--k`` snippet ids (and scores)
for each retriever. Smaller k values are prefixes of these lists, because every
retriever returns max(2k, 100) = 100 hits for k <= 50 before fusion, as in MedRAG.

Usage:
  python scripts/retrieve.py --chunk_dir data/textbooks/chunk --work_dir work \
      --datasets medqa mmlu --out retrieved/textbooks_top32.jsonl
"""
import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medrag_repro.retrieval import BM25, MedCPT, build_bm25_index, embed_corpus, load_corpus, rrf  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--chunk_dir", required=True)
p.add_argument("--work_dir", default="work")
p.add_argument("--benchmark", default=os.path.join(os.path.dirname(__file__), "..", "data", "mirage_benchmark.json"))
p.add_argument("--datasets", nargs="+", default=["medqa", "mmlu"])
p.add_argument("--k", type=int, default=32)
p.add_argument("--device", default="cpu")
p.add_argument("--limit", type=int, default=None, help="only the first N questions per dataset (smoke test)")
p.add_argument("--out", required=True)
a = p.parse_args()

os.makedirs(a.work_dir, exist_ok=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
k_pool = max(2 * a.k, 100)  # MedRAG: k_ = max(k * 2, 100) per retriever for RRF

t0 = time.time()
snippets = load_corpus(a.chunk_dir)
ids = [s["id"] for s in snippets]
print(f"corpus: {len(snippets)} snippets", flush=True)

build_bm25_index(a.chunk_dir, os.path.join(a.work_dir, "bm25_index"))
bm25 = BM25(os.path.join(a.work_dir, "bm25_index"))
emb = embed_corpus(snippets, os.path.join(a.work_dir, "medcpt_article_emb.npy"), a.device)
print(f"embeddings: {emb.shape} ({time.time()-t0:.0f}s so far)", flush=True)
medcpt = MedCPT(emb, ids, a.device)

bench = json.load(open(a.benchmark))
opener = gzip.open if a.out.endswith(".gz") else open
with opener(a.out, "wt") as fout:
    for ds in a.datasets:
        qids = sorted(bench[ds])[: a.limit]  # same order as MIRAGE QADataset
        questions = [bench[ds][q]["question"] for q in qids]
        dense = medcpt.search_many(questions, k_pool)
        for qid, q, (d_ids, d_sc) in zip(qids, questions, dense):
            b_ids, b_sc = bm25.search(q, k_pool)
            f_ids, f_sc = rrf([b_ids, d_ids], a.k)
            fout.write(json.dumps({
                "dataset": ds, "qid": qid,
                "bm25": b_ids[: a.k], "bm25_scores": b_sc[: a.k],
                "medcpt": d_ids[: a.k], "medcpt_scores": d_sc[: a.k],
                "rrf2": f_ids, "rrf2_scores": f_sc,
            }) + "\n")
        print(f"{ds}: {len(qids)} questions done ({time.time()-t0:.0f}s)", flush=True)
print("wrote", a.out)
