"""Embed a MedRAG corpus with the MedCPT article encoder (GPU recommended).

The official pre-computed Textbooks embeddings (SharePoint link in MedRAG's
src/utils.py) returned HTTP 403 when we tried them on 2026-11-08, so we
re-compute them with the same model, [title, content] input pairs and CLS
pooling as MedRAG's ``embed`` function.

Usage: python scripts/embed_corpus.py --chunk_dir <corpus>/chunk --out medcpt_article_emb.npy
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medrag_repro.retrieval import embed_corpus, load_corpus  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--chunk_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--device", default="cuda")
p.add_argument("--batch_size", type=int, default=64)
a = p.parse_args()

t = time.time()
snippets = load_corpus(a.chunk_dir)
emb = embed_corpus(snippets, a.out, a.device, a.batch_size)
print(f"{emb.shape} embeddings for {len(snippets)} snippets in {time.time()-t:.0f}s -> {a.out}")
