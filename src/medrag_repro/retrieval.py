"""Retrieval for MedRAG on the Textbooks corpus: BM25, MedCPT and RRF.

This mirrors the official implementation in
https://github.com/Teddy-XiongGZ/MedRAG/blob/7599a728a28789fd601728c08d313b1148051f41/src/utils.py
  * BM25: Pyserini/Lucene index over the chunk files (DefaultLuceneDocumentGenerator,
    which indexes the ``contents`` field).
  * MedCPT: article = ncbi/MedCPT-Article-Encoder on [title, content] pairs,
    query = ncbi/MedCPT-Query-Encoder, CLS pooling, exact inner-product search
    (faiss.IndexFlatIP), no normalisation.
  * RRF: each retriever returns max(2k, 100) hits; fused score is
    sum 1 / (rrf_k + rank + 1) with rrf_k = 100.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"


def load_corpus(chunk_dir: str) -> list[dict]:
    """Snippets in the same order MedRAG uses to build its faiss index (sorted file names)."""
    snippets = []
    for fname in sorted(f for f in os.listdir(chunk_dir) if f.endswith(".jsonl")):
        with open(os.path.join(chunk_dir, fname)) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    snippets.append({"id": d["id"], "title": d["title"], "content": d["content"]})
    return snippets


# ---------------------------------------------------------------- BM25
def build_bm25_index(chunk_dir: str, index_dir: str, threads: int = 4) -> None:
    if os.path.exists(index_dir):
        return
    subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                    "--input", chunk_dir, "--index", index_dir,
                    "--generator", "DefaultLuceneDocumentGenerator", "--threads", str(threads)], check=True)


class BM25:
    def __init__(self, index_dir: str):
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(index_dir)  # Pyserini defaults (k1=0.9, b=0.4), as in MedRAG

    def search(self, query: str, k: int) -> tuple[list[str], list[float]]:
        hits = self.searcher.search(query, k=k)
        return [h.docid for h in hits], [float(h.score) for h in hits]


# ---------------------------------------------------------------- MedCPT
def cls_sentence_transformer(name: str, device: str):
    """MedRAG's CustomizeSentenceTransformer: Transformer + CLS pooling."""
    from sentence_transformers import SentenceTransformer, models
    tr = models.Transformer(name)
    pool = models.Pooling((tr.get_embedding_dimension() if hasattr(tr, "get_embedding_dimension") else tr.get_word_embedding_dimension()), "cls")
    return SentenceTransformer(modules=[tr, pool], device=device)


def embed_corpus(snippets: list[dict], out_path: str, device: str, batch_size: int = 32) -> np.ndarray:
    if os.path.exists(out_path):
        return np.load(out_path)
    import torch
    model = cls_sentence_transformer(ARTICLE_ENCODER, device)
    model.eval()
    texts = [[s["title"], s["content"]] for s in snippets]
    with torch.no_grad():
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    emb = emb.astype(np.float32)
    np.save(out_path, emb)
    return emb


class MedCPT:
    def __init__(self, embeddings: np.ndarray, ids: list[str], device: str = "cpu"):
        import faiss
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.ids = ids
        self.encoder = cls_sentence_transformer(QUERY_ENCODER, device)
        self.encoder.eval()

    def search_many(self, queries: list[str], k: int, batch_size: int = 32):
        import torch
        with torch.no_grad():
            q = self.encoder.encode(queries, batch_size=batch_size, convert_to_numpy=True).astype(np.float32)
        scores, idx = self.index.search(q, k)
        return [([self.ids[i] for i in row], s.tolist()) for row, s in zip(idx, scores)]


# ---------------------------------------------------------------- RRF
def rrf(ranked_lists: list[list[str]], k: int, rrf_k: int = 100) -> tuple[list[str], list[float]]:
    """Reciprocal Rank Fusion exactly as MedRAG.RetrievalSystem.merge.

    Ties keep first-seen order (Python's sort is stable), matching the original.
    """
    fused: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    items = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    return [i for i, _ in items], [s for _, s in items]
