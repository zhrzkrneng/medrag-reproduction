"""Chunk the Persian medical corpus (gaokerena/medical_corpus) the way MedRAG chunks Textbooks.

MedRAG (src/data/textbooks.py @ 7599a72) uses LangChain's RecursiveCharacterTextSplitter with
chunk_size=1000 characters and chunk_overlap=200, then collapses whitespace in each chunk.
The same splitter and settings are used here. Differences, because the corpus is different:
  * documents are web articles with no title; the source site is used as the snippet title;
  * exact duplicate documents (same text after stripping) are removed before chunking.

The dataset card states no licence, so chunks are written outside the repository (git-ignored
data_local/ by default). Cite Ghassabi et al. 2025, arXiv:2505.16000.

Output: jsonl.gz of {"id", "title", "content"} (the format scripts/retrieve.py reads) and a
stats JSON for the README.

Usage: python scripts/persian/chunk_corpus.py --out data_local/fa_corpus_chunks.jsonl.gz \
           --stats results/persian/fa_corpus_chunk_stats.json
"""
import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import re
import statistics

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", default=None, help="dataset commit; default: current head (recorded in stats)")
    p.add_argument("--local_dir", default=None, help="already downloaded copy (folder with data/*.parquet)")
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--chunk_overlap", type=int, default=200)
    p.add_argument("--out", default=os.path.join(ROOT, "data_local", "fa_corpus_chunks.jsonl.gz"))
    p.add_argument("--stats", default=os.path.join(ROOT, "results", "persian", "fa_corpus_chunk_stats.json"))
    a = p.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    repo = "gaokerena/medical_corpus"
    revision = a.revision or HfApi().dataset_info(repo).sha
    path = a.local_dir or snapshot_download(repo, repo_type="dataset", revision=revision)
    splitter = RecursiveCharacterTextSplitter(chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap)

    seen, n_docs, n_dupes, n_empty = set(), 0, 0, 0
    lengths, per_source = [], collections.Counter()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with gzip.open(a.out, "wt", encoding="utf-8") as out:
        doc_idx = 0
        for f in sorted(glob.glob(os.path.join(path, "data", "*.parquet"))):
            for row in pq.read_table(f).to_pylist():
                n_docs += 1
                text = (row["content"] or "").strip()
                if len(text) < 20:
                    n_empty += 1
                    continue
                h = hashlib.md5(text.encode("utf-8")).hexdigest()
                if h in seen:
                    n_dupes += 1
                    continue
                seen.add(h)
                source = re.sub(r"^https?://(www\.)?", "", row["source"] or "").strip("/")
                for i, chunk in enumerate(splitter.split_text(text)):
                    chunk = re.sub(r"\s+", " ", chunk).strip()
                    if not chunk:
                        continue
                    out.write(json.dumps({"id": f"fa{doc_idx:06d}_{i}", "title": source, "content": chunk},
                                         ensure_ascii=False) + "\n")
                    lengths.append(len(chunk))
                    per_source[source] += 1
                doc_idx += 1

    stats = {"dataset": repo, "revision": revision, "splitter": "langchain RecursiveCharacterTextSplitter",
             "chunk_size": a.chunk_size, "chunk_overlap": a.chunk_overlap, "documents": n_docs,
             "exact_duplicates_removed": n_dupes, "empty_removed": n_empty, "documents_kept": len(seen),
             "snippets": len(lengths), "chars_mean": round(statistics.mean(lengths)),
             "chars_median": statistics.median(lengths), "snippets_per_source": dict(per_source.most_common())}
    os.makedirs(os.path.dirname(os.path.abspath(a.stats)), exist_ok=True)
    json.dump(stats, open(a.stats, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
