"""Question-only dense retrieval with BGE-M3 over the shard embeddings from embed_bge_m3.py.

Queries are encoded like the snippets ([CLS], L2-normalised, no instruction prefix) and scored by
inner product (= cosine), exact search on GPU. Writes the top-k snippet ids and scores per question.

Benchmarks: MIRAGE-format files from prepare_fa_mmlu.py / prepare_kopp.py (text already cleaned
with medrag_repro.persian.clean_text).
"""
import argparse
import glob
import gzip
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb_dir", required=True)
    p.add_argument("--benchmarks", nargs="+", required=True, help="<file>:<dataset key> pairs")
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    ids, mats = [], []
    for f in sorted(glob.glob(os.path.join(a.emb_dir, "shard*_ids.json"))):
        ids += json.load(open(f))
        mats.append(np.load(f.replace("_ids.json", ".npy")))
    # fp16 on GPU; fp32 on CPU, where half-precision matrix products are slow or unsupported
    dtype = torch.float16 if a.device == "cuda" else torch.float32
    corpus = torch.from_numpy(np.concatenate(mats)).to(a.device, dtype)  # n x 1024
    print("corpus vectors", corpus.shape, flush=True)

    queries = []
    for spec in a.benchmarks:  # "<path>:<dataset key>", MIRAGE format (prepare_fa_mmlu.py, prepare_kopp.py)
        path, ds = spec.rsplit(":", 1)
        for qid, it in json.load(open(path))[ds].items():
            queries.append((ds, qid, it["question"]))

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16 if a.device == "cuda" else torch.float32).to(a.device).eval()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with gzip.open(a.out, "wt") as out:
        for s in range(0, len(queries), 64):
            batch = queries[s:s + 64]
            enc = tok([q[2] for q in batch], padding=True, truncation=True, max_length=a.max_length, return_tensors="pt").to(a.device)
            with torch.no_grad():
                q = torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0].float(), dim=-1).to(dtype)
                scores, top = torch.topk(q @ corpus.T, a.k, dim=1)
            for (ds, qid, _), sc, tp in zip(batch, scores.float().cpu().tolist(), top.cpu().tolist()):
                out.write(json.dumps({"dataset": ds, "qid": qid, "bge_m3": [ids[i] for i in tp],
                                      "bge_m3_scores": [round(x, 4) for x in sc]}) + "\n")
    print("queries", len(queries), flush=True)


if __name__ == "__main__":
    main()
