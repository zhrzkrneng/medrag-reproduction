"""Dense embeddings of the Persian corpus with BGE-M3 (BAAI/bge-m3, MIT licence), one GPU per shard.

BGE-M3 dense retrieval = the [CLS] vector of the last layer, L2-normalised; queries need no
instruction prefix (model card). Snippets are encoded from their content only (the title is the
source web site). fp16, length-sorted batches, max 512 tokens (snippets are <= 1,000 characters).

Quota guard: after --probe_batches batches the script projects the shard's total time; if it is
above --time_budget_s it stops and writes nothing but a note, so a slow run cannot eat the quota.

Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/persian/embed_bge_m3.py --chunks fa_corpus_chunks.jsonl.gz \
      --shard 0 --num_shards 2 --out_dir emb
"""
import argparse
import gzip
import json
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", required=True)
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--probe_batches", type=int, default=60)
    p.add_argument("--time_budget_s", type=float, default=9000)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    ids, texts = [], []
    with gzip.open(a.chunks, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % a.num_shards == a.shard:
                s = json.loads(line)
                ids.append(s["id"])
                texts.append(s["content"])
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float16 if a.device == "cuda" else torch.float32).to(a.device).eval()

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)  # longest first: OOM shows early
    emb = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    t0, done = time.time(), 0
    batches = [order[i:i + a.batch_size] for i in range(0, len(order), a.batch_size)]
    for b, idx in enumerate(batches):
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=a.max_length,
                  return_tensors="pt").to(a.device)
        with torch.no_grad():
            cls = model(**enc).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls.float(), dim=-1)
        emb[idx] = cls.cpu().numpy().astype(np.float16)
        done += len(idx)
        if b + 1 == a.probe_batches:
            # longest batches come first, so this projection is pessimistic
            rate = done / (time.time() - t0)
            projected = len(texts) / rate
            print(json.dumps({"shard": a.shard, "probe_rate_per_s": round(rate, 1),
                              "projected_shard_s_upper_bound": round(projected)}), flush=True)
            if projected > a.time_budget_s:
                print(f"STOP: projected {projected:.0f}s > budget {a.time_budget_s:.0f}s", flush=True)
                os.makedirs(a.out_dir, exist_ok=True)
                json.dump({"stopped": True, "projected_s": projected},
                          open(os.path.join(a.out_dir, f"shard{a.shard}_stopped.json"), "w"))
                return
        if (b + 1) % 500 == 0:
            print(json.dumps({"shard": a.shard, "done": done, "of": len(texts),
                              "elapsed_s": round(time.time() - t0)}), flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, f"shard{a.shard}.npy"), emb)
    json.dump(ids, open(os.path.join(a.out_dir, f"shard{a.shard}_ids.json"), "w"))
    print(json.dumps({"shard": a.shard, "n": len(texts), "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
