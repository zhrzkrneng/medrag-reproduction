"""Answer MIRAGE questions with an open LLM (vLLM), with or without retrieved snippets.

Faithful to MedRAG.medrag_answer (src/medrag.py at commit 7599a72):
  * snippets formatted as 'Document [i] (Title: t) content', joined by '\n',
    tokenised WITHOUT special tokens and truncated to ``context_length`` tokens;
  * greedy decoding (temperature 0) and the same system/user prompts;
  * total length (prompt + answer) bounded by ``max_length``.
Documented deviation: the answer is additionally capped at ``--max_new_tokens``
(default 1024) to bound GPU time; the number of hits on the cap is logged.

Usage (on a GPU machine):
  python scripts/generate.py --model Qwen/Qwen2.5-7B-Instruct \
      --retrieved retrieved/textbooks_top32.jsonl.gz --corpus corpus/textbooks.jsonl.gz \
      --configs cot rrf2@32 bm25@32 medcpt@32 rrf2@1 rrf2@4 --out results
"""
import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medrag_repro.templates import build_messages, format_snippet  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Length budgets copied from MedRAG.__init__; Qwen2.5 is not in MedRAG, so it gets the
# Llama-3 budget (8192 / 7168), the closest model MedRAG supports.
MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": {"max_length": 8192, "context_length": 7168, "template": None},
    "axiong/PMC_LLaMA_13B": {"max_length": 2048, "context_length": 1024,
                             "template": os.path.join(ROOT, "templates", "pmc_llama.jinja")},
}


def jopen(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--benchmark", default=os.path.join(ROOT, "data", "mirage_benchmark.json"))
    p.add_argument("--retrieved", help="output of scripts/retrieve.py")
    p.add_argument("--corpus", help="jsonl(.gz) of {id,title,content}")
    p.add_argument("--datasets", nargs="+", default=["medqa", "mmlu"])
    p.add_argument("--configs", nargs="+", default=["cot"], help="'cot' or '<retriever>@<k>', e.g. rrf2@32")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--tp", type=int, default=2, help="tensor parallel size (2 for Kaggle T4 x2)")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cfg = MODELS[a.model]
    template = None
    if cfg["template"]:
        # MedRAG loads its jinja files with .replace('    ', '').replace('\n', '')
        template = open(cfg["template"]).read().replace("    ", "").replace("\n", "")
    llm = LLM(a.model, dtype="half", tensor_parallel_size=a.tp, max_model_len=cfg["max_length"],
              gpu_memory_utilization=0.92, seed=0)
    tok = llm.get_tokenizer()
    if template:
        tok.chat_template = template

    bench = json.load(open(a.benchmark))
    retrieved, corpus = {}, {}
    if any(c != "cot" for c in a.configs):
        with jopen(a.retrieved) as f:
            for line in f:
                r = json.loads(line)
                retrieved[(r["dataset"], r["qid"])] = r
        with jopen(a.corpus) as f:
            for line in f:
                s = json.loads(line)
                corpus[s["id"]] = s

    slug = a.model.replace("/", "__")
    for conf in a.configs:
        for ds in a.datasets:
            out_path = os.path.join(a.out, slug, ds, f"{conf.replace('@', '_k')}.jsonl")
            if os.path.exists(out_path):
                print("skip (exists)", out_path, flush=True)
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            qids = sorted(bench[ds])[: a.limit]
            prompts, params, meta = [], [], []
            for qid in qids:
                item = bench[ds][qid]
                context = None
                if conf != "cot":
                    retr, k = conf.split("@")
                    snips = [corpus[i] for i in retrieved[(ds, qid)][retr][: int(k)]]
                    joined = "\n".join(format_snippet(j, s) for j, s in enumerate(snips))
                    ctx_ids = tok.encode(joined, add_special_tokens=False)[: cfg["context_length"]]
                    context = tok.decode(ctx_ids)
                msgs = build_messages(item["question"], item["options"], context)
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ids = tok.encode(text, add_special_tokens=False)
                budget = max(1, min(a.max_new_tokens, cfg["max_length"] - len(ids)))
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                params.append(SamplingParams(temperature=0.0, max_tokens=budget))
                meta.append((qid, item["answer"], len(ids), budget))
            t = time.time()
            outs = llm.generate(prompts, params)
            dt = time.time() - t
            capped = 0
            with open(out_path, "w") as f:
                for (qid, gold, n_in, budget), o in zip(meta, outs):
                    n_out = len(o.outputs[0].token_ids)
                    capped += int(n_out >= budget)
                    f.write(json.dumps({"qid": qid, "answer": gold, "response": o.outputs[0].text,
                                        "prompt_tokens": n_in, "output_tokens": n_out}) + "\n")
            msg = {"model": a.model, "config": conf, "dataset": ds, "n": len(qids), "seconds": round(dt, 1),
                   "sec_per_q": round(dt / len(qids), 3), "hit_length_cap": capped}
            print(json.dumps(msg), flush=True)
            # also persisted, because vLLM's own logging can push this line out of a truncated console log
            with open(os.path.join(a.out, "timing.jsonl"), "a") as tf:
                tf.write(json.dumps(msg) + "\n")


if __name__ == "__main__":
    main()
