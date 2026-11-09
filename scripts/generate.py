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
    # official 4-bit AWQ build of the same model: fits on one T4, used only if fp16 is too slow (documented)
    "Qwen/Qwen2.5-7B-Instruct-AWQ": {"max_length": 8192, "context_length": 7168, "template": None},
    "axiong/PMC_LLaMA_13B": {"max_length": 2048, "context_length": 1024, "add_bos": True,
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
    p.add_argument("--chunk", type=int, default=200, help="questions per vLLM call; results appended after each")
    p.add_argument("--tp", type=int, default=2, help="tensor parallel size (2 for Kaggle T4 x2)")
    p.add_argument("--time_budget_s", type=float, default=None,
                   help="stop before starting a new chunk once this many seconds have passed (answers so far are kept)")
    p.add_argument("--gpu_mem", type=float, default=0.92, help="vLLM gpu_memory_utilization")
    p.add_argument("--enforce_eager", action="store_true", help="no CUDA graphs (saves GPU memory)")
    p.add_argument("--quantization", default=None, help="e.g. bitsandbytes (documented deviation)")
    p.add_argument("--pp", type=int, default=1, help="pipeline parallel size")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    T_START = time.time()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cfg = MODELS[a.model]
    template = None
    if cfg["template"]:
        # MedRAG loads its jinja files with .replace('    ', '').replace('\n', '')
        template = open(cfg["template"]).read().replace("    ", "").replace("\n", "")
    llm = LLM(a.model, dtype="half", tensor_parallel_size=a.tp, pipeline_parallel_size=a.pp, max_model_len=cfg["max_length"],
              gpu_memory_utilization=a.gpu_mem, enforce_eager=a.enforce_eager, quantization=a.quantization, seed=0)
    tok = llm.get_tokenizer()
    if template:
        # vLLM wraps its tokenizer, and setting chat_template on the wrapper does not reach
        # apply_chat_template; build prompts with the plain Hugging Face tokenizer instead
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
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
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            qids = sorted(bench[ds])[: a.limit]
            done = set()
            if os.path.exists(out_path):  # resume: keep answers already written
                with open(out_path) as f:
                    done = {json.loads(line)["qid"] for line in f if line.strip()}
            todo = [q for q in qids if q not in done]
            if not todo:
                print("skip (complete)", out_path, flush=True)
                continue
            t_all, capped = time.time(), 0
            # answered in chunks and appended after each one, so progress (and real speed) is visible
            for start in range(0, len(todo), a.chunk):
                if a.time_budget_s and time.time() - T_START > a.time_budget_s:
                    print(json.dumps({"stopped": "time budget", "config": conf, "dataset": ds,
                                      "answered": len(done) + start, "of": len(qids)}), flush=True)
                    return
                part = todo[start: start + a.chunk]
                prompts, params, meta = [], [], []
                for qid in part:
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
                    if cfg.get("add_bos") and tok.bos_token_id is not None:
                        # MedRAG runs LLaMA models through a HF pipeline, whose tokenizer adds <s>
                        ids = [tok.bos_token_id] + ids
                    budget = max(1, min(a.max_new_tokens, cfg["max_length"] - len(ids)))
                    prompts.append(TokensPrompt(prompt_token_ids=ids))
                    params.append(SamplingParams(temperature=0.0, max_tokens=budget))
                    meta.append((qid, item["answer"], len(ids), budget))
                t = time.time()
                outs = llm.generate(prompts, params, use_tqdm=False)
                dt = max(time.time() - t, 1e-6)
                n_in = n_out_all = 0
                with open(out_path, "a") as f:
                    for (qid, gold, n_prompt, budget), o in zip(meta, outs):
                        n_out = len(o.outputs[0].token_ids)
                        capped += int(n_out >= budget)
                        n_in += n_prompt
                        n_out_all += n_out
                        f.write(json.dumps({"qid": qid, "answer": gold, "response": o.outputs[0].text,
                                            "prompt_tokens": n_prompt, "output_tokens": n_out}) + "\n")
                print(json.dumps({"progress": f"{len(done) + start + len(part)}/{len(qids)}", "config": conf,
                                  "dataset": ds, "chunk_sec": round(dt, 1), "sec_per_q": round(dt / len(part), 3),
                                  "prompt_tok_per_sec": round(n_in / dt), "output_tok_per_sec": round(n_out_all / dt)}),
                      flush=True)
            dt = max(time.time() - t_all, 1e-6)
            msg = {"model": a.model, "config": conf, "dataset": ds, "n": len(todo), "seconds": round(dt, 1),
                   "sec_per_q": round(dt / len(todo), 3), "hit_length_cap": capped}
            print(json.dumps(msg), flush=True)
            # also persisted, because vLLM's own logging can push this line out of a truncated console log
            with open(os.path.join(a.out, "timing.jsonl"), "a") as tf:
                tf.write(json.dumps(msg) + "\n")

if __name__ == "__main__":
    main()
