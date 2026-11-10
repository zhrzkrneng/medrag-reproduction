"""Machine-translate a Persian MIRAGE-format benchmark (question + options) into English with an LLM
(vLLM, greedy), so the English MedRAG pipeline can be run on it ("translate first" baseline).

The answer letters are kept. If the model output cannot be parsed, the item keeps its Persian text
and is counted in the report (it is not dropped, so all runs keep the same questions).

Usage: python scripts/persian/translate_questions.py --benchmark fa_mmlu.json --dataset fa_mmlu \
           --out fa_mmlu_mt.json --out_key fa_mmlu_mt --report mt_report.json
"""
import argparse
import json
import re
import time

SYSTEM = "You are a professional medical translator."
USER = ("Translate this Persian multiple-choice medical exam question and its options into English. "
        "Translate faithfully and keep medical terms precise; do not answer the question.\n"
        "Return only JSON of the form {{\"question\": \"...\", \"options\": {{\"A\": \"...\", \"B\": \"...\", ...}}}}.\n\n{item}")


def parse(text, letters):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    opts = d.get("options")
    if not isinstance(d.get("question"), str) or not isinstance(opts, dict) or sorted(opts) != sorted(letters):
        return None
    if not all(isinstance(v, str) for v in opts.values()):
        return None
    return {"question": d["question"].strip(), "options": {k: opts[k].strip() for k in letters}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out_key", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    from vllm import LLM, SamplingParams
    bench = json.load(open(a.benchmark))[a.dataset]
    qids = sorted(bench)[: a.limit]
    llm = LLM(a.model, dtype="half", tensor_parallel_size=a.tp, max_model_len=4096, seed=0)
    tok = llm.get_tokenizer()
    prompts = []
    for q in qids:
        item = json.dumps({"question": bench[q]["question"], "options": bench[q]["options"]}, ensure_ascii=False)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(item=item)}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    t = time.time()
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=1024), use_tqdm=False)
    secs = time.time() - t
    res, failed = {}, []
    for q, o in zip(qids, outs):
        d = parse(o.outputs[0].text, list(bench[q]["options"]))
        if d is None:
            failed.append(q)
            d = {"question": bench[q]["question"], "options": bench[q]["options"]}
        res[q] = {**d, "answer": bench[q]["answer"]}
    json.dump({a.out_key: res}, open(a.out, "w"), ensure_ascii=False)
    rep = {"n": len(qids), "parse_failures": len(failed), "failed_qids": failed, "seconds": round(secs, 1),
           "sec_per_q": round(secs / len(qids), 3), "model": a.model, "prompt": USER}
    json.dump(rep, open(a.report, "w"), indent=1)
    print({k: v for k, v in rep.items() if k not in ("failed_qids", "prompt")}, flush=True)


if __name__ == "__main__":
    main()
