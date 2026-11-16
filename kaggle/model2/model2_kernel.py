# Kaggle kernel: a second model family for the Persian results (reviewers will ask whether the findings are
# Qwen-specific). Same prompts (MedRAG), corpus, chunks, top-32 lists and length budget as run_fa / rq2.
#   English MMLU-Med CoT (language gap on the same 1,089 items), Persian CoT and RRF-2@4 on FA_MED_MMLU and KOPP.
# MODEL = Llama-3.1-8B-Instruct from Meta's Kaggle Models copy (needs the licence accepted on kaggle.com once);
# fallback Qwen/Qwen3-8B (Apache-2.0, public on Hugging Face, run without thinking mode).
# Cost estimate from Qwen2.5's Persian speed: ~3 GPU-hours.
CODE_STAMP = "code-2026-11-16b"   # must match scripts/persian/CODE_STAMP in the attached code dataset
EXPECTED = {"snippets": 541926, "fingerprint": "399761caa137ba97"}
CORPUS_REVISION = "1765dbdc64f75742d019e5e80010c62db27c63a1"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # or "Qwen/Qwen3-8B"
KAGGLE_MODEL = "metaresearch/llama-3.1/transformers/8b-instruct"
KERNEL_BUDGET_S = int(10.0 * 3600)
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
stamp = open(os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")).read().strip() if os.path.exists(os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
model_path = None
if MODEL.startswith("meta-llama"):
    cs = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True)
          if glob.glob(os.path.join(os.path.dirname(c), "*.safetensors")) and "llama" in c.lower()]
    assert cs, "Llama weights not mounted: accept the licence on the Kaggle model page"
    model_path = cs[0]
    print("model dir:", model_path, sorted(os.listdir(model_path)), flush=True)
    tc = json.load(open(os.path.join(model_path, "tokenizer_config.json")))
    assert tc.get("chat_template"), "no chat template in the Kaggle copy"
    # compare with the (gated) Hugging Face repo's public metadata where it is readable; recorded, not required
    check = {}
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(MODEL, files_metadata=True)
        for f in info.siblings:
            if f.rfilename.endswith(".safetensors"):
                lp = os.path.join(model_path, f.rfilename)
                hh = hashlib.sha256()
                with open(lp, "rb") as fh:
                    for blk in iter(lambda: fh.read(1 << 24), b""):
                        hh.update(blk)
                check[f.rfilename] = os.path.exists(lp) and f.lfs is not None and hh.hexdigest() == f.lfs.sha256
        check["hf_revision"] = info.sha
    except Exception as e:  # gated repo metadata may be refused without a token
        check["error"] = repr(e)[:300]
    print("weights check", json.dumps(check), flush=True)
    os.makedirs(f"{OUT}/results", exist_ok=True)
    json.dump({"source": "kaggle:" + KAGGLE_MODEL, "check": check}, open(f"{OUT}/results/model_source.json", "w"), indent=1)
P = lambda *x: os.path.join(ROOT, "scripts", *x)
assert glob.glob(os.path.join(ROOT, "**", "fa_retrieval_top32.jsonl*"), recursive=True), "retrieval file missing"
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)
sh("pip install -q langchain-text-splitters 2>&1 | tail -2")
assert sh(f"{sys.executable} {P('persian', 'prepare_fa_mmlu.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--out {TMP}/fa_mmlu.json --report {TMP}/align.json | tail -2") == 0
assert sh(f"{sys.executable} {P('persian', 'prepare_kopp.py')} --out {TMP}/kopp.json") == 0
bench = {**json.load(open(f"{TMP}/fa_mmlu.json")), **json.load(open(f"{TMP}/kopp.json"))}
json.dump(bench, open(f"{TMP}/fa_benchmarks.json", "w"), ensure_ascii=False)
print({k: len(v) for k, v in bench.items()}, flush=True)
assert sh(f"{sys.executable} {P('persian', 'chunk_corpus.py')} --revision {CORPUS_REVISION} --out {TMP}/chunks.jsonl.gz "
          f"--stats {TMP}/chunk_stats.json | grep -E 'snippets\"'") == 0
h, n = hashlib.sha256(), 0
for line in gzip.open(f"{TMP}/chunks.jsonl.gz", "rt", encoding="utf-8"):
    s = json.loads(line); h.update(s["id"].encode()); h.update(hashlib.md5(s["content"].encode()).digest()); n += 1
print("snippets", n, "fingerprint", h.hexdigest()[:16], flush=True)
assert n == EXPECTED["snippets"] and h.hexdigest()[:16] == EXPECTED["fingerprint"], "chunks differ from the local ones"
# Kaggle decompresses .gz files in datasets, so the file may arrive as .jsonl
retrieved = glob.glob(os.path.join(ROOT, "**", "fa_retrieval_top32.jsonl*"), recursive=True)[0]
print("retrieved:", retrieved, flush=True)
t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
print(f"setup done at {time.time()-T0:.0f}s", flush=True)
STEPS = [("mmlu", None, ["cot"]), ("fa", f"{TMP}/fa_benchmarks.json", ["cot", "rrf2@4"])]
for name, benchfile, configs in STEPS:
    left = KERNEL_BUDGET_S - (time.time() - T0)
    if left < 600:
        print(json.dumps({"skipped": name, "reason": "kernel budget"}), flush=True)
        continue
    cmd = [sys.executable, P("generate.py"), "--model", MODEL, "--configs", *configs, "--chunk", "200",
           "--time_budget_s", str(int(left)), "--out", f"{OUT}/results"]
    if model_path:
        cmd += ["--model_path", model_path]
    if benchfile:
        cmd += ["--benchmark", benchfile, "--datasets", "fa_mmlu", "kopp", "--retrieved", retrieved,
                "--corpus", f"{TMP}/chunks.jsonl.gz"]
    else:
        cmd += ["--datasets", "mmlu"]
    r = subprocess.run(cmd, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
    print(f"RESULT {name} rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
