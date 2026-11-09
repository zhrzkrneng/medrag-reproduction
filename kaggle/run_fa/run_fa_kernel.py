# Kaggle kernel: Persian MedRAG with Qwen2.5-7B-Instruct on FA_MED_MMLU (1,089) and KOPP (1,878):
#   CoT (no retrieval) and RAG with RRF-2 (BM25 + BGE-M3) over the Persian web corpus, k = 4.
# k = 4 instead of 32: Persian snippets take about twice as many Qwen tokens as English ones
# (k=4 prompt ~2,300 tokens, k=32 ~7,500), and k=32 does not fit this week's GPU quota.
# Prompts: MedRAG's English instructions with Persian question, options and snippets.
CODE_STAMP = "run_fa-v1"   # must match scripts/persian/CODE_STAMP in the attached code dataset
EXPECTED = {"snippets": 541926, "fingerprint": "399761caa137ba97"}
CORPUS_REVISION = "1765dbdc64f75742d019e5e80010c62db27c63a1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONFIGS = ["cot", "rrf2@4"]
DATASETS = ["fa_mmlu", "kopp"]
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
stamp = open(os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")).read().strip() if os.path.exists(os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
P = lambda *x: os.path.join(ROOT, "scripts", *x)
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
retrieved = glob.glob(os.path.join(ROOT, "**", "fa_retrieval_top32.jsonl.gz"), recursive=True)[0]
t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
print(f"setup done at {time.time()-T0:.0f}s", flush=True)
cmd = [sys.executable, P("generate.py"), "--model", MODEL, "--benchmark", f"{TMP}/fa_benchmarks.json",
       "--retrieved", retrieved, "--corpus", f"{TMP}/chunks.jsonl.gz", "--configs", *CONFIGS,
       "--datasets", *DATASETS, "--chunk", "200", "--out", f"{OUT}/results"]
r = subprocess.run(cmd, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
print(f"RESULT generate rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
