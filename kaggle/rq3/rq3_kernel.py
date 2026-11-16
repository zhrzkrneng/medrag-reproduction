# Kaggle kernel, RQ3: do MedRAG's findings on retrievers and on k hold in Persian? Qwen2.5-7B-Instruct, MedRAG
# prompts, the same Persian corpus, chunks and top-32 lists as run_fa / run_fa32 (fa_retrieval_top32: bm25, bge_m3, rrf2).
#   1. retriever comparison at k = 4: BM25 on FA_MED_MMLU and KOPP, BGE-M3 on KOPP (BGE-M3 on FA_MED_MMLU already
#      exists: the 2x2 run used identical top-4 lists for all 1,089 questions, checked locally); RRF-2@4 exists (run_fa).
#   2. k sweep on FA_MED_MMLU: RRF-2 at k = 1, 2, 8, 16 (k = 4 and 32 exist), as in MedRAG's Figure on k.
# Cost estimate from earlier Persian runs (2.5-2.7 s/question at k = 4, English time x 2.7): ~9.5 GPU-hours.
# Steps run in priority order, each with its own generate.py call; a kernel-wide budget keeps it under Kaggle's 12 h.
CODE_STAMP = "code-2026-11-16a"   # code dataset version used for this run; must match scripts/persian/CODE_STAMP in the attached code dataset
EXPECTED = {"snippets": 541926, "fingerprint": "399761caa137ba97"}
CORPUS_REVISION = "1765dbdc64f75742d019e5e80010c62db27c63a1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
STEPS = [(["bm25@4"], ["fa_mmlu", "kopp"]), (["bge_m3@4"], ["kopp"]), (["rrf2@1", "rrf2@2", "rrf2@8", "rrf2@16"], ["fa_mmlu"])]
KERNEL_BUDGET_S = int(11.0 * 3600)   # no new chunk after 11 h of kernel time
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
stamp = open(os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")).read().strip() if os.path.exists(os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
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
for configs, datasets in STEPS:
    left = KERNEL_BUDGET_S - (time.time() - T0)
    if left < 600:
        print(json.dumps({"skipped": configs, "datasets": datasets, "reason": "kernel budget"}), flush=True)
        continue
    cmd = [sys.executable, P("generate.py"), "--model", MODEL, "--benchmark", f"{TMP}/fa_benchmarks.json",
           "--retrieved", retrieved, "--corpus", f"{TMP}/chunks.jsonl.gz", "--configs", *configs,
           "--datasets", *datasets, "--chunk", "100", "--time_budget_s", str(int(left)), "--out", f"{OUT}/results"]
    r = subprocess.run(cmd, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
    print(f"RESULT {configs} {datasets} rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
