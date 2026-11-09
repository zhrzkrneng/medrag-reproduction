# Kaggle kernel: BGE-M3 dense index of the Persian corpus + question-only retrieval for
# FA_MED_MMLU and KOPP. The corpus is re-chunked here from the Hub (same revision and settings as
# locally) instead of being uploaded, and checked against the local fingerprint before use.
EXPECTED = {"snippets": 541926, "fingerprint": "399761caa137ba97"}   # from the local chunk file
TIME_BUDGET_S = 9000      # per GPU shard; the embedder stops itself early if its projection is above this
CORPUS_REVISION = "1765dbdc64f75742d019e5e80010c62db27c63a1"
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /kaggle/working | tail -1")
roots = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)]
SRC = roots[0]; ROOT = os.path.dirname(SRC); print("code root:", ROOT, flush=True)
P = lambda *x: os.path.join(ROOT, "scripts", "persian", *x)
sh("pip install -q langchain-text-splitters 2>&1 | tail -2")
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)
assert sh(f"{sys.executable} {P('prepare_fa_mmlu.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--out {TMP}/fa_mmlu.json --report {OUT}/fa_mmlu_alignment.json | tail -4") == 0
assert sh(f"{sys.executable} {P('chunk_corpus.py')} --revision {CORPUS_REVISION} --out {TMP}/chunks.jsonl.gz "
          f"--stats {OUT}/chunk_stats.json | grep -E 'snippets\"|documents_kept'") == 0
h, n = hashlib.sha256(), 0
for line in gzip.open(f"{TMP}/chunks.jsonl.gz", "rt", encoding="utf-8"):
    s = json.loads(line); h.update(s["id"].encode()); h.update(hashlib.md5(s["content"].encode()).digest()); n += 1
print("snippets", n, "fingerprint", h.hexdigest()[:16], "expected", EXPECTED, flush=True)
assert n == EXPECTED["snippets"] and h.hexdigest()[:16] == EXPECTED["fingerprint"], "chunks differ from the local ones"
print(f"setup done at {time.time()-T0:.0f}s", flush=True)

# one embedding process per GPU, output streamed live
procs = []
for g in (0, 1):
    cmd = [sys.executable, P("embed_bge_m3.py"), "--chunks", f"{TMP}/chunks.jsonl.gz", "--shard", str(g),
           "--num_shards", "2", "--out_dir", f"{OUT}/emb", "--time_budget_s", str(TIME_BUDGET_S)]
    procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
rcs = [p.wait() for p in procs]
print(f"embedding rc={rcs} at {time.time()-T0:.0f}s", flush=True)
if glob.glob(f"{OUT}/emb/*stopped.json") or any(rcs):
    print("RESULT: embedding stopped or failed; no retrieval run", flush=True)
else:
    rc = subprocess.run([sys.executable, P("dense_search.py"), "--emb_dir", f"{OUT}/emb", "--fa_mmlu", f"{TMP}/fa_mmlu.json",
                         "--out", f"{OUT}/bge_m3_top100.jsonl.gz"]).returncode
    print(f"RESULT: search rc={rc}; total {time.time()-T0:.0f}s", flush=True)
