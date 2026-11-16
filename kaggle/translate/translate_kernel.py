# Kaggle kernel: "translate first" baseline for Persian questions (RQ1), Qwen2.5-7B-Instruct.
#   1. Qwen translates the 1,089 FA_MED_MMLU questions and options into English (greedy);
#   2. BGE-M3 retrieves from English Textbooks with the translated question (Textbooks embeddings
#      come from kernel medrag-repro-xling, attached as an input);
#   3. CoT and RAG (k = 4) on the translated items, the same settings as the 2x2 cells.
# Compare with: English original (CoT 78.42%, BGE-M3/Textbooks@4 73.92%) and Persian
# (CoT 59.14%, Textbooks@4 47.75%, Persian corpus@4 52.98%).
CODE_STAMP = "translate-v1"
EXPECTED_TB = {"snippets": 125847, "fingerprint": "7f9bfd3cdc0dba9c"}
MODEL = "Qwen/Qwen2.5-7B-Instruct"
GEN_BUDGET_S = 3600      # generate.py stops before a new chunk after this; answers so far are kept
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
sf = os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")
stamp = open(sf).read().strip() if os.path.exists(sf) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
emb = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/emb_textbooks/shard0_ids.json", recursive=True)]
assert emb, "attach the output of medrag-repro-xling (Textbooks BGE-M3 embeddings)"
emb = emb[0]; print("Textbooks embeddings:", emb, flush=True)
P = lambda *x: os.path.join(ROOT, "scripts", *x)
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True); os.makedirs(f"{OUT}/results", exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))
sh("pip install -q langchain-text-splitters 2>&1 | tail -2")
assert sh(f"{sys.executable} {P('persian', 'prepare_fa_mmlu.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--out {TMP}/fa_mmlu.json --report {TMP}/align.json | tail -2") == 0

from huggingface_hub import snapshot_download
from medrag_repro.retrieval import load_corpus
tb = load_corpus(os.path.join(snapshot_download("MedRAG/textbooks", repo_type="dataset", local_dir=f"{TMP}/textbooks"), "chunk"))
h = hashlib.sha256()
for s in tb:
    h.update(s["id"].encode()); h.update(hashlib.md5(s["content"].encode()).digest())
print("textbooks", len(tb), h.hexdigest()[:16], flush=True)
assert (len(tb), h.hexdigest()[:16]) == (EXPECTED_TB["snippets"], EXPECTED_TB["fingerprint"])
with gzip.open(f"{TMP}/tb.jsonl.gz", "wt") as f:
    for s in tb:
        f.write(json.dumps({"id": s["id"], "title": s["title"], "content": s["content"]}) + "\n")
del tb

t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
print(f"setup done at {time.time()-T0:.0f}s", flush=True)
# 1. translation (own process, so its GPU memory is released before retrieval)
rc = subprocess.run([sys.executable, P("persian", "translate_questions.py"), "--benchmark", f"{TMP}/fa_mmlu.json",
                     "--dataset", "fa_mmlu", "--out", f"{OUT}/fa_mmlu_mt.json", "--out_key", "fa_mmlu_mt",
                     "--report", f"{OUT}/results/mt_report.json"], env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")).returncode
print(f"translation rc={rc} at {time.time()-T0:.0f}s", flush=True)
assert rc == 0
# 2. retrieval with the translated question
rc = subprocess.run([sys.executable, P("persian", "dense_search.py"), "--emb_dir", emb,
                     "--benchmarks", f"{OUT}/fa_mmlu_mt.json:fa_mmlu_mt", "--k", "32", "--out", f"{TMP}/mt_dense.jsonl.gz"]).returncode
print(f"search rc={rc} at {time.time()-T0:.0f}s", flush=True)
assert rc == 0
with gzip.open(f"{TMP}/mt_dense.jsonl.gz", "rt") as f, open(f"{OUT}/results/mt_retrieved.jsonl", "w") as g:
    for r in map(json.loads, f):
        g.write(json.dumps({"dataset": r["dataset"], "qid": r["qid"], "bge_tb": r["bge_m3"]}) + "\n")
# 3. generation
cmd = [sys.executable, P("generate.py"), "--model", MODEL, "--benchmark", f"{OUT}/fa_mmlu_mt.json",
       "--retrieved", f"{OUT}/results/mt_retrieved.jsonl", "--corpus", f"{TMP}/tb.jsonl.gz",
       "--configs", "cot", "bge_tb@4", "--datasets", "fa_mmlu_mt", "--chunk", "200",
       "--time_budget_s", str(GEN_BUDGET_S), "--out", f"{OUT}/results"]
r = subprocess.run(cmd, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
print(f"RESULT generate rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
