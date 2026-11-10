# Kaggle kernel: does fixing the reasoning language help on Persian questions? (finding: Qwen reasons
# in English or Chinese, almost never Persian, on Persian questions.)
#   Persian FA_MED_MMLU questions as is, MedRAG prompts + one added system sentence (below);
#   CoT, and BGE-M3/Textbooks k = 4 (same retrieved lists as the 2x2 cell "Persian question, Textbooks").
# Compare, question by question, with the same configs without the sentence (CoT 59.14%, Textbooks@4 47.75%).
CODE_STAMP = "en_think-v1"
EXPECTED_TB = {"snippets": 125847, "fingerprint": "7f9bfd3cdc0dba9c"}
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SUFFIX = " Write your step-by-step thinking in English, even if the question is in another language."
GEN_BUDGET_S = 2400      # generate.py stops before a new chunk after this; answers so far are kept
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
sf = os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")
stamp = open(sf).read().strip() if os.path.exists(sf) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
retrieved = glob.glob(os.path.join(ROOT, "**", "xling_rag_top32.jsonl*"), recursive=True)
assert retrieved, "retrieval file missing"
retrieved = retrieved[0]
P = lambda *x: os.path.join(ROOT, "scripts", *x)
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)
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
cmd = [sys.executable, P("generate.py"), "--model", MODEL, "--benchmark", f"{TMP}/fa_mmlu.json", "--retrieved", retrieved,
       "--corpus", f"{TMP}/tb.jsonl.gz", "--configs", "cot", "bge_tb@4", "--datasets", "fa_mmlu", "--chunk", "200",
       "--system_suffix", SUFFIX, "--tag", "en_think", "--time_budget_s", str(GEN_BUDGET_S), "--out", f"{OUT}/results"]
r = subprocess.run(cmd, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
print(f"RESULT generate rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
