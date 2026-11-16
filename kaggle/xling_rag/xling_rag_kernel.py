# Kaggle kernel: 2x2 RAG accuracy, question language x corpus (RQ1), Qwen2.5-7B-Instruct, k = 4.
#   questions: the same 1,089 MMLU-Med items in English (MIRAGE "mmlu") and Persian ("fa_mmlu");
#   corpora:   MedRAG Textbooks (English) and the Persian web corpus;
#   retriever: BGE-M3 for all four cells (the only retriever that works across languages).
# Retrieved lists come from results/persian/xling_rag_top32.jsonl.gz (computed earlier, ids only).
CODE_STAMP = "xling_rag-v1"
EXPECTED_FA = {"snippets": 541926, "fingerprint": "399761caa137ba97"}
EXPECTED_TB = {"snippets": 125847, "fingerprint": "7f9bfd3cdc0dba9c"}
CORPUS_REVISION = "1765dbdc64f75742d019e5e80010c62db27c63a1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONFIGS = ["bge_tb@4", "bge_fa@4"]      # Textbooks cells first (cheaper), then the Persian corpus
DATASETS = ["fa_mmlu", "mmlu"]
GEN_BUDGET_S = 9000                      # generate.py stops before a new chunk after this; answers are kept
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
sf = os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")
stamp = open(sf).read().strip() if os.path.exists(sf) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
retrieved = glob.glob(os.path.join(ROOT, "**", "xling_rag_top32.jsonl*"), recursive=True)   # .gz is decompressed by Kaggle
assert retrieved, "retrieval file missing"
retrieved = retrieved[0]; print("retrieved:", retrieved, flush=True)
P = lambda *x: os.path.join(ROOT, "scripts", *x)
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))
sh("pip install -q langchain-text-splitters 2>&1 | tail -2")

# questions: English MIRAGE MMLU-Med + aligned Persian FA_MED_MMLU
assert sh(f"{sys.executable} {P('persian', 'prepare_fa_mmlu.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--out {TMP}/fa_mmlu.json --report {TMP}/align.json | tail -2") == 0
bench = {"mmlu": json.load(open(f"{ROOT}/data/mirage_benchmark.json"))["mmlu"], **json.load(open(f"{TMP}/fa_mmlu.json"))}
json.dump(bench, open(f"{TMP}/bench.json", "w"), ensure_ascii=False)

# one corpus file with both corpora (ids do not collide: Textbooks "<book>_<i>", Persian "fa<doc>_<i>")
def fingerprint(rows):
    h, n = hashlib.sha256(), 0
    for s in rows:
        h.update(s["id"].encode()); h.update(hashlib.md5(s["content"].encode()).digest()); n += 1
    return n, h.hexdigest()[:16]
from huggingface_hub import snapshot_download
from medrag_repro.retrieval import load_corpus
tb = load_corpus(os.path.join(snapshot_download("MedRAG/textbooks", repo_type="dataset", local_dir=f"{TMP}/textbooks"), "chunk"))
print("textbooks", fingerprint(tb), flush=True)
assert fingerprint(tb) == (EXPECTED_TB["snippets"], EXPECTED_TB["fingerprint"])
assert sh(f"{sys.executable} {P('persian', 'chunk_corpus.py')} --revision {CORPUS_REVISION} --out {TMP}/fa.jsonl.gz "
          f"--stats {TMP}/chunk_stats.json | grep -E 'snippets\"'") == 0
fa = [json.loads(l) for l in gzip.open(f"{TMP}/fa.jsonl.gz", "rt", encoding="utf-8")]
print("persian", fingerprint(fa), flush=True)
assert fingerprint(fa) == (EXPECTED_FA["snippets"], EXPECTED_FA["fingerprint"])
with gzip.open(f"{TMP}/both.jsonl.gz", "wt", encoding="utf-8") as f:
    for s in tb + fa:
        f.write(json.dumps({"id": s["id"], "title": s["title"], "content": s["content"]}, ensure_ascii=False) + "\n")
del tb, fa

t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
print(f"setup done at {time.time()-T0:.0f}s", flush=True)
cmd = [sys.executable, P("generate.py"), "--model", MODEL, "--benchmark", f"{TMP}/bench.json", "--retrieved", retrieved,
       "--corpus", f"{TMP}/both.jsonl.gz", "--configs", *CONFIGS, "--datasets", *DATASETS, "--chunk", "200",
       "--time_budget_s", str(GEN_BUDGET_S), "--out", f"{OUT}/results"]
r = subprocess.run(cmd, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
print(f"RESULT generate rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
