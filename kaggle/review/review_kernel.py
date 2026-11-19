# Kaggle kernel: checks requested by a (self-)review of the paper. Qwen2.5-7B-Instruct, MedRAG prompts unless noted.
#   1. translate first on KOPP (no English original, so a gain cannot be recall of the source items): Qwen translates
#      the 1,878 questions and options into English, then CoT. Compare with Persian CoT on KOPP (51.65%).
#   2. option rotation on English MMLU-Med (A->B, ..., D->A, texts unchanged), CoT: a drop would point to recall of
#      MMLU items (question -> letter). Compare with English CoT (78.42%).
#   3. oracle retrieval: k = 4 with the snippets that contain the correct option's text put first (ids from
#      results/persian/oracle_retrieval.jsonl.gz for Persian; the same construction on Textbooks for English, built
#      here). Compare with CoT and RRF-2@4 on the same questions, in both languages.
#   4. MedRAG's prompts translated into Persian (--prompt_style medrag_fa): CoT and RRF-2@4 on FA_MED_MMLU.
# Estimated ~3.5 GPU-hours. Steps run in this order, each its own generate.py call, under a kernel-wide budget.
CODE_STAMP = "code-2026-11-18a"
EXPECTED = {"snippets": 541926, "fingerprint": "399761caa137ba97"}
EXPECTED_TB = {"snippets": 125847, "fingerprint": "7f9bfd3cdc0dba9c"}
CORPUS_REVISION = "1765dbdc64f75742d019e5e80010c62db27c63a1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
KERNEL_BUDGET_S = int(10.0 * 3600)
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
sf = os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")
stamp = open(sf).read().strip() if os.path.exists(sf) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
P = lambda *x: os.path.join(ROOT, "scripts", *x)
find = lambda name: glob.glob(os.path.join(ROOT, "**", name + "*"), recursive=True)[0]  # Kaggle may decompress .gz
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True); os.makedirs(f"{OUT}/results", exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))
ENV = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
def jopen(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")

# --- data -------------------------------------------------------------------------------------------------------
sh("pip install -q langchain-text-splitters 2>&1 | tail -2")
assert sh(f"{sys.executable} {P('persian', 'prepare_fa_mmlu.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--out {TMP}/fa_mmlu.json --report {TMP}/align.json | tail -2") == 0
assert sh(f"{sys.executable} {P('persian', 'prepare_kopp.py')} --out {TMP}/kopp.json") == 0
fa = {**json.load(open(f"{TMP}/fa_mmlu.json")), **json.load(open(f"{TMP}/kopp.json"))}
json.dump(fa, open(f"{TMP}/fa_benchmarks.json", "w"), ensure_ascii=False)
en = json.load(open(f"{ROOT}/data/mirage_benchmark.json"))
print({k: len(v) for k, v in fa.items()}, flush=True)
assert sh(f"{sys.executable} {P('persian', 'chunk_corpus.py')} --revision {CORPUS_REVISION} --out {TMP}/chunks.jsonl.gz "
          f"--stats {TMP}/chunk_stats.json | grep -E 'snippets\"'") == 0
h, n = hashlib.sha256(), 0
for line in gzip.open(f"{TMP}/chunks.jsonl.gz", "rt", encoding="utf-8"):
    s = json.loads(line); h.update(s["id"].encode()); h.update(hashlib.md5(s["content"].encode()).digest()); n += 1
print("snippets", n, "fingerprint", h.hexdigest()[:16], flush=True)
assert n == EXPECTED["snippets"] and h.hexdigest()[:16] == EXPECTED["fingerprint"], "chunks differ from the local ones"
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

# oracle lists and the matching question subsets
fa_or = f"{TMP}/oracle_fa.jsonl"
with jopen(find("oracle_retrieval.jsonl")) as f, open(fa_or, "w") as g:
    rows = [json.loads(l) for l in f if l.strip()]
    for r in rows:
        g.write(json.dumps(r) + "\n")
json.dump({ds: {r["qid"]: fa[ds][r["qid"]] for r in rows if r["dataset"] == ds} for ds in ("fa_mmlu", "kopp")},
          open(f"{TMP}/oracle_fa_bench.json", "w"), ensure_ascii=False)
tb_top = find("textbooks_top32.jsonl")
if not tb_top.endswith(".gz"):
    sh(f"gzip -c {tb_top} > {TMP}/textbooks_top32.jsonl.gz"); tb_top = f"{TMP}/textbooks_top32.jsonl.gz"
assert sh(f"{sys.executable} {P('persian', 'make_oracle_lists.py')} --retrieved {tb_top} --keys rrf2 bm25 medcpt "
          f"--corpus {TMP}/tb.jsonl.gz --benchmarks {ROOT}/data/mirage_benchmark.json --datasets mmlu "
          f"--out {OUT}/results/oracle_en.jsonl.gz") == 0
en_rows = [json.loads(l) for l in gzip.open(f"{OUT}/results/oracle_en.jsonl.gz", "rt")]
json.dump({"mmlu": {r["qid"]: en["mmlu"][r["qid"]] for r in en_rows}}, open(f"{TMP}/oracle_en_bench.json", "w"))
assert sh(f"{sys.executable} {P('persian', 'shuffle_options.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--dataset mmlu --out {TMP}/mmlu_rot.json --out_key mmlu_rot") == 0
fa_top = find("fa_retrieval_top32.jsonl")

t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
print(f"setup done at {time.time()-T0:.0f}s", flush=True)

def left():
    return KERNEL_BUDGET_S - (time.time() - T0)

def gen(name, bench, datasets, configs, retrieved=None, corpus=None, extra=()):
    if left() < 600:
        print(json.dumps({"skipped": name, "reason": "kernel budget"}), flush=True); return
    cmd = [sys.executable, P("generate.py"), "--model", MODEL, "--benchmark", bench, "--configs", *configs,
           "--datasets", *datasets, "--chunk", "100", "--time_budget_s", str(int(left())), "--out", f"{OUT}/results", *extra]
    if retrieved:
        cmd += ["--retrieved", retrieved, "--corpus", corpus]
    r = subprocess.run(cmd, env=ENV)
    print(f"RESULT {name} rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)

# 1. translate first on KOPP
rc = subprocess.run([sys.executable, P("persian", "translate_questions.py"), "--benchmark", f"{TMP}/kopp.json",
                     "--dataset", "kopp", "--out", f"{TMP}/kopp_mt.json", "--out_key", "kopp_mt",
                     "--report", f"{OUT}/results/kopp_mt_report.json"], env=ENV).returncode
print(f"translation rc={rc} at {time.time()-T0:.0f}s", flush=True)
if rc == 0:
    gen("kopp translate-first", f"{TMP}/kopp_mt.json", ["kopp_mt"], ["cot"])
# 3. oracle, Persian then English
gen("oracle fa", f"{TMP}/oracle_fa_bench.json", ["fa_mmlu", "kopp"], ["oracle@4"], fa_or, f"{TMP}/chunks.jsonl.gz")
gen("oracle en", f"{TMP}/oracle_en_bench.json", ["mmlu"], ["oracle@4"], f"{OUT}/results/oracle_en.jsonl.gz", f"{TMP}/tb.jsonl.gz")
# 2. option rotation
gen("mmlu rotated", f"{TMP}/mmlu_rot.json", ["mmlu_rot"], ["cot"])
# 4. Persian prompt
gen("persian prompt", f"{TMP}/fa_benchmarks.json", ["fa_mmlu"], ["cot", "rrf2@4"], fa_top, f"{TMP}/chunks.jsonl.gz",
    ["--prompt_style", "medrag_fa", "--tag", "fa_prompt"])
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
