# Kaggle kernel: 2x2 cross-lingual retrieval with BGE-M3, to separate question language from corpus.
#   corpora: MedRAG Textbooks (English, embedded here) and the Persian web corpus (embeddings from
#            kernel medrag-repro-embed-fa, attached as an input);
#   queries: the same 1,089 MMLU-Med questions in English (MIRAGE) and Persian (FA_MED_MMLU).
# Textbooks are downloaded from the Hub and checked against the local fingerprint.
EXPECTED = {"snippets": 125847, "fingerprint": "7f9bfd3cdc0dba9c"}
TIME_BUDGET_S = 2400   # per GPU shard; the embedder stops itself if its projection is above this
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
sys.path.insert(0, os.path.join(ROOT, "src")); print("code root:", ROOT, flush=True)
P = lambda *x: os.path.join(ROOT, "scripts", "persian", *x)
fa_emb = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/shard0_ids.json", recursive=True)]
print("Persian embeddings:", fa_emb, flush=True)
assert fa_emb, "attach the output of medrag-repro-embed-fa"
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)

from huggingface_hub import snapshot_download
from medrag_repro.retrieval import load_corpus
corp = snapshot_download("MedRAG/textbooks", repo_type="dataset", local_dir=f"{TMP}/textbooks")
h, n = hashlib.sha256(), 0
with gzip.open(f"{TMP}/textbooks.jsonl.gz", "wt") as f:
    for s in load_corpus(os.path.join(corp, "chunk")):
        h.update(s["id"].encode()); h.update(hashlib.md5(s["content"].encode()).digest()); n += 1
        f.write(json.dumps(s) + "\n")
print("textbooks", n, h.hexdigest()[:16], "expected", EXPECTED, flush=True)
assert n == EXPECTED["snippets"] and h.hexdigest()[:16] == EXPECTED["fingerprint"], "Textbooks differ from the local copy"
assert sh(f"{sys.executable} {P('prepare_fa_mmlu.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--out {TMP}/fa_mmlu.json --report {TMP}/align.json | tail -3") == 0

procs = []
for g in (0, 1):
    cmd = [sys.executable, P("embed_bge_m3.py"), "--chunks", f"{TMP}/textbooks.jsonl.gz", "--shard", str(g),
           "--num_shards", "2", "--out_dir", f"{OUT}/emb_textbooks", "--time_budget_s", str(TIME_BUDGET_S)]
    procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
rcs = [p.wait() for p in procs]
print(f"embedding rc={rcs} at {time.time()-T0:.0f}s", flush=True)
if glob.glob(f"{OUT}/emb_textbooks/*stopped.json") or any(rcs):
    print("RESULT: embedding stopped or failed; no retrieval run", flush=True)
else:
    en_q, fa_q = f"{ROOT}/data/mirage_benchmark.json:mmlu", f"{TMP}/fa_mmlu.json:fa_mmlu"
    runs = [("textbooks", f"{OUT}/emb_textbooks", [en_q, fa_q]),   # cells 1 and 2
            ("fa_web", fa_emb[0], [en_q])]                          # cell 4 (cell 3 exists already)
    for name, emb, qs in runs:
        rc = subprocess.run([sys.executable, P("dense_search.py"), "--emb_dir", emb, "--benchmarks", *qs,
                             "--out", f"{OUT}/xling_{name}_top100.jsonl.gz"]).returncode
        print(f"search {name} rc={rc}", flush=True)
    print(f"RESULT: done; total {time.time()-T0:.0f}s", flush=True)
