# Kaggle kernel: MedRAG generation with vLLM. Edit MODEL / CONFIGS / LIMIT below for each run.
MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONFIGS = ["cot", "rrf2@32", "bm25@32", "medcpt@32"]
LIMIT = None        # questions per dataset; None = all
import glob, gzip, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-1500:], flush=True); return r.returncode
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
roots = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)]
if not roots:
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        subprocess.run(["unzip", "-oq", z, "-d", "/kaggle/working/code"], check=True)
    roots = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/working/code/**/medrag_repro/retrieval.py", recursive=True)]
SRC = roots[0]; ROOT = os.path.dirname(SRC); print("src:", SRC, "dataset root:", ROOT); sys.path.insert(0, SRC)
print(sorted(glob.glob(os.path.join(ROOT, "**", "*"), recursive=True))[:60])
retrieved = glob.glob(os.path.join(ROOT, "**", "textbooks_top32.jsonl*"), recursive=True)[0]
bench = glob.glob(os.path.join(ROOT, "**", "mirage_benchmark.json"), recursive=True)[0]
gen = glob.glob(os.path.join(ROOT, "**", "scripts", "generate.py"), recursive=True)[0]
t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
from huggingface_hub import snapshot_download
from medrag_repro.retrieval import load_corpus
corp = snapshot_download("MedRAG/textbooks", repo_type="dataset", local_dir="/kaggle/tmp/textbooks")
with gzip.open("/kaggle/tmp/textbooks.jsonl.gz", "wt") as f:
    for s in load_corpus(os.path.join(corp, "chunk")):
        f.write(json.dumps(s) + "\n")
cmd = [sys.executable, gen, "--model", MODEL, "--benchmark", bench, "--retrieved", retrieved,
       "--corpus", "/kaggle/tmp/textbooks.jsonl.gz", "--configs", *CONFIGS, "--out", "/kaggle/working/results"]
if LIMIT:
    cmd += ["--limit", str(LIMIT)]
# stream output live (so progress is visible while running); keep vLLM's own logging quiet
env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")  # no TQDM_DISABLE: it makes vLLM divide by zero
t = time.time(); r = subprocess.run(cmd, env=env)
print(f"RESULT generate rc={r.returncode} {time.time()-t:.0f}s; total kernel {time.time()-T0:.0f}s", flush=True)
print(open("/kaggle/working/results/timing.jsonl").read() if os.path.exists("/kaggle/working/results/timing.jsonl") else "no timing file")
