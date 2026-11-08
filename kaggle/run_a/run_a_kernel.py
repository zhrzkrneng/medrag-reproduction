# Kaggle kernel A: CoT on MedQA + MMLU-Med, then RAG (RRF-2, k=32) on MedQA.
# (Kernel B: RRF-2 k=32 on MMLU-Med.) Measured speed: ~13.3 s per RAG question on 2xT4.
MODEL = "Qwen/Qwen2.5-7B-Instruct"
RUNS = [(["cot"], ["medqa", "mmlu"]), (["rrf2@32"], ["medqa"])]
CHUNK = 200
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
for configs, datasets in RUNS:
    cmd = [sys.executable, gen, "--model", MODEL, "--benchmark", bench, "--retrieved", retrieved,
           "--corpus", "/kaggle/tmp/textbooks.jsonl.gz", "--configs", *configs, "--datasets", *datasets,
           "--chunk", str(CHUNK), "--out", "/kaggle/working/results"]
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    t = time.time(); r = subprocess.run(cmd, env=env)
    print(f"RESULT {configs} {datasets} rc={r.returncode} {time.time()-t:.0f}s; kernel so far {time.time()-T0:.0f}s", flush=True)
p = "/kaggle/working/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
