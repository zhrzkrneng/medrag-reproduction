# Kaggle kernel: speed diagnosis. The fp16 tensor-parallel baseline ran at 13.3 s/question
# (speed test, 100 MedQA questions, rrf2@32). Each variant answers the SAME first 20 MedQA
# questions with their real rrf2@32 contexts, in a separate process (frees GPU memory).
N = 20
FP16 = "Qwen/Qwen2.5-7B-Instruct"
AWQ = "Qwen/Qwen2.5-7B-Instruct-AWQ"
VARIANTS = [
    # name, model, extra args, extra env
    ("fp16_tp2_flashinfer", FP16, ["--tp", "2"], {"VLLM_ATTENTION_BACKEND": "FLASHINFER"}),
    ("fp16_pp2", FP16, ["--tp", "1", "--pp", "2"], {}),          # no all-reduce between GPUs
    ("awq_tp1_one_gpu", AWQ, ["--tp", "1"], {"CUDA_VISIBLE_DEVICES": "0"}),  # x2 if one copy per GPU
]
import glob, gzip, json, os, re, subprocess, sys, time
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
KEEP = re.compile(r"backend|KV cache|concurrency|Traceback|Error|error|sec_per_q|progress|quantiz", re.I)
summary = []
for name, model, extra, env_extra in VARIANTS:
    out = f"/kaggle/working/diag/{name}"
    cmd = [sys.executable, gen, "--model", model, "--benchmark", bench, "--retrieved", retrieved,
           "--corpus", "/kaggle/tmp/textbooks.jsonl.gz", "--configs", "rrf2@32", "--datasets", "medqa",
           "--limit", str(N), "--chunk", str(N), "--out", out, *extra]
    env = dict(os.environ, VLLM_LOGGING_LEVEL="INFO", **env_extra)
    print(f"\n===== {name} =====", flush=True)
    t = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
        rc, log = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, log = "timeout", (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
    wall = time.time() - t
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "log.txt"), "w").write(log)
    lines = [l[:300] for l in log.splitlines() if KEEP.search(l)]
    print("\n".join(lines[-25:]), flush=True)
    if rc != 0:
        print("--- tail ---\n" + "\n".join(l[:300] for l in log.splitlines()[-15:]), flush=True)
    timing = [json.loads(l) for l in log.splitlines() if l.startswith('{"model"')]
    summary.append({"variant": name, "rc": rc, "wall_sec_incl_load": round(wall),
                    "sec_per_q": timing[0]["sec_per_q"] if timing else None})
    print("SUMMARY", json.dumps(summary[-1]), flush=True)
print("\nBASELINE fp16_tp2 (speed test): 13.3 s/q")
for s in summary:
    print("SUMMARY", json.dumps(s))
json.dump(summary, open("/kaggle/working/diag/summary.json", "w"), indent=1)
print(f"total kernel {time.time()-T0:.0f}s")
