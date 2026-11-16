# Kaggle kernel: PMC-LLaMA-13B feasibility and speed on 2xT4 before the full run.
# Weights are float32 .bin (52 GB); fp16 weights alone are ~24 GiB of the 2x14.6 GiB.
# Same first 30 MedQA questions, CoT and rrf2@32 (context cut to 1,024 tokens, as in MedRAG).
N = 30
MODEL = "axiong/PMC_LLaMA_13B"
VARIANTS = [
    ("fp16_tp2_eager", ["--tp", "2", "--gpu_mem", "0.97", "--enforce_eager"]),
    # bitsandbytes was tried in v1: vLLM 0.33 no longer supports it
]
import os
os.environ["HF_HOME"] = "/kaggle/tmp/hf"   # not /kaggle/working (20 GB limit)
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
sh("df -h / /kaggle/working /kaggle/tmp /tmp 2>&1; free -g")
t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
from huggingface_hub import snapshot_download
from medrag_repro.retrieval import load_corpus
corp = snapshot_download("MedRAG/textbooks", repo_type="dataset", local_dir="/kaggle/tmp/textbooks")
with gzip.open("/kaggle/tmp/textbooks.jsonl.gz", "wt") as f:
    for s in load_corpus(os.path.join(corp, "chunk")):
        f.write(json.dumps(s) + "\n")
KEEP = re.compile(r"KV cache|concurrency|Traceback|Error|error|sec_per_q|progress|memory|No space", re.I)
summary = []
for name, extra in VARIANTS:
    out = f"/kaggle/working/pmc/{name}"
    cmd = [sys.executable, gen, "--model", MODEL, "--benchmark", bench, "--retrieved", retrieved,
           "--corpus", "/kaggle/tmp/textbooks.jsonl.gz", "--configs", "cot", "rrf2@32", "--datasets", "medqa",
           "--limit", str(N), "--chunk", str(N), "--out", out, *extra]
    env = dict(os.environ, VLLM_LOGGING_LEVEL="INFO")
    print(f"\n===== {name} =====", flush=True)
    t = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2400)
        rc, log = r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, log = "timeout", e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
    wall = time.time() - t
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "log.txt"), "w").write(log)
    print("\n".join(l[:300] for l in log.splitlines() if KEEP.search(l))[-6000:], flush=True)
    if rc != 0:
        print("--- tail ---\n" + "\n".join(l[:300] for l in log.splitlines()[-15:]), flush=True)
    timing = [json.loads(l) for l in log.splitlines() if l.startswith('{"model"')]
    stats = {}
    for f in glob.glob(f"{out}/*/medqa/*.jsonl"):
        rows = [json.loads(l) for l in open(f)]
        stats[os.path.basename(f)] = {"n": len(rows), "avg_prompt": round(sum(r["prompt_tokens"] for r in rows) / len(rows)),
                                      "avg_out": round(sum(r["output_tokens"] for r in rows) / len(rows))}
    summary.append({"variant": name, "rc": rc, "wall_sec_incl_load": round(wall),
                    "timing": [{k: x[k] for k in ("config", "sec_per_q", "hit_length_cap")} for x in timing], "tokens": stats})
    print("SUMMARY", json.dumps(summary[-1]), flush=True)
    sh("df -h /kaggle/tmp | tail -1")
for s in summary:
    print("SUMMARY", json.dumps(s))
json.dump(summary, open("/kaggle/working/pmc/summary.json", "w"), indent=1)
print(f"total kernel {time.time()-T0:.0f}s")
