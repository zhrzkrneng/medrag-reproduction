# Kaggle kernel, RQ2: RAG vs a Persian medical fine-tuned model, on the same Persian questions.
#   Gaokerena-V (gaokerena/gaokerena-v1.0: LoRA r = 8 on Aya-Expanse-8B, trained on gaokerena/medical_corpus
#   and MF3QA; Ghassabi et al. 2025, arXiv:2505.16000) and its base model Aya-Expanse-8B as the control.
#   Configs: CoT, and RRF-2 (BM25 + BGE-M3) over the Persian corpus at k = 4, same lists and prompts as run_fa.
#   The adapter is merged into the base on CPU (peft merge_and_unload) and the merged model is served by vLLM.
#   fp16 on T4 (no bf16): the pilot checks that the answers parse before the full run is paid for.
# Needs no secret: the base comes from Kaggle Models (verified against Hugging Face), the adapter repo is public.
CODE_STAMP = "code-2026-11-15c"
LIMIT = 50               # pilot: first 50 questions of each dataset; None for the full run
DATASETS = ["fa_mmlu", "kopp"]
CONFIGS = ["cot", "rrf2@4"]
GEN_BUDGET_S = 3 * 3600  # per model; generate.py stops before a new chunk after this
EXPECTED = {"snippets": 541926, "fingerprint": "399761caa137ba97"}
CORPUS_REVISION = "1765dbdc64f75742d019e5e80010c62db27c63a1"
BASE, ADAPTER = "CohereLabs/aya-expanse-8b", "gaokerena/gaokerena-v1.0"
import glob, gzip, hashlib, json, os, shutil, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
sf = os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")
stamp = open(sf).read().strip() if os.path.exists(sf) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
# Aya-Expanse-8B comes from Cohere's own Kaggle Models copy (cohereforai/aya-expanse/transformers/8b/5, attached
# below), because Kaggle does not hand user secrets to API-pushed kernels, so the gated Hugging Face repo is out of
# reach. Each file is checked against the Hugging Face repo's sizes and LFS sha256 (public metadata) before use.
def find_base():
    cs = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True)
          if glob.glob(os.path.join(os.path.dirname(c), "model-0000*-of-00004.safetensors"))]
    assert cs, "Aya-Expanse-8B (Kaggle model cohereforai/aya-expanse/transformers/8b) is not attached"
    return cs[0]
P = lambda *x: os.path.join(ROOT, "scripts", *x)
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)
sh("pip install -q langchain-text-splitters 2>&1 | tail -2")
assert sh(f"{sys.executable} {P('persian', 'prepare_fa_mmlu.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--out {TMP}/fa_mmlu.json --report {TMP}/align.json | tail -2") == 0
assert sh(f"{sys.executable} {P('persian', 'prepare_kopp.py')} --out {TMP}/kopp.json") == 0
bench = {**json.load(open(f"{TMP}/fa_mmlu.json")), **json.load(open(f"{TMP}/kopp.json"))}
json.dump(bench, open(f"{TMP}/fa_benchmarks.json", "w"), ensure_ascii=False)
assert sh(f"{sys.executable} {P('persian', 'chunk_corpus.py')} --revision {CORPUS_REVISION} --out {TMP}/chunks.jsonl.gz "
          f"--stats {TMP}/chunk_stats.json | grep -E 'snippets\"'") == 0
h, n = hashlib.sha256(), 0
for line in gzip.open(f"{TMP}/chunks.jsonl.gz", "rt", encoding="utf-8"):
    s = json.loads(line); h.update(s["id"].encode()); h.update(hashlib.md5(s["content"].encode()).digest()); n += 1
print("snippets", n, "fingerprint", h.hexdigest()[:16], flush=True)
assert n == EXPECTED["snippets"] and h.hexdigest()[:16] == EXPECTED["fingerprint"], "chunks differ from the local ones"
retrieved = glob.glob(os.path.join(ROOT, "**", "fa_retrieval_top32.jsonl*"), recursive=True)[0]

from huggingface_hub import HfApi, snapshot_download
rev = {r: HfApi().model_info(r).sha for r in [BASE, ADAPTER]}
print("revisions", rev, flush=True)
base_dir = find_base()
print("base dir:", base_dir, sorted(os.listdir(base_dir)), flush=True)
# verify the Kaggle copy file by file against the Hugging Face repo
info = HfApi().model_info(BASE, revision=rev[BASE], files_metadata=True)
check = {}
for f in info.siblings:
    if not (f.rfilename.endswith(".safetensors") or f.rfilename in ("tokenizer.json", "config.json", "tokenizer_config.json")):
        continue
    lp = os.path.join(base_dir, f.rfilename)
    assert os.path.exists(lp), f"missing in the Kaggle copy: {f.rfilename}"
    size_ok = os.path.getsize(lp) == f.size
    want = f.lfs.sha256 if f.lfs else None
    got = None
    if want:
        hh = hashlib.sha256()
        with open(lp, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""):
                hh.update(b)
        got = hh.hexdigest()
    else:  # small git file: compare the git blob sha1
        data = open(lp, "rb").read()
        got = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        want = f.blob_id
    check[f.rfilename] = {"size_ok": size_ok, "hash_ok": got == want}
print("base verification", json.dumps(check), flush=True)
assert all(v["size_ok"] and v["hash_ok"] for k, v in check.items() if k.endswith(".safetensors")), "Kaggle weights differ from Hugging Face"
os.makedirs(f"{OUT}/results", exist_ok=True)
json.dump({**rev, "base_source": "kaggle:cohereforai/aya-expanse/transformers/8b/5", "base_verification": check},
          open(f"{OUT}/results/model_revisions.json", "w"), indent=1)
print(f"verification done at {time.time()-T0:.0f}s", flush=True)
ad_dir = snapshot_download(ADAPTER, revision=rev[ADAPTER], local_dir=f"{TMP}/gaokerena")
print(f"downloads done at {time.time()-T0:.0f}s", flush=True)
# merge on CPU in fp16 (the dtype vLLM runs on T4); tokenizer from the base, as in the model card
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)
m = PeftModel.from_pretrained(m, ad_dir).merge_and_unload()
merged = f"{TMP}/gaokerena_merged"
m.save_pretrained(merged, safe_serialization=True)
AutoTokenizer.from_pretrained(base_dir).save_pretrained(merged)
del m
print(f"merge done at {time.time()-T0:.0f}s", flush=True)
t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
print(f"setup done at {time.time()-T0:.0f}s", flush=True)
for model, path in [(ADAPTER, merged), (BASE, base_dir)]:
    cmd = [sys.executable, P("generate.py"), "--model", model, "--model_path", path, "--benchmark", f"{TMP}/fa_benchmarks.json",
           "--retrieved", retrieved, "--corpus", f"{TMP}/chunks.jsonl.gz", "--configs", *CONFIGS, "--datasets", *DATASETS,
           "--chunk", "200", "--time_budget_s", str(GEN_BUDGET_S), "--out", f"{OUT}/results"]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    r = subprocess.run(cmd, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
    print(f"RESULT {model} rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
