# Kaggle kernel (model2_en): the English RAG cell for Llama-3.1-8B-Instruct that kaggle/model2 left out, so the
# English and Persian RAG results can be compared on the same 1,089 MMLU-Med items: RRF-2@4 over Textbooks (the
# committed top-32 lists, same as Qwen's English k = 4 run). ~0.4 GPU-hours.
# Original header of kaggle/model2 follows.
# Kaggle kernel: a second model family for the Persian results (reviewers will ask whether the findings are
# Qwen-specific). Same prompts (MedRAG), corpus, chunks, top-32 lists and length budget as run_fa / rq2.
#   English MMLU-Med CoT (language gap on the same 1,089 items), Persian CoT and RRF-2@4 on FA_MED_MMLU and KOPP.
# MODEL = Llama-3.1-8B-Instruct, weights from unsloth's ungated copy, checked file by file against Meta's SHA-256;
# fallback Qwen/Qwen3-8B (Apache-2.0, public on Hugging Face, run without thinking mode).
# Cost estimate from Qwen2.5's Persian speed: ~3 GPU-hours.
CODE_STAMP = "code-2026-11-16b"   # must match scripts/persian/CODE_STAMP in the attached code dataset
EXPECTED = {"snippets": 541926, "fingerprint": "399761caa137ba97"}
CORPUS_REVISION = "1765dbdc64f75742d019e5e80010c62db27c63a1"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # or "Qwen/Qwen3-8B"
MIRROR = "unsloth/Llama-3.1-8B-Instruct"
KERNEL_BUDGET_S = int(10.0 * 3600)
import glob, gzip, hashlib, json, os, subprocess, sys, time
T0 = time.time()
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-2500:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
stamp = open(os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")).read().strip() if os.path.exists(os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
model_path = None
if MODEL.startswith("meta-llama"):
    # Kaggle has paused access requests for Meta's Kaggle copy (2026-11-16) and the Hugging Face repo is gated, so the
    # weights come from unsloth's public, ungated copy (Llama 3.1 Community License allows redistribution). Each
    # safetensors file is checked against the SHA-256 that Meta's gated repo publishes in its (public) file metadata.
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    mirror_rev = api.model_info(MIRROR).sha
    model_path = snapshot_download(MIRROR, revision=mirror_rev, local_dir=f"{TMP}/llama",
                                   allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    print("mirror", MIRROR, mirror_rev, sorted(os.listdir(model_path)), flush=True)
    tc = json.load(open(os.path.join(model_path, "tokenizer_config.json")))
    assert tc.get("chat_template"), "no chat template in the mirror"
    check = {"mirror": MIRROR, "mirror_revision": mirror_rev}
    try:
        info = api.model_info(MODEL, files_metadata=True)
        check["official_revision"] = info.sha
        want = {f.rfilename: f.lfs.sha256 for f in info.siblings if f.rfilename.endswith(".safetensors") and f.lfs}
        got = {}
        for fn in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
            hh = hashlib.sha256()
            with open(fn, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 24), b""):
                    hh.update(blk)
            got[os.path.basename(fn)] = hh.hexdigest()
        check["files_match"] = {k: got.get(k) == v for k, v in want.items()}
        check["extra_files"] = sorted(set(got) - set(want))
        # small files: git blob sha1 against Meta's (recorded, not required: a mirror may rename _name_or_path etc.)
        blobs = {f.rfilename: f.blob_id for f in info.siblings if f.rfilename in
                 ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")}
        for fn, bid in blobs.items():
            lp = os.path.join(model_path, fn)
            if os.path.exists(lp):
                data = open(lp, "rb").read()
                check[f"same_{fn}"] = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == bid
    except Exception as e:  # gated metadata may be refused; then the run is still recorded as unverified
        check["error"] = repr(e)[:300]
    print("weights check", json.dumps(check), flush=True)
    os.makedirs(f"{OUT}/results", exist_ok=True)
    json.dump(check, open(f"{OUT}/results/model_source.json", "w"), indent=1)
    if "files_match" in check:
        assert want and all(check["files_match"].values()) and not check["extra_files"], "mirror weights differ from Meta's"
P = lambda *x: os.path.join(ROOT, "scripts", *x)
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)
# (the Persian data preparation of kaggle/model2 is not needed here)
t = time.time(); rc = sh("pip install -q vllm 2>&1 | tail -3"); print(f"vllm install rc={rc} {time.time()-t:.0f}s", flush=True)
print(f"setup done at {time.time()-T0:.0f}s", flush=True)
import gzip as _gz
from huggingface_hub import snapshot_download as _sd
sys.path.insert(0, os.path.join(ROOT, "src"))
from medrag_repro.retrieval import load_corpus
corp = _sd("MedRAG/textbooks", repo_type="dataset", local_dir=f"{TMP}/textbooks")
with _gz.open(f"{TMP}/textbooks.jsonl.gz", "wt") as f:
    for s_ in load_corpus(os.path.join(corp, "chunk")):
        f.write(json.dumps(s_) + "\n")
en_ret = glob.glob(os.path.join(ROOT, "**", "textbooks_top32.jsonl*"), recursive=True)[0]
cmd = [sys.executable, P("generate.py"), "--model", MODEL, "--model_path", model_path, "--configs", "rrf2@4",
       "--datasets", "mmlu", "--retrieved", en_ret, "--corpus", f"{TMP}/textbooks.jsonl.gz", "--chunk", "200",
       "--out", f"{OUT}/results"]
r = subprocess.run(cmd, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
print(f"RESULT en rrf2@4 rc={r.returncode}; total {time.time()-T0:.0f}s", flush=True)
p = f"{OUT}/results/timing.jsonl"
print(open(p).read() if os.path.exists(p) else "no timing file")
