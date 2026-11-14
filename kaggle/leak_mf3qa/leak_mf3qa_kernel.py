# Kaggle kernel (CPU only, no GPU quota): is FA_MED_MMLU or KOPP inside MF3QA, the question-answer
# set Gaokerena was instruction-tuned on? Same n-gram check as RQ4 (scripts/persian/leakage_check.py),
# each MF3QA row = Question + Answer, with a planted positive control. Writes counts and ids only.
CODE_STAMP = "leak_mf3qa-v1"
import glob, json, os, subprocess, sys
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-3000:], flush=True); return r.returncode
ROOT = os.path.dirname(os.path.dirname([os.path.dirname(p) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)][0]))
sf = os.path.join(ROOT, "scripts", "persian", "CODE_STAMP")
stamp = open(sf).read().strip() if os.path.exists(sf) else None
print("code root:", ROOT, "stamp:", stamp, flush=True)
assert stamp == CODE_STAMP, f"stale code dataset (stamp {stamp!r}, expected {CODE_STAMP!r})"
P = lambda *x: os.path.join(ROOT, "scripts", *x)
TMP, OUT = "/kaggle/tmp", "/kaggle/working"
os.makedirs(TMP, exist_ok=True)
sh("pip install -q langchain-text-splitters 2>&1 | tail -2")
assert sh(f"{sys.executable} {P('persian', 'prepare_fa_mmlu.py')} --benchmark {ROOT}/data/mirage_benchmark.json "
          f"--out {TMP}/fa_mmlu.json --report {TMP}/align.json | tail -2") == 0
from huggingface_hub import snapshot_download
mf = snapshot_download("gaokerena/MF3QA", repo_type="dataset", local_dir=f"{TMP}/mf3qa")
kopp = snapshot_download("gaokerena/KOPP", repo_type="dataset", local_dir=f"{TMP}/kopp")
from huggingface_hub import HfApi
rev = {r: HfApi().dataset_info(r).sha for r in ["gaokerena/MF3QA", "gaokerena/KOPP", "gaokerena/FA_MED_MMLU"]}
print("revisions", rev, flush=True)
os.makedirs(f"{OUT}/results", exist_ok=True)
json.dump(rev, open(f"{OUT}/results/revisions.json", "w"), indent=1)
for name, pat in [("train", "train-*.parquet"), ("all_splits", "*.parquet")]:
    rc = sh(f"{sys.executable} {P('persian', 'leakage_check.py')} --corpus_files '{mf}/data/{pat}' --text_fields Question,Answer "
            f"--plant 20 --kopp_dir {kopp} --fa_mmlu {TMP}/fa_mmlu.json --out {OUT}/results/leakage_mf3qa_{name}.json")
    print("RESULT", name, "rc", rc, flush=True)
