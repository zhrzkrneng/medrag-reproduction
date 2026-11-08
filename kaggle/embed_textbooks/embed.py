# Kaggle kernel: embed the MedRAG Textbooks corpus with MedCPT-Article-Encoder (fp32, CLS pooling).
import glob, hashlib, json, os, subprocess, sys, time
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True, capture_output=True, text=True).stdout)
# locate the uploaded repo code (dataset may be mounted as dirs or zips)
roots = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/medrag_repro/retrieval.py", recursive=True)]
if not roots:
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        subprocess.run(["unzip", "-oq", z, "-d", "/kaggle/working/code"], check=True)
    roots = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/working/code/**/medrag_repro/retrieval.py", recursive=True)]
print("code root:", roots); sys.path.insert(0, roots[0])
from medrag_repro.retrieval import embed_corpus, load_corpus
from huggingface_hub import snapshot_download
t = time.time()
corp = snapshot_download("MedRAG/textbooks", repo_type="dataset", local_dir="/kaggle/tmp/textbooks")
snips = load_corpus(os.path.join(corp, "chunk"))
print(f"{len(snips)} snippets, download {time.time()-t:.0f}s", flush=True)
t = time.time()
emb = embed_corpus(snips, "/kaggle/working/medcpt_article_emb_textbooks.npy", "cuda", batch_size=64)
dt = time.time() - t
ids_sha = hashlib.sha256("\n".join(s["id"] for s in snips).encode()).hexdigest()
meta = {"shape": list(emb.shape), "dtype": str(emb.dtype), "seconds": round(dt), "ids_sha256": ids_sha,
        "first_id": snips[0]["id"], "last_id": snips[-1]["id"], "model": "ncbi/MedCPT-Article-Encoder"}
json.dump(meta, open("/kaggle/working/embedding_meta.json", "w"), indent=2)
print("RESULT", meta)
