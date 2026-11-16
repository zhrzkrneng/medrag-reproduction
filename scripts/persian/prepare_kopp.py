"""KOPP (Iranian medical licensing exam questions, gaokerena/KOPP) in MIRAGE's format.

Splits sept23 (168) and other_years (1,710); the repository's data/train-* file duplicates
sept23 and is not in its configs, so it is skipped. Question ids: kopp/<split>/<row>.
"Correct answer" is 1-4 -> A-D. Text is cleaned with medrag_repro.persian.clean_text
(Unicode bidi marks from PDF copies; 184 of 1,878 questions have them). No licence is stated, so the output goes to data_local/.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from medrag_repro.persian import clean_text  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--revision", default="0438720931439c00b2565e92f00e4db57aea99ae")
    p.add_argument("--out", default=os.path.join(ROOT, "data_local", "kopp_benchmark.json"))
    a = p.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    path = snapshot_download("gaokerena/KOPP", repo_type="dataset", revision=a.revision)
    out = {}
    for split in ["sept23", "other_years"]:
        rows = pq.read_table(glob.glob(os.path.join(path, "data", f"{split}-*.parquet"))[0]).to_pylist()
        for i, r in enumerate(rows):
            out[f"kopp/{split}/{i}"] = {"question": clean_text(r["Question"]),
                                        "options": {L: clean_text(r[f"Option{j}"]) for j, L in enumerate("ABCD", 1)},
                                        "answer": "ABCD"[int(r["Correct answer"]) - 1], "topic": r["Topic"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"kopp": out}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(len(out), "questions ->", a.out)


if __name__ == "__main__":
    main()
