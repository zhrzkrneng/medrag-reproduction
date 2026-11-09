"""Download FA_MED_MMLU (Persian translation of MMLU-Med), check it is aligned item by item with
MIRAGE's MMLU-Med, and write it in MIRAGE's format with the same question ids.

The dataset card (gaokerena/FA_MED_MMLU) states no licence, so the data is not committed to this
repository: this script writes it to data_local/ (git-ignored). Cite Ghassabi et al. 2025,
arXiv:2505.16000, when using it.

Text is cleaned with medrag_repro.persian.clean_text (Unicode bidi marks; 1 question has them).

Checks:
  * same subjects and item counts as MIRAGE MMLU-Med;
  * answer key: option<i> -> letter, compared item by item (random alignment would agree ~25%);
  * numbers in question + options survive translation (after mapping Persian digits);
    misses are reported (most are formatting, e.g. "2,300" -> "2300", or "2" written as a word).

Usage: python scripts/persian/prepare_fa_mmlu.py --out data_local/fa_mmlu_benchmark.json \
           --report results/persian/fa_mmlu_alignment.json
"""
import argparse
import collections
import glob
import json
import os
import sys
import re

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from medrag_repro.persian import clean_text  # noqa: E402
DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٫", "01234567890123456789.")
NUM = re.compile(r"\d+(?:\.\d+)?")


def numbers(text):
    return collections.Counter(NUM.findall(text.translate(DIGITS)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default=os.path.join(ROOT, "data", "mirage_benchmark.json"))
    p.add_argument("--revision", default=None, help="pin a dataset commit for exact reproducibility")
    p.add_argument("--out", default=os.path.join(ROOT, "data_local", "fa_mmlu_benchmark.json"))
    p.add_argument("--report", default=os.path.join(ROOT, "results", "persian", "fa_mmlu_alignment.json"))
    a = p.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download

    repo = "gaokerena/FA_MED_MMLU"
    revision = a.revision or HfApi().dataset_info(repo).sha
    path = snapshot_download(repo, repo_type="dataset", revision=revision)
    mmlu = json.load(open(a.benchmark))["mmlu"]

    out, report = {}, {"dataset": repo, "revision": revision, "subjects": {}, "number_mismatches": []}
    agree = total = with_numbers = numbers_kept = 0
    for f in sorted(glob.glob(os.path.join(path, "data", "*.parquet"))):
        subject = os.path.basename(f).split("-")[0]
        rows = pq.read_table(f).to_pylist()
        ids = sorted(k for k in mmlu if k.rsplit("-", 1)[0] == subject)
        if len(ids) != len(rows):
            raise SystemExit(f"{subject}: {len(rows)} Persian items vs {len(ids)} in MIRAGE")
        sub_agree = 0
        for qid, r in zip(ids, rows):
            letter = "ABCD"[int(str(r["answer"]).strip()) - 1]
            sub_agree += letter == mmlu[qid]["answer"]
            options = {L: clean_text(r[f"option{i}"]) for i, L in enumerate("ABCD", 1)}
            out[qid] = {"question": clean_text(r["question"]), "options": options, "answer": letter}
            en = numbers(mmlu[qid]["question"] + " " + " ".join(mmlu[qid]["options"].values()))
            fa = numbers(r["question"] + " " + " ".join(options.values()))
            if en:
                with_numbers += 1
                if en <= fa:
                    numbers_kept += 1
                else:
                    report["number_mismatches"].append({"qid": qid, "missing": dict(en - fa)})
        report["subjects"][subject] = {"items": len(rows), "answer_key_agrees": sub_agree}
        agree += sub_agree
        total += len(rows)
    report.update({"items": total, "answer_key_agrees": agree,
                   "questions_with_numbers": with_numbers, "all_numbers_kept": numbers_kept})

    for target, obj in [(a.out, {"fa_mmlu": out}), (a.report, report)]:
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        json.dump(obj, open(target, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "number_mismatches"}, indent=1))


if __name__ == "__main__":
    main()
