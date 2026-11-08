"""Score every results/<model>/<dataset>/<config>.jsonl file and print a table.

Usage: python scripts/evaluate.py --results results [--csv results/summary.csv]
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from medrag_repro.evaluate import score  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--results", default="results")
p.add_argument("--csv", default=None)
a = p.parse_args()

rows = []
for path in sorted(glob.glob(os.path.join(a.results, "*", "*", "*.jsonl"))):
    model, ds, conf = path.split(os.sep)[-3], path.split(os.sep)[-2], os.path.basename(path)[:-6]
    recs = [json.loads(line) for line in open(path)]
    s = score(recs, pmc_llama="pmc_llama" in model.lower())
    rows.append({"model": model, "dataset": ds, "config": conf, **s})

print(f"{'model':32} {'dataset':8} {'config':12} {'n':>5} {'acc':>7} {'±std':>6} {'strict':>7} {'unparsed':>8}")
for r in rows:
    print(f"{r['model']:32} {r['dataset']:8} {r['config']:12} {r['n']:5d} {100*r['acc_official']:6.2f}% "
          f"{100*r['std']:5.2f} {100*r['acc_strict']:6.2f}% {r['parse_failures']:8d}")
if a.csv and rows:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
