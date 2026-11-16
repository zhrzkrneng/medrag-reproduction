"""Validate gk_parse.parse against Gaokerena's hand labels (their published result files).

Usage: python scripts/persian/gk_parse_check.py --gk_repo <clone of Mehrdadghassabi/Gaokerena-V> --out <json>
Only agreement statistics are written (no question or answer text).
"""
import argparse, ast, collections, glob, json, os, sys
import openpyxl
sys.path.insert(0, os.path.dirname(__file__))
from gk_parse import parse

ap = argparse.ArgumentParser(); ap.add_argument("--gk_repo", required=True); ap.add_argument("--out")
a = ap.parse_args()
stats = collections.defaultdict(lambda: collections.Counter())
conf = collections.Counter()
for x in sorted(glob.glob(f"{a.gk_repo}/evaluation/multiple_choice_qa/*/*/*.xlsx")):
    model = x.split("/")[-2]
    if not any(k in model for k in ("gaokerena", "aya", "qwen")):
        continue
    model = "gaokerena" if "gaokerena" in model else "aya" if "aya" in model else "qwen"
    for r in openpyxl.load_workbook(x, read_only=True).active.iter_rows(min_row=2, values_only=True):
        if not r[0]:
            continue
        try:
            opts = ast.literal_eval(r[1]) if isinstance(r[1], str) else None
        except Exception:
            opts = None
        hand = str(r[5]).strip()
        hand = hand if hand in "ABCD" and hand else None
        auto = parse(r[4], opts)
        s = stats[model]
        s["n"] += 1
        s["agree"] += auto == hand
        s["hand_invalid"] += hand is None
        s["auto_invalid"] += auto is None
        s["hand_correct"] += hand == str(r[2]).strip()
        s["auto_correct"] += auto == str(r[2]).strip()
        conf[(model, hand or "-", auto or "-")] += 1
rep = {m: dict(s, agree_pct=round(100 * s["agree"] / s["n"], 2),
               hand_acc=round(100 * s["hand_correct"] / s["n"], 2),
               auto_acc=round(100 * s["auto_correct"] / s["n"], 2)) for m, s in stats.items()}
print(json.dumps(rep, indent=1))
dis = [(k, v) for k, v in conf.items() if k[1] != k[2]]
print(sorted(dis, key=lambda kv: -kv[1])[:15])
if a.out:
    json.dump({"source": "github.com/Mehrdadghassabi/Gaokerena-V evaluation/multiple_choice_qa", "per_model": rep}, open(a.out, "w"), indent=1)
