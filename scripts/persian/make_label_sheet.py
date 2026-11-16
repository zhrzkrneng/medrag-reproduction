"""Blind hand-labelling sheet for checking the answer parsers on OUR Persian outputs.

Samples 100 model answers (seed 0): 50 under MedRAG's prompt (read by medrag_repro.evaluate's official
parser) and 50 under Gaokerena's prompt (read by scripts/persian/gk_parse.py), split evenly over
Gaokerena, Aya and Qwen and over FA_MED_MMLU and KOPP. 40% of each cell are "hard" cases (the strict
parser fails, or the Gaokerena parser returns nothing), so that parser errors are not missed; the
report gives agreement for random and hard rows separately, and the random rows estimate the overall
error rate.

The sheet shows only an opaque row id, the four options and the model's answer: no gold answer, no
parser output, no model or prompt name. The key (row id -> source, parser output) is written next to
it. Both files contain benchmark text and model answers, so they go to data_local/ (git-ignored) and
are never committed; only the agreement statistics are.

Usage: python scripts/persian/make_label_sheet.py --runs <dir with out_rq2full, out_rq2gk, out_run_fa>
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from gk_parse import parse as gk_parse  # noqa: E402
from medrag_repro.evaluate import official_prediction, strict_prediction  # noqa: E402

MODELS = {"gaokerena": "gaokerena__gaokerena-v1.0", "aya": "CohereLabs__aya-expanse-8b", "qwen": "Qwen__Qwen2.5-7B-Instruct"}
INSTRUCTIONS = [
    "راهنمای برچسب‌زدن",
    "",
    "در هر ردیف، جوابی را که یک مدل زبانی به یک سؤال چهارگزینه‌ای پزشکی داده می‌بینید.",
    "کار شما فقط این است: مدل در نهایت کدام گزینه را انتخاب کرده؟",
    "مهم: این بررسیِ انسانی است. لطفاً خودتان و بدون کمک هیچ ابزار هوش مصنوعی برچسب بزنید؛ وگرنه نمی‌توان آن را در مقاله «برچسب دستی» گزارش کرد.",
    "",
    "در ستون «برچسب شما» یکی از این‌ها را بنویسید:",
    "  A یا B یا C یا D : گزینه‌ای که مدل در نهایت انتخاب کرده",
    "  invalid : مدل هیچ گزینه‌ای انتخاب نکرده (مثلاً «هیچ‌کدام درست نیست»)، یا دو گزینه یا بیشتر را با هم انتخاب کرده، یا جواب ناتمام مانده",
    "  unsure : مطمئن نیستید (در ستون «توضیح» بنویسید چرا)",
    "",
    "قواعد:",
    "روش سریع: اول ستون سبز «پایان جواب مدل» را بخوانید؛ بیشتر مدل‌ها انتخاب نهایی را در آخر جواب می‌نویسند.",
    "نمونه‌ها: «پاسخ صحیح گزینه [C] است» → C ؛ \"answer_choice\": \"B\" → B ؛ «بنابراین گزینه الف درست است» → A ؛ «هیچ‌کدام از گزینه‌ها درست نیست» → invalid",
    "اگر در پایان جواب، انتخابی نبود یا مبهم بود، ستون «جواب کامل مدل» را بخوانید.",
    "",
    "1. جواب نهایی ملاک است. اگر مدل اول B گفته و در آخر C را انتخاب کرده، C بنویسید.",
    "2. لازم نیست بدانید جواب درست کدام است. درستی پزشکی مهم نیست؛ فقط انتخاب مدل مهم است.",
    "3. اگر مدل به جای حرف، متن گزینه را نوشته، با ستون «گزینه‌ها» حرفش را پیدا کنید.",
    "4. گزینه‌ها ممکن است با حروف فارسی (الف، ب، ج، د) یا عدد (۱ تا ۴) آمده باشند: الف = A، ب = B، ج = C، د = D و ۱ = A، ۲ = B، ۳ = C، ۴ = D.",
    "5. جواب‌های انگلیسی هم هستند (قالب JSON با answer_choice)؛ همان قاعده‌ها.",
    "6. ردیف‌ها را به ترتیب و بدون عجله بروید؛ وسط کار ذخیره کنید. حدود ۱ تا ۲ دقیقه برای هر ردیف کافی است.",
    "",
    "وقتی تمام شد، فایل را ذخیره کنید و به Claude خبر بدهید.",
]


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out_dir", default=os.path.join(ROOT, "data_local", "labeling"))
    p.add_argument("--n_per_cell", type=int, nargs=4, default=[25, 25, 25, 25],
                   help="rows per (prompt, dataset) cell, in the order medrag/fa_mmlu, medrag/kopp, gaokerena/fa_mmlu, gaokerena/kopp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude_key", nargs="*", default=[], help="label_key.json files whose rows are not drawn again")
    p.add_argument("--name", default="label_sheet", help="output base name (<name>.xlsx and <name>_key.json)")
    a = p.parse_args()
    bench = {**json.load(open(os.path.join(ROOT, "data_local", "fa_mmlu_benchmark.json"))),
             **json.load(open(os.path.join(ROOT, "data_local", "kopp_benchmark.json")))}
    rng = random.Random(a.seed)
    seen = {(k["dataset"], k["qid"], k["model"], k["prompt"], k["config"]) for f in a.exclude_key for k in json.load(open(f))}
    rows = []
    cells = iter(a.n_per_cell)
    for prompt in ("medrag", "gaokerena"):
        for ds in ("fa_mmlu", "kopp"):
            n_cell = next(cells)
            per_model = [n_cell // 3 + (1 if i < n_cell % 3 else 0) for i in range(3)]
            for (mname, slug), n in zip(MODELS.items(), per_model):
                pool = []
                for conf in ("cot", "rrf2_k4"):
                    if prompt == "medrag":
                        run = "out_run_fa" if mname == "qwen" else "out_rq2full"
                        path = os.path.join(a.runs, run, "results", slug, ds, f"{conf}.jsonl")
                    else:
                        if mname == "qwen" and conf != "cot":
                            continue
                        path = os.path.join(a.runs, "out_rq2gk", "results", slug, ds, f"{conf}_gk.jsonl")
                    for r in jl(path):
                        if (ds, r["qid"], mname, prompt, conf) in seen:
                            continue
                        resp = r["response"]
                        if prompt == "medrag":
                            parsed = official_prediction(resp)
                            hard = strict_prediction(resp) is None
                        else:
                            parsed = gk_parse(resp, bench[ds][r["qid"]]["options"])
                            hard = parsed is None
                        pool.append({"dataset": ds, "qid": r["qid"], "model": mname, "prompt": prompt, "config": conf,
                                     "parsed": parsed, "hard": hard, "response": resp})
                hard_pool = [x for x in pool if x["hard"]]
                easy_pool = [x for x in pool if not x["hard"]]
                n_hard = min(len(hard_pool), round(0.4 * n))
                pick = rng.sample(hard_pool, n_hard) + rng.sample(easy_pool, n - n_hard)
                for x in pick:
                    x["sample"] = "hard" if x["hard"] else "random"
                rows += pick
    rng.shuffle(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    key = []
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "برچسب"
    ws.sheet_view.rightToLeft = True
    head = ["شناسه", "گزینه‌ها", "پایان جواب مدل (معمولاً انتخاب نهایی اینجاست)", "جواب کامل مدل", "برچسب شما", "توضیح (اختیاری)"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for i, x in enumerate(rows, 1):
        rid = f"L{i:03d}"
        opts = bench[x["dataset"]][x["qid"]]["options"]
        resp = x["response"].strip()
        tail = resp if len(resp) <= 400 else "… " + resp[-400:]
        ws.append([rid, "\n".join(f"{k}: {v}" for k, v in opts.items()), tail, resp[:32000], "", ""])
        key.append({"id": rid, **{k: x[k] for k in ("dataset", "qid", "model", "prompt", "config", "parsed", "sample")}})
    for col, w in zip("ABCDEF", (9, 40, 60, 90, 14, 30)):
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        row[4].fill = PatternFill("solid", fgColor="FFF2CC")
        row[2].fill = PatternFill("solid", fgColor="E2EFDA")
    ws.freeze_panes = "A2"
    ins = wb.create_sheet("راهنما", 0)
    ins.sheet_view.rightToLeft = True
    for line in INSTRUCTIONS:
        ins.append([line])
    ins.column_dimensions["A"].width = 140
    ins["A1"].font = Font(bold=True, size=14)
    xlsx = os.path.join(a.out_dir, f"{a.name}.xlsx")
    wb.save(xlsx)
    json.dump(key, open(os.path.join(a.out_dir, f"{a.name}_key.json" if a.name != "label_sheet" else "label_key.json"), "w"), indent=1)
    from collections import Counter
    print(xlsx, len(rows), Counter((x["prompt"], x["model"], x["sample"]) for x in rows))


if __name__ == "__main__":
    main()
