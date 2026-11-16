"""Agreement between the answer parsers and a second reading of 150 Persian model answers.

IMPORTANT, provenance of the labels: they were produced with an AI tool (files saved by openpyxl;
the tool's own notes say so), NOT by a human annotator. They are reported as an AI-assisted check
of the parsers, never as human annotation or inter-annotator agreement.

Inputs (git-ignored, in data_local/labeling/): the two keys written by make_label_sheet.py and the
returned sheets. For the first 100 rows the final column of the third review is used; for the 50
fresh rows, the only label column. 'unsure' rows are counted apart. A parser output of None (and,
for MedRAG's official parser, nothing: it never returns None) is compared as 'invalid'. MedRAG's
official parser falls back to 'A' when it finds no answer (src/medrag_repro/evaluate.py,
locate_answer), which is why hard rows are reported separately.

Wilson 95% intervals. Output: results/persian/parser_check_ai_labels.json (counts only).

Usage: python scripts/persian/score_labels.py
"""
import json
import math
import os
from collections import defaultdict

import openpyxl

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
L = os.path.join(ROOT, "data_local", "labeling")


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def labels(path, column):
    ws = openpyxl.load_workbook(path)["برچسب"]
    hdr = [c.value for c in ws[1]]
    j = hdr.index(column)
    return {r[0].value: str(r[j].value).strip() for r in ws.iter_rows(min_row=2)}


def main():
    sets = [(os.path.join(L, "label_key.json"), os.path.join(L, "in", "third_review.xlsx"), "برچسب بازبینی سوم"),
            (os.path.join(L, "label_sheet_human50_key.json"), os.path.join(L, "in", "human50_labeled.xlsx"), "برچسب شما")]
    cells = defaultdict(lambda: {"n": 0, "agree": 0, "unsure": 0, "parser_A_label_other": 0})
    for keyf, sheet, col in sets:
        lab = labels(sheet, col)
        for k in json.load(open(keyf)):
            l = lab[k["id"]]
            parser = k["parsed"] or "invalid"
            for c in [(k["prompt"], k["sample"]), (k["prompt"], "all"), ("all", k["sample"]), ("all", "all")]:
                x = cells[c]
                if l == "unsure":
                    x["unsure"] += 1
                    continue
                x["n"] += 1
                x["agree"] += int(l == parser)
                x["parser_A_label_other"] += int(parser == "A" and l != "A")
    out = {"labels": "AI-assisted (not human); see the docstring", "rows": 150,
           "method": __doc__.split("Usage")[0].strip(), "agreement": {}}
    for (prompt, sample), x in sorted(cells.items()):
        parser = {"medrag": "MedRAG official (medrag_repro.evaluate)", "gaokerena": "gk_parse.py", "all": "both"}[prompt]
        out["agreement"][f"{prompt}/{sample}"] = {"parser": parser, **x,
                                                  "agree_%": round(100 * x["agree"] / x["n"], 1) if x["n"] else None,
                                                  "ci95": wilson(x["agree"], x["n"])}
    json.dump(out, open(os.path.join(ROOT, "results", "persian", "parser_check_ai_labels.json"), "w"), indent=1,
              ensure_ascii=False)
    for k, v in out["agreement"].items():
        print(k, {kk: v[kk] for kk in ("n", "agree", "agree_%", "ci95", "unsure", "parser_A_label_other")})


if __name__ == "__main__":
    main()
