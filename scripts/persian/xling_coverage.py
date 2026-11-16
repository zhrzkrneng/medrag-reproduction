"""2x2 cross-lingual analysis: question language x corpus, all with BGE-M3 dense retrieval.

The same 1,089 MMLU-Med questions exist in English (MIRAGE) and Persian (FA_MED_MMLU, aligned item by
item). Coverage (does the correct option's text occur in the top-k snippets?) is always scored in
the CORPUS language, using that language's version of the options, so within a column the
measurement is identical and only the question language changes:

                      English Textbooks       Persian web corpus
  English question    cell 1                  cell 4 (cross-lingual)
  Persian question    cell 2 (cross-lingual)  cell 3

Snippet text is content only in both corpora. Paired comparisons (same question, two cells) use an
exact McNemar test on "correct option present".

Usage: python scripts/persian/xling_coverage.py --textbooks_dir <MedRAG textbooks/chunk> \
           --fa_chunks data_local/fa_corpus_chunks.jsonl.gz --out results/persian/xling_coverage.json
"""
import argparse
import gzip
import json
import os
import sys
from math import comb

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from answer_coverage import GENERIC, norm  # noqa: E402
from medrag_repro.retrieval import load_corpus  # noqa: E402

KS = [1, 4, 8, 16, 32]


def present_gold(item, snippets):
    t = norm(item["options"][item["answer"]])
    return t.strip() not in GENERIC and len(t.strip()) >= 3 and any(t in s for s in snippets)


def mcnemar(a, b):
    f = sum(x and not y for x, y in zip(a, b))
    g = sum(y and not x for x, y in zip(a, b))
    n = f + g
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(min(f, g) + 1)) / 2 ** n), 4) if n else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--textbooks_dir", required=True)
    p.add_argument("--fa_chunks", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    R = os.path.join(ROOT, "results", "persian")

    def load(path, ds):
        return {r["qid"]: r["bge_m3"] for r in map(json.loads, gzip.open(path, "rt")) if r["dataset"] == ds}

    en = json.load(open(os.path.join(ROOT, "data", "mirage_benchmark.json")))["mmlu"]
    fa = json.load(open(os.path.join(ROOT, "data_local", "fa_mmlu_benchmark.json")))["fa_mmlu"]
    cells = {  # name: (retrieved lists, corpus, options used for scoring)
        "1_en_q__textbooks": (load(f"{R}/xling_textbooks_top100.jsonl.gz", "mmlu"), "en", en),
        "2_fa_q__textbooks": (load(f"{R}/xling_textbooks_top100.jsonl.gz", "fa_mmlu"), "en", en),
        "3_fa_q__fa_web": (load(f"{R}/bge_m3_top100.jsonl.gz", "fa_mmlu"), "fa", fa),
        "4_en_q__fa_web": (load(f"{R}/xling_fa_web_top100.jsonl.gz", "mmlu"), "fa", fa),
    }
    need = {i for ret, _, _ in cells.values() for lst in ret.values() for i in lst[:32]}
    text = {s["id"]: norm(s["content"]) for s in load_corpus(a.textbooks_dir) if s["id"] in need}
    for line in gzip.open(a.fa_chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            text[s["id"]] = norm(s["content"])

    qids = sorted(en)
    hits, report = {}, {"questions": len(qids), "coverage_%": {}, "paired_tests_k32": {}}
    for name, (ret, _, bench) in cells.items():
        report["coverage_%"][name] = {}
        for k in KS:
            h = [present_gold(bench[q], [text[i] for i in ret[q][:k]]) for q in qids]
            report["coverage_%"][name][k] = round(100 * sum(h) / len(h), 1)
            hits[(name, k)] = h
    for x, y in [("1_en_q__textbooks", "2_fa_q__textbooks"), ("3_fa_q__fa_web", "4_en_q__fa_web")]:
        report["paired_tests_k32"][f"{x} vs {y}"] = mcnemar(hits[(x, 32)], hits[(y, 32)])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
