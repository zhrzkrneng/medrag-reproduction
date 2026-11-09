"""Answer coverage of retrieval, Persian vs English: how often does the text of the correct option
(and of a wrong option) occur in the top-k retrieved snippets? A lexical proxy (see
scripts/retrieval_quality.py); here with Unicode-aware normalisation for Persian (Arabic yeh/kaf
unified, ZWNJ -> space, diacritics removed).

Caveat: Persian options are translations, and Persian morphology and spelling variants make exact
phrase matching harder, so part of a Persian-English gap can come from the measurement itself.

Usage:
  python scripts/persian/answer_coverage.py --en_chunk_dir textbooks/chunk \
      --fa_chunks data_local/fa_corpus_chunks.jsonl.gz --out results/persian/answer_coverage_bm25.json
"""
import argparse
import gzip
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from medrag_repro.retrieval import load_corpus  # noqa: E402

GENERIC = {"none", "all", "both", "neither", "none of the above", "all of the above", "yes", "no",
           "هیچکدام", "همه موارد", "هر دو"}


def norm(text):
    t = unicodedata.normalize("NFKC", text).replace("ي", "ی").replace("ك", "ک").replace("‌", " ")
    t = re.sub(r"[ً-ْٰ]", "", t.replace("ة", "ه")).lower()
    return " " + " ".join(re.findall(r"\w+", t)) + " "


def coverage(bench, ret, corpus, k):
    gold = wrong = 0
    for qid, item in bench.items():
        snips = [corpus[i] for i in ret[qid][:k]]
        found = set()
        for o, text in item["options"].items():
            t = norm(text)
            if t.strip() not in GENERIC and len(t.strip()) >= 3 and any(t in s for s in snips):
                found.add(o)
        gold += item["answer"] in found
        wrong += bool(found - {item["answer"]})
    return {"gold_present_%": round(100 * gold / len(bench), 1), "wrong_present_%": round(100 * wrong / len(bench), 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--en_chunk_dir", required=True)
    p.add_argument("--en_retrieved", default=os.path.join(ROOT, "results", "retrieval", "textbooks_top32.jsonl.gz"))
    p.add_argument("--fa_chunks", required=True)
    p.add_argument("--fa_retrieved", default=os.path.join(ROOT, "results", "persian", "fa_mmlu_bm25_top32.jsonl.gz"))
    p.add_argument("--fa_benchmark", default=os.path.join(ROOT, "data_local", "fa_mmlu_benchmark.json"))
    p.add_argument("--retriever", default="bm25")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    en = json.load(open(os.path.join(ROOT, "data", "mirage_benchmark.json")))["mmlu"]
    fa = json.load(open(a.fa_benchmark))["fa_mmlu"]
    eret = {}
    for line in gzip.open(a.en_retrieved, "rt"):
        r = json.loads(line)
        if r["dataset"] == "mmlu":
            eret[r["qid"]] = r[a.retriever]
    fret = {r["qid"]: r[a.retriever] for r in map(json.loads, gzip.open(a.fa_retrieved, "rt"))}
    ecorp = {s["id"]: norm(s["title"] + " " + s["content"]) for s in load_corpus(a.en_chunk_dir)}
    need = {i for v in fret.values() for i in v}
    fcorp = {}
    for line in gzip.open(a.fa_chunks, "rt", encoding="utf-8"):
        s = json.loads(line)
        if s["id"] in need:
            fcorp[s["id"]] = norm(s["content"])
    report = {"retriever": a.retriever, "questions": len(fa),
              "english_textbooks": {k: coverage(en, eret, ecorp, k) for k in [1, 4, 8, 16, 32]},
              "persian_web_corpus": {k: coverage(fa, fret, fcorp, k) for k in [1, 4, 8, 16, 32]}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
