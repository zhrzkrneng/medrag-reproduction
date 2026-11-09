"""RQ4: do Persian benchmark questions appear in the Persian corpus (benchmark leakage)?

Gaokerena was trained on gaokerena/medical_corpus, and our Persian RAG retrieves from it, so a
question copied into the corpus would make both "fine-tuning" and "RAG" look better than they are.

Method (n-gram overlap, the approach used for contamination checks in GPT-3 / later LLM reports):
  * normalise text (Unicode NFKC, Arabic yeh/kaf -> Persian, ZWNJ -> space, diacritics removed,
    Persian/Arabic digits -> ASCII) and split into word tokens;
  * for every question, take its word n-grams (n = 8 and n = 13) from the question stem;
  * stream all corpus documents and record which question n-grams occur, and in which document;
  * a question is flagged when >= 50% of its n-grams occur in ONE document (stems with fewer than
    n words cannot be checked and are reported separately);
  * for flagged questions, also check whether the correct option's text occurs in that document.

Only counts, question ids and document indices are written (no corpus text).

Usage: python scripts/persian/leakage_check.py --corpus_dir <medical_corpus snapshot> \
           --kopp_dir <KOPP snapshot> --fa_mmlu data_local/fa_mmlu_benchmark.json \
           --out results/persian/leakage.json
"""
import argparse
import collections
import glob
import json
import os
import re
import unicodedata

DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def tokens(text):
    t = unicodedata.normalize("NFKC", text or "").replace("ي", "ی").replace("ك", "ک").replace("ة", "ه")
    t = re.sub(r"[ً-ْٰ]", "", t.replace("‌", " ")).translate(DIGITS).lower()
    return re.findall(r"\w+", t)


def ngrams(toks, n):
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def load_benchmarks(a):
    import pyarrow.parquet as pq
    qs = {}
    for split in ["sept23", "other_years"]:  # the repo's data/train-* file duplicates sept23 and is not in its configs
        for i, r in enumerate(pq.read_table(glob.glob(os.path.join(a.kopp_dir, "data", f"{split}-*.parquet"))[0]).to_pylist()):
            qs[f"kopp/{split}/{i}"] = {"question": r["Question"], "answer_text": r[f"Option{int(r['Correct answer'])}"]}
    for qid, it in json.load(open(a.fa_mmlu))["fa_mmlu"].items():
        qs[f"fa_mmlu/{qid}"] = {"question": it["question"], "answer_text": it["options"][it["answer"]]}
    return qs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir", required=True)
    p.add_argument("--kopp_dir", required=True)
    p.add_argument("--fa_mmlu", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import pyarrow.parquet as pq

    qs = load_benchmarks(a)
    report = {"threshold": a.threshold, "benchmarks": {}}
    for n in (8, 13):
        qgrams = {q: ngrams(tokens(v["question"]), n) for q, v in qs.items()}
        index = collections.defaultdict(list)
        for q, gs in qgrams.items():
            for g in gs:
                index[g].append(q)
        # per question: document index -> number of its n-grams found there
        hits = collections.defaultdict(collections.Counter)
        docs_text = {}
        doc_i = 0
        for f in sorted(glob.glob(os.path.join(a.corpus_dir, "data", "*.parquet"))):
            for row in pq.read_table(f).to_pylist():
                toks = tokens(row["content"])
                found = set()
                for g in ngrams(toks, n):
                    if g in index:
                        found.add(g)
                for g in found:
                    for q in index[g]:
                        hits[q][doc_i] += 1
                if found:
                    docs_text[doc_i] = " " + " ".join(toks) + " "
                doc_i += 1
        for bench in ("kopp", "fa_mmlu"):
            ids = [q for q in qs if q.startswith(bench + "/")]
            checkable = [q for q in ids if qgrams[q]]
            flagged, best = [], {}
            for q in checkable:
                if not hits[q]:
                    best[q] = 0.0
                    continue
                doc, c = hits[q].most_common(1)[0]
                share = c / len(qgrams[q])
                best[q] = share
                if share >= a.threshold:
                    ans = " " + " ".join(tokens(qs[q]["answer_text"])) + " "
                    flagged.append({"qid": q, "share": round(share, 3), "doc": doc,
                                    "answer_in_doc": len(ans.strip()) > 0 and ans in docs_text[doc]})
            report["benchmarks"].setdefault(bench, {})[f"{n}-gram"] = {
                "questions": len(ids), "checkable": len(checkable),
                "flagged": len(flagged), "flagged_%": round(100 * len(flagged) / max(1, len(checkable)), 2),
                "flagged_with_answer_in_doc": sum(x["answer_in_doc"] for x in flagged),
                # distribution of each checkable question's highest overlap with a single document
                "max_share_histogram": {f">={lo}": sum(v >= lo for v in best.values()) for lo in (0.1, 0.2, 0.3, 0.4, 0.5)},
                "highest_overlaps": sorted(({"qid": q, "share": round(v, 3)} for q, v in best.items()),
                                           key=lambda x: -x["share"])[:10],
                "flagged_items": flagged}
        print(n, {b: {k: v for k, v in r[f"{n}-gram"].items() if k != "flagged_items"}
                  for b, r in report["benchmarks"].items()}, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
