"""Rotate the answer options of a MIRAGE-format dataset (A->B, B->C, ..., last->A), keeping each option's text.

A contamination probe: a model that recalls MMLU items (question -> letter) loses accuracy when the letters move;
one that reads the options does not. Deterministic, every item rotated by one place.
Usage: python scripts/persian/shuffle_options.py --benchmark data/mirage_benchmark.json --dataset mmlu \
           --out mmlu_rot.json --out_key mmlu_rot
"""
import argparse
import json


def rotate(item):
    keys = sorted(item["options"])
    texts = [item["options"][k] for k in keys]
    new = {keys[(i + 1) % len(keys)]: t for i, t in enumerate(texts)}
    return {**item, "options": {k: new[k] for k in keys}, "answer": keys[(keys.index(item["answer"]) + 1) % len(keys)]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--out_key", required=True)
    a = p.parse_args()
    bench = json.load(open(a.benchmark))[a.dataset]
    out = {q: rotate(it) for q, it in bench.items()}
    for q, it in out.items():  # the correct text is unchanged, only its letter moves
        assert it["options"][it["answer"]] == bench[q]["options"][bench[q]["answer"]]
    json.dump({a.out_key: out}, open(a.out, "w"), ensure_ascii=False)
    print(a.out_key, len(out))


if __name__ == "__main__":
    main()
