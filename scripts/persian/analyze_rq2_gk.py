"""RQ2 with Gaokerena's own prompt (kaggle/rq2_gk): accuracy, paired tests, comparison with the
MedRAG-prompt run (results/persian/rq2) and with Gaokerena's published hand-labelled results.

Usage: python scripts/persian/analyze_rq2_gk.py --run <kaggle output dir> --gk_repo <clone of Gaokerena-V>
         [--medrag_run <kaggle/rq2 output dir>] --out_dir results/persian/rq2_gk
Writes ids, letters and statistics only (no question or answer text).
"""
import argparse, ast, glob, gzip, json, os, re, sys
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from gk_parse import parse
from xling_coverage import mcnemar
from medrag_repro.evaluate import official_prediction
from medrag_repro.persian import clean_text

ROOT = os.path.join(HERE, "..", "..")
M = {"gaokerena": "gaokerena__gaokerena-v1.0", "aya": "CohereLabs__aya-expanse-8b", "qwen": "Qwen__Qwen2.5-7B-Instruct"}


def load(path):
    return {(r := json.loads(l))["qid"]: r for l in open(path)} if os.path.exists(path) else None


def norm(s):
    s = clean_text(str(s)).replace("ي", "ی").replace("ك", "ک").replace("‌", " ")
    return re.sub(r"\s+", "", s)


def capped(r):
    return r["output_tokens"] >= min(1024, 8192 - r["prompt_tokens"])


def pct(x):
    return round(100 * sum(x) / len(x), 2) if x else None


def pair(a, b):
    return {"diff_points": round(100 * (sum(b) - sum(a)) / len(a), 2), "p": round(mcnemar(a, b), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--gk_repo", required=True)
    ap.add_argument("--medrag_run"); ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    R = a.run + "/results"
    rep = {"model_revisions": json.load(open(f"{R}/model_revisions.json")),
           "prompt": "Gaokerena prompt_eval_bare (github.com/Mehrdadghassabi/Gaokerena-V @ 13eb1f7); RAG adds RRF-2 k = 4 under '## متن'",
           "parser": "scripts/persian/gk_parse.py (agreement with their hand labels: results/persian/rq2_gk/parser_check.json)"}
    G = f"{a.gk_repo}/evaluation/multiple_choice_qa"
    for ds in ["fa_mmlu", "kopp"]:
        bench = json.load(open(f"{ROOT}/data_local/{ds}_benchmark.json"))[ds]
        runs = {}
        for m, slug in M.items():
            for cf in ["cot", "rrf2_k4"]:
                d = load(f"{R}/{slug}/{ds}/{cf}_gk.jsonl")
                if d:
                    runs[f"{m}_{cf}"] = d
        qids = sorted(set.intersection(*[set(d) for d in runs.values()]))
        pred = {k: {q: parse(d[q]["response"], bench[q]["options"]) for q in qids} for k, d in runs.items()}
        corr = {k: [pred[k][q] == bench[q]["answer"] for q in qids] for k in runs}
        out = {"n": len(qids), "accuracy_%": {k: pct(v) for k, v in corr.items()},
               "invalid": {k: sum(p is None for p in pred[k].values()) for k in runs},
               "answer_hit_length_cap": {k: sum(capped(runs[k][q]) for q in qids) for k in runs}}
        pairs = [("aya_cot", "gaokerena_cot"), ("aya_cot", "aya_rrf2_k4"), ("gaokerena_cot", "gaokerena_rrf2_k4"),
                 ("gaokerena_cot", "aya_rrf2_k4"), ("aya_cot", "gaokerena_rrf2_k4"), ("aya_rrf2_k4", "gaokerena_rrf2_k4"),
                 ("qwen_cot", "gaokerena_cot"), ("qwen_cot", "aya_cot")]
        out["paired_tests"] = {f"{x} vs {y}": pair(corr[x], corr[y]) for x, y in pairs if x in corr and y in corr}
        # same model and config with MedRAG's prompt (kaggle/rq2; Qwen from run_fa)
        if a.medrag_run:
            mr = {}
            for m in ["gaokerena", "aya"]:
                for cf in ["cot", "rrf2_k4"]:
                    d = load(f"{a.medrag_run}/results/{M[m]}/{ds}/{cf}.jsonl")
                    if d and f"{m}_{cf}" in corr:
                        c2 = [official_prediction(d[q]["response"]) == bench[q]["answer"] for q in qids]
                        mr[f"{m}_{cf}"] = {"medrag_prompt_%": pct(c2), "gaokerena_prompt_%": pct(corr[f"{m}_{cf}"]),
                                           **pair(c2, corr[f"{m}_{cf}"])}
            qp = f"{ROOT}/results/persian/run_fa/{ds}_predictions.jsonl.gz"
            if os.path.exists(qp) and "qwen_cot" in corr:
                qq = {(r := json.loads(l))["qid"]: r for l in gzip.open(qp, "rt")}
                key = next(k for k in qq[qids[0]] if k.startswith("cot"))
                c2 = [qq[q][key] == bench[q]["answer"] for q in qids]
                mr["qwen_cot"] = {"medrag_prompt_%": pct(c2), "gaokerena_prompt_%": pct(corr["qwen_cot"]), **pair(c2, corr["qwen_cot"])}
            out["prompt_effect (MedRAG -> Gaokerena prompt)"] = mr
        # their published results (hand labels) on the same items
        import openpyxl
        # question text alone is not unique (48 FA_MED_MMLU stems repeat with other options): key on stem + options
        key = lambda qu, op: norm(qu) + "|" + "|".join(norm(op[k]) for k in "ABCD")
        idx = {key(v["question"], v["options"]): q for q, v in bench.items()}
        stems = {}
        for q, v in bench.items():
            stems.setdefault(norm(v["question"]), []).append(q)
        sub = "kopp" if ds == "kopp" else "MMLU*"
        pub = {}
        for m, pat in [("gaokerena", "gaokerena"), ("aya", "aya*"), ("qwen", "qwen2.5")]:
            lab = {}
            for x in glob.glob(f"{G}/{sub}/{pat}/*.xlsx"):
                for r in openpyxl.load_workbook(x, read_only=True).active.iter_rows(min_row=2, values_only=True):
                    if not r[0]:
                        continue
                    try:
                        q = idx.get(key(r[0], ast.literal_eval(r[1])))
                    except Exception:
                        q = None
                    if q is None and len(stems.get(norm(r[0]), [])) == 1:  # option text differs slightly; stem unique
                        q = stems[norm(r[0])][0]
                    if q is not None:
                        h = str(r[5]).strip()
                        lab[q] = h if h in ("A", "B", "C", "D") else None
            k = f"{m}_cot"
            if k not in pred:
                continue
            qs = [q for q in qids if q in lab]
            ours = [pred[k][q] == bench[q]["answer"] for q in qs]
            theirs = [lab[q] == bench[q]["answer"] for q in qs]
            pub[m] = {"n_matched": len(qs), "published_%": pct(theirs), "ours_%": pct(ours), **pair(theirs, ours),
                      "same_letter_%": pct([pred[k][q] == lab[q] for q in qs])}
        out["vs_published (their hand labels, same items, no retrieval)"] = pub
        rep[ds] = out
        with gzip.open(f"{a.out_dir}/{ds}_predictions.jsonl.gz", "wt") as f:
            for q in qids:
                f.write(json.dumps({"qid": q, "gold": bench[q]["answer"], **{k: pred[k][q] for k in runs}}) + "\n")
    json.dump(rep, open(f"{a.out_dir}/summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({ds: rep[ds] for ds in ["fa_mmlu", "kopp"]}, indent=1))


if __name__ == "__main__":
    main()
