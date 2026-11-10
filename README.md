# medrag-reproduction

Reproduction of “Benchmarking Retrieval-Augmented Generation for Medicine” (Xiong et al., 2024): the MedRAG toolkit on the MIRAGE benchmark, scaled down to run on free hardware (CPU, plus the free Kaggle T4 GPU tier).

[English](#english) · [فارسی](#فارسی)

---

## English

### What is reproduced

| Component | Paper | This repo, phase 1 |
|---|---|---|
| Benchmark | MIRAGE, 7,663 questions in 5 datasets | MedQA-US (1,273) + MMLU-Med (1,089) |
| Corpus | PubMed, StatPearls, Textbooks, Wikipedia, MedCorp | **Textbooks** (18 books, 125,847 snippets; paper Table 3: 125.8k) |
| Retrievers | BM25, Contriever, SPECTER, MedCPT, RRF-2, RRF-4 | **BM25, MedCPT, RRF-2** |
| LLMs | GPT-4, GPT-3.5, Mixtral, Llama2-70B, MEDITRON-70B, PMC-LLaMA-13B | **Qwen2.5-7B-Instruct** (main), **PMC-LLaMA-13B** (to compare with the paper) |
| Settings | Zero-shot, multi-choice, RAG, question-only retrieval | same |

Textbooks fits the scaled-down setup because the paper reports it helps most on exam-style questions (MedQA, MMLU).

### Faithfulness to the original code

The official code is public domain (NCBI). Pinned reference commits:
[MedRAG@7599a72](https://github.com/Teddy-XiongGZ/MedRAG/tree/7599a728a28789fd601728c08d313b1148051f41) and
[MIRAGE@392943a](https://github.com/Teddy-XiongGZ/MIRAGE/tree/392943af99cd94cafd50a0de2e7fca24bbf65494).

- **Prompts:** copied verbatim from MedRAG `src/template.py` (`src/medrag_repro/templates.py`).
- **Context:** snippets are formatted as `Document [i] (Title: t) content` and truncated to `context_length` tokens. That is 7,168 for Qwen (MedRAG's Llama-3 budget) and 1,024 for PMC-LLaMA (MedRAG's value).
- **BM25:** Pyserini/Lucene with default parameters. **MedCPT:** CLS pooling, `[title, content]` pairs, exact inner product. **RRF:** `1/(100 + rank + 1)` over 100 hits per retriever.
- **Scoring:** the official MIRAGE `locate_answer` parser. It returns `"A"` when it cannot find an answer, so we **also** report a strict accuracy and the number of unparsable answers.
- `data/mirage_benchmark.json` is the official MIRAGE file (sha256 `6f7f08c6…f8fa`).

### Documented deviations

1. **Different models.** GPT-3.5/4 need a paid API, and 70B models do not fit on 2×T4. Only PMC-LLaMA-13B appears in the paper.
2. **Recomputed MedCPT embeddings.** The official pre-computed Textbooks embeddings returned HTTP 403 on 2026-11-08, so they are recomputed with the same encoder and settings (`scripts/embed_corpus.py`).
3. **Answer length capped** at 1,024 new tokens to bound GPU time. The number of answers that hit the cap is logged.
4. **fp16 instead of bf16**, because T4 GPUs do not support bf16.
5. **vLLM** instead of the Hugging Face pipeline, for speed. Decoding is greedy in both.

### Layout

```
src/medrag_repro/   templates.py · retrieval.py · evaluate.py
scripts/            retrieve.py · embed_corpus.py · generate.py · evaluate.py · retrieval_quality.py
templates/          pmc_llama.jinja (from MedRAG)
data/               mirage_benchmark.json
tests/              unit tests for parsing, RRF and prompts
```

### Running

```bash
pip install -r requirements.txt   # retrieval (CPU); needs Java 21 for Pyserini
python scripts/embed_corpus.py --chunk_dir textbooks/chunk --out work/medcpt_article_emb.npy   # GPU: minutes
python scripts/retrieve.py --chunk_dir textbooks/chunk --work_dir work --out retrieved/textbooks_top32.jsonl.gz
python scripts/generate.py --model Qwen/Qwen2.5-7B-Instruct --retrieved retrieved/textbooks_top32.jsonl.gz \
    --corpus corpus/textbooks.jsonl.gz --configs cot rrf2@32 --out results   # GPU
python scripts/evaluate.py --results results
```

### Results

**Qwen2.5-7B-Instruct, Textbooks corpus, 32 snippets** (full test sets; official MIRAGE parser; ± is the binomial standard deviation, as in the paper).

| Dataset | CoT (no RAG) | MedRAG, RRF-2 (k=32) | Change | Paired test (exact McNemar) |
|---|---|---|---|---|
| MedQA-US (n=1,273) | 61.67 ± 1.36 | 61.12 ± 1.37 | −0.55 | p = 0.73 (148 fixed, 155 broken) |
| MMLU-Med (n=1,089) | 78.42 ± 1.25 | 74.75 ± 1.32 | **−3.67** | **p = 0.006** (83 fixed, 123 broken) |

Strict accuracy (unparsable answers counted wrong): MedQA 61.67 / 60.57, MMLU 78.42 / 74.66. Unparsable answers: MedQA 3 / 24, MMLU 0 / 10. With RAG, 22 (MedQA) and 7 (MMLU) answers hit the length limit; most of these hit MedRAG's 8,192-token total limit, because the context alone is about 7,200 tokens.

**Reference numbers from the paper** ([arXiv:2402.13178](https://arxiv.org/abs/2402.13178), Tables 6 and 7):

| Paper setting | MMLU-Med: CoT → MedRAG | MedQA-US: CoT → MedRAG |
|---|---|---|
| GPT-3.5, Textbooks, RRF-2 (Table 7) | 72.91 → 76.68 (+3.77) | 65.04 → 65.91 (+0.87) |
| GPT-4, MedCorp, RRF-4 (Table 6) | 89.44 → 87.24 (−2.20) | 83.97 → 82.80 (−1.17) |
| Mixtral 8×7B, MedCorp, RRF-4 (Table 6) | 74.01 → 75.85 (+1.84) | 64.10 → 60.02 (−4.08) |
| PMC-LLaMA-13B, MedCorp, RRF-4 (Table 6) | 52.16 → 52.53 (+0.37) | 44.38 → 42.58 (−1.80) |

**Reading.** We do **not** reproduce the gain the paper reports for GPT-3.5 with Textbooks + RRF-2. With Qwen2.5-7B, RAG gives no change on MedQA-US and a significant drop on MMLU-Med. The paper's own Table 6 shows that on these two exam-style datasets the effect of RAG depends on the model and is negative for several (GPT-4, Llama2-70B, and PMC-LLaMA and Mixtral on MedQA), so our result is consistent with that part of the paper. Two explanations were considered: a 7B model being distracted by long contexts, and retrieved text that adds little beyond what the model already knows. The number-of-snippets sweep below tests the first.

#### Number of snippets (k)

![Accuracy vs. number of snippets](docs/k_sweep.png)

| k | 1 | 2 | 4 | 8 | 16 | 32 | CoT |
|---|---|---|---|---|---|---|---|
| MedQA-US | 58.92 | 57.89 | 57.50 | 58.68 | 61.12 | 61.12 | 61.67 |
| MMLU-Med | 72.64 | 72.45 | 74.29 | 73.83 | 76.12 | 74.75 | 78.42 |

Paired exact McNemar test against CoT (p-values; 6 tests per dataset, so the Bonferroni threshold is 0.0083): MedQA k=1…32: 0.037, 0.005, 0.002, 0.026, 0.72, 0.73; MMLU-Med: <0.0001, <0.0001, 0.002, 0.0004, 0.081, 0.006.

- Accuracy **rises** with k, which matches the paper's Figure 3 (roughly log-linear growth for k ≤ 32, GPT-3.5, MedCorp, RRF-4).
- But with this model, RAG **never beats CoT**. Few snippets hurt the most, and 16–32 snippets only bring accuracy back to about the CoT level.
- So the "distracted by long context" explanation is **not supported**: more context helps, it does not hurt. A more likely reading is that the model leans on the few retrieved passages even when they are not the right evidence, and more snippets dilute that effect. This is a hypothesis from these numbers, not yet tested directly; a per-question check of whether the retrieved snippets contain the answer would test it.

#### Does the context contain the answer? (retrieval-quality proxy)

MIRAGE has no gold evidence for these datasets, so `scripts/retrieval_quality.py` uses a lexical proxy: an option counts as present when its text occurs in one of the top-k snippets. Questions are grouped by what the context contains, and CoT and RAG answers are compared inside each group (full table: `results/retrieval_quality/retrieval_quality.md`).

| Group (MedQA-US, k=32) | share | CoT | RAG | fixed / broken | p (McNemar) |
|---|---|---|---|---|---|
| only the correct option appears | 5.7% | 71.2 | 75.3 | 6 / 3 | 0.51 |
| only wrong options appear | 14.8% | 58.7 | 50.8 | 16 / 31 | **0.04** |
| no option appears | 71.0% | 60.2 | 59.7 | 111 / 115 | 0.84 |

- **The model follows the context.** When only wrong options appear, RAG is significantly worse than CoT on both datasets and at both k = 2 and k = 32 (p between 0.001 and 0.04). It picks a wrong option that is in the context more often than CoT does (MedQA k=32: 31.7% vs 24.9%; MMLU-Med: 21.4% vs 13.0%). When only the correct option appears, RAG tends to help, but these groups are too small to be significant.
- **Retrieval rarely finds the answer.** With 32 snippets the correct option appears in only 14% (MedQA) and 19% (MMLU-Med) of contexts. For most questions no option appears at all. In that group RAG is significantly worse at k = 2 (MedQA p = 0.012, MMLU-Med p < 0.001) and the harm shrinks with more snippets, which explains the rising k-curve.
- **Conclusion.** For this model the bottleneck is retrieval relevance (Textbooks with question-only retrieval), not context length. This agrees with the paper's finding that the choice of corpus matters most (StatPearls was best for MedQA-US in its Table 7). A manual reading of the 25 MedQA cases where RAG switched from a correct CoT answer to a wrong option present in the context (`results/retrieval_quality/manual_review_medqa_k32.md`) found that 80% of the snippets are on the question's topic but describe a neighbouring diagnosis or treatment ("near-miss" retrieval), 16% are incidental mentions, and 1 case is a proxy error. Caveats: lexical presence is not evidence (a snippet may mention a drug to rule it out), long sentence-like options (MMLU-Med) are rarely matched, and several groups and tests are compared.

**Cost.** On 2×T4 (Kaggle free tier), CoT takes about 0.3–0.5 s per question, and RAG with 32 snippets about 13.4 s per question (~8.8 GPU hours for both datasets); the k = 1…16 sweep took 0.4–5.6 s per question (~6.2 GPU hours). T4 GPUs support only the Triton/Flex attention backends in vLLM 0.33; a diagnosis of other settings is in `results/speed_diagnosis/`.

### Persian extension (work in progress)

`scripts/persian/prepare_fa_mmlu.py` downloads [FA_MED_MMLU](https://huggingface.co/datasets/gaokerena/FA_MED_MMLU) (Persian MMLU-Med, Ghassabi et al. 2025), checks it against MIRAGE and writes it in MIRAGE's format to `data_local/` (not committed: the dataset card states no licence). Check result: same 6 subjects and 1,089 items, and the answer key agrees for **all 1,089** items in order (a misaligned file would agree about 25%), so Persian and English results can be compared question by question. Numbers survive translation in 343 of 453 questions that contain them; the sampled misses are formatting (“2,300” → “2300”, “2 months” → “دو ماه”). Report: `results/persian/fa_mmlu_alignment.json`.

**Persian corpus.** `scripts/persian/chunk_corpus.py` chunks [gaokerena/medical_corpus](https://huggingface.co/datasets/gaokerena/medical_corpus) with MedRAG's Textbooks settings (LangChain `RecursiveCharacterTextSplitter`, 1,000 characters, 200 overlap; checked in MedRAG `src/data/textbooks.py`): 101,954 documents → 2,173 exact duplicates removed → **541,926 snippets** (4.3× the English Textbooks). Unlike Textbooks, it is a **web health corpus**: 64% of snippets come from hidoctor.ir and 17% from niniban.com (pregnancy and parenting). Stats: `results/persian/fa_corpus_chunk_stats.json`.

**Persian BM25.** `scripts/persian/bm25_fa.py` indexes it with Lucene's PersianAnalyzer (`--language fa`; the searcher also needs `set_language("fa")`, otherwise a Persian index returns no hits) and retrieves for FA_MED_MMLU with question-only queries (indexing 31 s, 1,089 queries 9 s on CPU).

**First RQ1 measurement (answer coverage, BM25, same 1,089 questions; `scripts/persian/answer_coverage.py`):**

| k | English, Textbooks: correct option present | Persian, web corpus: correct option present |
|---|---|---|
| 1 | 5.2% | 2.4% |
| 8 | 13.3% | 6.3% |
| 32 | 17.8% | 10.3% |

Persian retrieval finds the correct option's text about half as often. Two confounds must be separated before this is a finding: the corpora differ (textbooks vs web articles), and exact phrase matching is harder in Persian (morphology, spelling variants, translated options), so part of the gap may be the measurement. Planned controls: the same proxy on Persian↔English translated pairs of snippets, and multilingual dense retrieval.

**Dense and fused Persian retrieval.** BGE-M3 (`BAAI/bge-m3`, MIT; [CLS], L2-normalised) embeddings of all 541,926 snippets were computed on 2×T4 in 47 min (`kaggle/embed_fa/`; the kernel re-chunks the corpus from the Hub and checks it against the local fingerprint instead of uploading it). RRF-2 here fuses BM25 and BGE-M3 with MedRAG's formula, because MedCPT is English-only. Answer coverage (correct option's text in the top-32; same 1,089 MMLU-Med questions in both languages, same normalisation):

| Retriever | English, Textbooks | Persian, web corpus | KOPP (Persian) |
|---|---|---|---|
| BM25 | 17.8% | 10.3% | 13.2% |
| dense (MedCPT / BGE-M3) | 17.7% | 10.9% | 13.7% |
| RRF-2 | 18.5% | 11.8% | 14.6% |

The two Persian retrievers overlap in only 13% of their top-32 (English BM25/MedCPT: 12.5%), and fusion helps in both languages. A multilingual dense retriever does **not** close the gap, which points to the corpus rather than the retriever, though the measurement confound (exact matching is harder in Persian) remains. Planned control that removes it: Persian questions → English Textbooks with BGE-M3, scored on the aligned English options.

**Cross-lingual control (2×2).** To separate question language from corpus, the same 1,089 questions were run in both languages against both corpora with BGE-M3 (`kaggle/xling/`, `scripts/persian/xling_coverage.py`). Coverage is always scored in the corpus language with the aligned options, so within a column the measurement is identical and only the question language changes:

| Correct option present in top-32 | English Textbooks | Persian web corpus |
|---|---|---|
| English question | 20.1% | 10.3% (cross-lingual) |
| Persian question | 18.1% (cross-lingual) | 10.9% |
| paired test, English vs Persian question | p = 0.005 | p = 0.30 |

Asking in Persian instead of English costs about 2 points on Textbooks and nothing measurable on the Persian corpus: a multilingual retriever handles the question language well. The large gap is between the columns (about 8–10 points), i.e. the corpus, together with the remaining measurement difference (English vs Persian matching), which this design cannot separate further. For RQ1 this points to the knowledge source as the main lever; a direct test would be Persian questions with an English or translated textbook corpus in the full RAG pipeline.

**Data fix.** 184 of 1,878 KOPP questions (copied from PDFs) wrap every word in Unicode bidi control marks, which made BM25 return nothing for 115 of them. `medrag_repro.persian.clean_text` replaces the marks with spaces (some words are separated only by them); questions with no BM25 hit dropped from 115 to 3. The corpus has the marks in 1.7% of snippets; it is left unchanged for now so BM25 and the existing BGE-M3 embeddings index the same text, and will be cleaned at the next embedding run. Query embeddings were recomputed on CPU after the fix (same top-1 as the GPU run for 1,074 of 1,089 FA_MED_MMLU questions; 99% top-10 overlap).

**Leakage check (RQ4).** `scripts/persian/leakage_check.py` looks for benchmark questions copied into the Persian corpus (which Gaokerena was trained on and our RAG retrieves from): word 8-gram and 13-gram overlap of each question stem with every corpus document after Persian normalisation (Arabic yeh/kaf, ZWNJ, digits); a question is flagged if ≥ 50% of its n-grams occur in one document.

| Benchmark | checkable (≥ 8 words) | flagged (8-gram) | highest overlap with one document |
|---|---|---|---|
| KOPP (Iranian licensing exam, 1,878) | 1,743 | **0** | 20% |
| FA_MED_MMLU (1,089) | 983 | **0** | 25% |

Positive control: 20 benchmark questions planted into a copy of corpus documents (some with Arabic-yeh spelling) were all found when checkable (17 of 17 with ≥ 8 words; the 3 misses had 4–7 words), with no false positives. So there is **no verbatim leakage** from these benchmarks into the corpus. Not covered: paraphrased or translated copies, and questions shorter than 8 words (7% of KOPP, 10% of FA_MED_MMLU). Report: `results/persian/leakage.json`.



**Persian RAG accuracy (Qwen2.5-7B-Instruct, k = 4).** `kaggle/run_fa/`, analysis `scripts/persian/analyze_fa_runs.py` → `results/persian/run_fa/summary.json` (per-question predictions are committed as ids and letters only). Prompts are MedRAG's English instructions with Persian question, options and snippets; RRF-2 = BM25 + BGE-M3 over the Persian web corpus. k = 4 rather than 32 because Persian snippets take about twice as many tokens (k = 4 ≈ 2,300 prompt tokens, k = 32 ≈ 7,500) and k = 32 did not fit this week's GPU quota.

| | CoT (no retrieval) | RRF-2, k = 4 | RAG fixes / breaks | paired test |
|---|---|---|---|---|
| FA_MED_MMLU (1,089) | 59.14% | 53.35% | 143 / 206 | p < 0.001 |
| KOPP (1,878) | 51.65% | 47.55% | 286 / 363 | p = 0.003 |
| English MMLU-Med, same items (Textbooks) | 78.42% | 74.29% | | |

- **Language gap:** the same 1,089 items lose 19.3 points in Persian without retrieval (278 questions correct only in English, 68 only in Persian; p < 0.0001), and 20.9 points with RAG at k = 4.
- **RAG hurts in Persian** on both benchmarks (−5.8 and −4.1 points); in English at k = 4 it also hurts, by −4.1 points.
- **It depends on what is retrieved.** Split by whether the correct option's text is in the 4 snippets (lexical proxy above): on KOPP, RAG **helps** when it is (147 questions: 58.5% → 71.4%, p = 0.003) and hurts when it is not (1,696: 50.7% → 45.2%, p < 0.001). On FA_MED_MMLU only 61 questions have it (82.0% → 80.3%, n.s.). The proxy misses paraphrased evidence, so "absent" means "not found verbatim", not "no useful content".
- Reading: with this corpus, retrieval mostly adds distraction because the answer is rarely retrieved; this is consistent with the 2×2 control pointing at the knowledge source.

**2×2 RAG accuracy (question language × corpus; BGE-M3, k = 4, Qwen2.5-7B-Instruct).** The direct test proposed above: the same 1,089 items, each language against each corpus, in the full pipeline (`kaggle/xling_rag/`, `scripts/persian/analyze_xling_rag.py` → `results/persian/xling_rag/summary.json`; 2.3 GPU-hours).

| Accuracy | English Textbooks | Persian web corpus | paired test (corpus) | CoT, no retrieval |
|---|---|---|---|---|
| English question | **73.92%** | 70.43% (cross-lingual) | p = 0.015 | 78.42% |
| Persian question | 47.75% (cross-lingual) | **52.98%** | p = 0.004 | 59.14% |
| paired test (question language) | p < 0.0001 | p < 0.0001 | | |

- **Interaction, not a single best corpus.** English questions do better with Textbooks (+3.5), Persian questions do better with the Persian corpus (+5.2). The worst cell is a Persian question with English Textbooks, although its retrieval finds the correct option's text more often than the Persian corpus does (102 vs 65 questions at k = 4). A context in a different language from the question costs more than the better corpus gains.
- This **revises the reading of the 2×2 coverage control**: the corpus is the main lever for *retrieval* coverage, but for *accuracy* the language match between question and context matters as much. Translating the question or the snippets is the natural next test.
- **RAG below CoT in every cell** at k = 4 (paired test against CoT p < 0.001 in every cell). Where the correct option's text is in the 4 snippets, accuracy is close to or above CoT (e.g. Persian/Persian: 78.5% → 84.6%, 65 questions); where it is not, RAG loses 5–12 points. With these corpora and k = 4 the answer is rarely retrieved (5–11% of questions), so distraction dominates.
- Consistency check: cell 1 (BGE-M3, 73.92%) is close to the English RRF-2 result at k = 4 (74.29%).

**Error analysis (no GPU; `scripts/persian/error_analysis.py` → `results/persian/error_analysis.json`).**

- *Gap by subject (CoT, English vs Persian, same items):* anatomy −26.7, medical genetics −26.0, college biology −25.7, professional medicine −18.4, clinical knowledge −15.1, college medicine −12.1 points. The gap is largest where answers hinge on technical terms (anatomy, genetics, biology).
- *Reasoning language:* given a Persian question, Qwen2.5-7B-Instruct almost never reasons in Persian (≤ 7 of 1,089–1,878 answers per run). It reasons in English or **in Chinese**: 24% of FA_MED_MMLU answers without retrieval, 61% with the Persian corpus, 75% with English Textbooks; 45% and 59% on KOPP. With English questions it never switches. The answer letter is still parsed correctly (the JSON key is Latin). Accuracy is similar either way (e.g. FA_MED_MMLU CoT: English reasoning 57.5%, Chinese 64.3%; with Textbooks 45.1% vs 48.6%), and the language is the model's choice, so this is a description, not a causal effect. It does show that the long cross-lingual prompt pushes the model further from the question language. A prompt that fixes the reasoning language (e.g. "think in English") is a cheap next test.
- *Manual check of 12 random questions answered correctly only in English* (seed 0): the translation is faithful in 10; in 2 it loses something ("Marked microsatellite instability" loses "marked"; "Li-Fraumeni syndrome" is rendered with a word meaning "membership"). In the sampled items, then, the gap comes mostly from the model's Persian, not from translation errors (small sample).

**Translate first (Persian question → English with Qwen itself, then the English pipeline).** `kaggle/translate/`, `scripts/persian/translate_questions.py`, analysis `scripts/persian/analyze_translate.py` → `results/persian/translate/summary.json` (0.75 GPU-hours). Qwen translates question and options (greedy, 0.26 s/question; 13 of 1,089 outputs could not be parsed and stay in Persian, counted as they are). BGE-M3 retrieves from Textbooks with the translated question; CoT and k = 4 as in the 2×2.

| Same 1,089 items | CoT | Textbooks, k = 4 |
|---|---|---|
| English original | 78.42% | 73.92% |
| **Persian, machine-translated to English** | **66.85%** | **63.54%** |
| Persian as is | 59.14% | 47.75% (Persian corpus: 52.98%) |

- Translating first is the best Persian pipeline: +7.7 points without retrieval and +15.8 with Textbooks over asking in Persian, and +10.6 over the Persian corpus (all p < 0.0001). It recovers about 40% of the language gap without retrieval.
- **Retrieval is fully recovered**: the correct option's text is in the top 4 for 123 questions with the translated question vs 124 with the English original. What remains is in reading and answering.
- **Translation quality matters but does not explain the whole gap.** Round-trip chrF against the original English (FA_MED_MMLU is itself a translation of it) is 67.0 for questions and 68.3 for options. By question-chrF tercile the gap to English is 14.9, 10.5 and 9.3 points. Part of the loss is in the Persian dataset's own translation, which the round trip cannot undo, and low chrF also goes with longer, harder questions, so this is a correlation.
- RAG still costs 3.3 points against CoT here (p = 0.015). When the correct option's text is retrieved (123 questions), RAG is on par (81.3% → 82.1%).

### Citation

```bibtex
@inproceedings{xiong-etal-2024-benchmarking,
  title  = {Benchmarking Retrieval-Augmented Generation for Medicine},
  author = {Xiong, Guangzhi and Jin, Qiao and Lu, Zhiyong and Zhang, Aidong},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  year   = {2024}
}
```

---

<div dir="rtl">

## فارسی

### هدف
این مخزن نتایج مقالهٔ «Benchmarking Retrieval-Augmented Generation for Medicine» (Xiong و همکاران، ۲۰۲۴) را به صورت مستقل بازتولید می‌کند. اجرا روی سخت‌افزار رایگان است: CPU برای بازیابی، و GPU رایگان T4 کاگل برای مدل زبانی.

### چه چیزی بازتولید می‌شود (مرحلهٔ ۱)
- **بنچمارک:** MedQA-US (۱٬۲۷۳ سؤال) و MMLU-Med (۱٬۰۸۹ سؤال) از MIRAGE
- **منبع دانش:** کتاب‌های درسی (Textbooks)، شامل ۱۸ کتاب و ۱۲۵٬۸۴۷ قطعه. جدول ۳ مقاله این عدد را به شکل گردشده ۱۲۵.۸ هزار آورده است.
- **روش‌های بازیابی:** BM25، MedCPT، و ترکیب این دو با RRF-2
- **مدل‌های زبانی:** Qwen2.5-7B-Instruct به عنوان مدل اصلی، و PMC-LLaMA-13B که در خود مقاله هم آمده و مقایسهٔ مستقیم را ممکن می‌کند.

### وفاداری به کد اصلی
- promptها کلمه‌به‌کلمه از کد رسمی MedRAG کپی شده‌اند.
- کوتاه‌سازی context، فرمول RRF و روش استخراج جواب دقیقاً مطابق کد رسمی است.
- کد رسمی وقتی جوابی پیدا نکند `"A"` برمی‌گرداند. به همین دلیل **دقت سخت‌گیرانه** و **تعداد جواب‌های نامفهوم** را هم جداگانه گزارش می‌کنیم تا این رفتار نتیجه را گمراه‌کننده نکند.
- کد رسمی Public Domain (متعلق به NCBI) است. commitهای مرجع در بخش انگلیسی ثبت شده‌اند.

### تفاوت‌ها با مقاله (شفاف و ثبت‌شده)
۱. **مدل‌ها:** GPT-3.5 و GPT-4 به API پولی نیاز دارند و مدل‌های 70B روی دو T4 جا نمی‌شوند.
۲. **embeddingهای MedCPT دوباره ساخته شدند:** لینک رسمی در تاریخ ۸ نوامبر ۲۰۲۶ خطای 403 داد. embeddingها با همان مدل و همان تنظیمات دوباره ساخته شدند.
۳. **سقف ۱۰۲۴ توکن برای طول جواب** تا زمان GPU محدود بماند. تعداد جواب‌هایی که به این سقف رسیده‌اند ثبت می‌شود.
۴. **fp16 به‌جای bf16**، چون GPU مدل T4 از bf16 پشتیبانی نمی‌کند.
۵. **vLLM** به‌جای pipeline هاگینگ‌فیس، برای سرعت بیشتر. روش تولید متن در هر دو حالت greedy است.

### نتایج
**مدل Qwen2.5-7B-Instruct، منبع دانش Textbooks، ۳۲ قطعه** (کل سؤال‌های آزمون، با روش رسمی MIRAGE برای استخراج جواب؛ ± انحراف معیار دوجمله‌ای است، مثل خود مقاله)

| دیتاست | بدون RAG (CoT) | با RAG (RRF-2، ۳۲ قطعه) | تغییر | آزمون آماری زوجی (McNemar دقیق) |
|---|---|---|---|---|
| MedQA-US (۱٬۲۷۳ سؤال) | ۶۱.۶۷ ± ۱.۳۶ | ۶۱.۱۲ ± ۱.۳۷ | ۰.۵۵− | p = 0.73 (۱۴۸ درست‌شده، ۱۵۵ خراب‌شده) |
| MMLU-Med (۱٬۰۸۹ سؤال) | ۷۸.۴۲ ± ۱.۲۵ | ۷۴.۷۵ ± ۱.۳۲ | **۳.۶۷−** | **p = 0.006** (۸۳ درست‌شده، ۱۲۳ خراب‌شده) |

**مقایسه با مقاله:** مقاله برای GPT-3.5 با Textbooks و RRF-2 بهبود گزارش کرده است: در MMLU-Med از ۷۲.۹۱ به ۷۶.۶۸، و در MedQA از ۶۵.۰۴ به ۶۵.۹۱ (جدول ۷). ما این بهبود را با Qwen2.5-7B **بازتولید نکردیم**. ولی جدول ۶ خود مقاله نشان می‌دهد که روی این دو دیتاست امتحانی، اثر RAG به مدل بستگی دارد و برای چند مدل منفی است: برای GPT-4، Llama2-70B، و برای PMC-LLaMA و Mixtral روی MedQA. پس نتیجهٔ ما با این بخش از مقاله سازگار است.

**دو توضیح احتمالی بررسی شد:** یکی اینکه context طولانی حواس مدل 7B را پرت می‌کند، و دیگری اینکه متن بازیابی‌شده چیزی بیش از دانسته‌های خود مدل اضافه نمی‌کند. آزمایش تعداد قطعه‌ها در ادامه، توضیح اول را می‌سنجد.

#### تعداد قطعه‌ها (k)

| k | 1 | 2 | 4 | 8 | 16 | 32 | بدون RAG |
|---|---|---|---|---|---|---|---|
| MedQA-US | ۵۸.۹۲ | ۵۷.۸۹ | ۵۷.۵۰ | ۵۸.۶۸ | ۶۱.۱۲ | ۶۱.۱۲ | ۶۱.۶۷ |
| MMLU-Med | ۷۲.۶۴ | ۷۲.۴۵ | ۷۴.۲۹ | ۷۳.۸۳ | ۷۶.۱۲ | ۷۴.۷۵ | ۷۸.۴۲ |

- دقت با **بیشتر شدن k بالا می‌رود.** این روند با شکل ۳ مقاله سازگار است: رشد تقریباً لگاریتمی تا k=32، که مقاله با GPT-3.5 و MedCorp و RRF-4 گزارش کرده است.
- ولی با این مدل، RAG **در هیچ kای از حالت بدون RAG بهتر نشد.** کم بودن قطعه‌ها بیشترین ضرر را دارد. با ۱۶ تا ۳۲ قطعه، دقت فقط به حدود سطح بدون RAG برمی‌گردد.
- پس فرضیهٔ «context طولانی حواس مدل را پرت می‌کند» **تأیید نشد**: context بیشتر کمک می‌کند، نه ضرر.
- توضیح محتمل‌تر این است که مدل به چند قطعهٔ بازیابی‌شده تکیه می‌کند، حتی وقتی آن قطعه‌ها شاهد درستی نیستند. با قطعه‌های بیشتر، این اثر رقیق می‌شود. این هم فقط فرضیه‌ای است که از این اعداد برداشت شده و هنوز مستقیم آزمایش نشده.
- نمودار و آزمون‌های آماری (McNemar با اصلاح Bonferroni) در بخش انگلیسی آمده‌اند.

#### آیا context جواب را دارد؟ (معیار تقریبی کیفیت بازیابی)
MIRAGE برای این دیتاست‌ها شاهد درست (gold evidence) ندارد. برای همین `scripts/retrieval_quality.py` از یک معیار تقریبی واژگانی استفاده می‌کند: یک گزینه «حاضر» است اگر متنش در یکی از k قطعهٔ بازیابی‌شده آمده باشد.

- **مدل از context پیروی می‌کند:**
  - وقتی فقط گزینه‌های غلط در context هستند، RAG در هر دو دیتاست و هر دو k (یعنی 2 و 32) به شکل معنادار بدتر از حالت بدون RAG است (p بین ۰.۰۰۱ و ۰.۰۴).
  - در این حالت، مدل بیشتر گزینهٔ غلطی را انتخاب می‌کند که در متن آمده است. مثلاً در MedQA با k=32، این نرخ ۳۱.۷٪ با RAG است، در برابر ۲۴.۹٪ بدون RAG.
- **بازیابی به‌ندرت جواب را پیدا می‌کند:**
  - با ۳۲ قطعه، گزینهٔ درست فقط در ۱۴٪ (MedQA) و ۱۹٪ (MMLU-Med) از contextها هست.
  - در بیشتر سؤال‌ها هیچ گزینه‌ای در متن نیست. در این گروه، RAG با k=2 به شکل معنادار ضرر می‌زند و با قطعه‌های بیشتر این ضرر کم می‌شود. همین روند صعودی منحنی k را توضیح می‌دهد.
- **نتیجه:** برای این مدل، گلوگاه مرتبط بودن بازیابی است، نه طول context. این با یافتهٔ مقاله سازگار است که «انتخاب منبع دانش مهم‌ترین عامل است».
- **بررسی دستی:** ۲۵ سؤال MedQA را خواندم که در آن‌ها RAG جواب درستِ بدون RAG را به گزینهٔ غلطِ داخل متن تغییر داد. نتیجه:
  - در ۸۰٪ موارد، قطعه دربارهٔ همان موضوع سؤال است، ولی تشخیص یا درمانِ مشابهِ دیگری را توضیح می‌دهد (بازیابیِ «نزدیک ولی اشتباه»).
  - در ۱۶٪ موارد، گزینه فقط گذرا نام برده شده.
  - یک مورد خطای خود معیار بود.
- **محدودیت‌ها:** حضور واژگانی یک گزینه به معنای شاهد بودن آن نیست؛ گزینه‌های جمله‌ای MMLU کمتر پیدا می‌شوند؛ و چندین گروه و آزمون با هم مقایسه شده‌اند. جدول و آزمون‌ها در بخش انگلیسی آمده‌اند.

**هزینه:** روی دو T4 رایگان کاگل، CoT حدود ۰.۳ تا ۰.۵ ثانیه برای هر سؤال طول می‌کشد و RAG با ۳۲ قطعه حدود ۱۳.۴ ثانیه، یعنی حدود ۸.۸ ساعت GPU برای هر دو دیتاست. آزمایش k = 1 تا 16 بین ۰.۴ تا ۵.۶ ثانیه برای هر سؤال طول کشید، جمعاً حدود ۶.۲ ساعت GPU.

### توسعهٔ فارسی (در حال انجام)
- اسکریپت `scripts/persian/prepare_fa_mmlu.py` دیتاست FA_MED_MMLU (ترجمهٔ فارسی MMLU-Med) را دانلود می‌کند، با MIRAGE مقایسه می‌کند و در قالب MIRAGE ذخیره می‌کند.
- داده در مخزن قرار نمی‌گیرد، چون صفحهٔ دیتاست مجوزی اعلام نکرده است.
- **نتیجهٔ بررسی:**
  - همان ۶ موضوع و ۱٬۰۸۹ سؤال.
  - کلید جواب برای **هر ۱٬۰۸۹ سؤال** به همان ترتیب یکسان است. اگر فایل‌ها هم‌تراز نبودند، حدود ۲۵٪ تطابق انتظار می‌رفت.
  - پس نتایج فارسی و انگلیسی را می‌شود سؤال‌به‌سؤال مقایسه کرد.
- اعداد در ۳۴۳ از ۴۵۳ سؤالِ عددی عیناً حفظ شده‌اند. نمونه‌هایی که بررسی شد فقط تفاوت قالب‌بندی داشتند، مثل «2,300» که «2300» شده، یا «2 months» که «دو ماه» ترجمه شده.
- **منبع دانش فارسی:** با همان تنظیمات قطعه‌بندی MedRAG (۱٬۰۰۰ نویسه با ۲۰۰ نویسه هم‌پوشانی)، ۱۰۱٬۹۵۴ سند پس از حذف ۲٬۱۷۳ سند تکراری به **۵۴۱٬۹۲۶ قطعه** تبدیل شد. این ۴.۳ برابر Textbooks انگلیسی است. برخلاف Textbooks، این منبع **مقاله‌های سلامتِ وب** است: ۶۴٪ قطعه‌ها از hidoctor.ir و ۱۷٪ از niniban.com (بارداری و کودک).
- **BM25 فارسی:** با تحلیلگر فارسی Lucene ساخته شد. نکتهٔ مهم: موقع جست‌وجو هم باید `set_language("fa")` تنظیم شود، وگرنه هیچ نتیجه‌ای برنمی‌گردد.
- **اولین اندازه‌گیری برای RQ1:** روی همان ۱٬۰۸۹ سؤال و با BM25، متن گزینهٔ درست در ۳۲ قطعهٔ اول در **۱۷.۸٪** سؤال‌های انگلیسی (Textbooks) و **۱۰.۳٪** سؤال‌های فارسی (منبع وب) پیدا شد؛ یعنی حدود نصف.
- **دو عامل مخدوش‌کننده** را باید قبل از نتیجه‌گیری جدا کرد:
  - منبع‌ها فرق دارند: کتاب درسی در برابر مقالهٔ وب.
  - تطابق عینی عبارت در فارسی سخت‌تر است: صرف کلمات، تنوع املایی، و ترجمه‌ای بودن گزینه‌ها. پس بخشی از این فاصله ممکن است از خود معیار باشد.
- **بازیابی معنایی و ترکیبی فارسی:** embedding همهٔ ۵۴۱٬۹۲۶ قطعه با BGE-M3 روی دو T4 در ۴۷ دقیقه ساخته شد. چون MedCPT فقط انگلیسی است، RRF-2 در فارسی ترکیب BM25 و BGE-M3 است. پوشش جواب در ۳۲ قطعهٔ اول:
  - انگلیسی (Textbooks): BM25 ۱۷.۸٪، MedCPT ۱۷.۷٪، RRF-2 ۱۸.۵٪
  - فارسی (منبع وب): BM25 ۱۰.۳٪، BGE-M3 ۱۰.۹٪، RRF-2 ۱۱.۸٪
  - KOPP: BM25 ۱۳.۲٪، BGE-M3 ۱۳.۷٪، RRF-2 ۱۴.۶٪

  بازیابی معنایی چندزبانه **شکاف را نمی‌بندد**. این نشان می‌دهد عامل اصلی بیشتر منبع دانش است تا روش بازیابی. ولی عامل مخدوش‌کنندهٔ معیار هنوز باقی است.
- **آزمایش کنترلی ۲×۲:** همان ۱٬۰۸۹ سؤال، به هر دو زبان، روی هر دو منبع با BGE-M3 جست‌وجو شد. پوشش همیشه به زبان خودِ منبع سنجیده شد. نتیجه، یعنی سهم سؤال‌هایی که متن گزینهٔ درست در ۳۲ قطعهٔ اول آمده:
  - Textbooks انگلیسی: با سؤال انگلیسی ۲۰.۱٪، با سؤال فارسی ۱۸.۱٪
  - منبع وب فارسی: با سؤال انگلیسی ۱۰.۳٪، با سؤال فارسی ۱۰.۹٪
  - **زبان سؤال** فقط حدود ۲ واحد اثر دارد (روی منبع فارسی هم اثر معناداری ندارد).
  - **منبع دانش** (همراه با تفاوت زبانِ معیار) حدود ۸ تا ۱۰ واحد اثر دارد.
  - پس در RQ1، اهرم اصلی **منبع دانش** است.
- **اصلاح داده:** در ۱۸۴ سؤال KOPP، که از PDF کپی شده‌اند، هر کلمه بین نویسه‌های نامرئی جهت متن قرار گرفته بود. به همین دلیل BM25 برای ۱۱۵ سؤال هیچ نتیجه‌ای برنمی‌گرداند. بعد از پاک‌سازی، این عدد به ۳ رسید.
- **بررسی نشت داده (RQ4):** هم‌پوشانی ۸ و ۱۳ کلمه‌ای متن هر سؤال با تک‌تک سندهای منبع فارسی سنجیده شد. نتیجه: **هیچ سؤالی** از KOPP یا FA_MED_MMLU عیناً در منبع نیامده است. بیشترین هم‌پوشانی با یک سند ۲۰ تا ۲۵٪ بود، در حالی که آستانهٔ علامت‌گذاری ۵۰٪ است.
  - **کنترل مثبت:** ۲۰ سؤال را عمداً در سندهای ساختگی کاشتم، بعضی با املای عربیِ «ي». همهٔ سؤال‌های قابل بررسی (۱۷ از ۱۷) پیدا شدند و هیچ هشدار اشتباهی هم نبود.
  - **محدودیت:** نسخه‌های بازنویسی‌شده یا ترجمه‌شده، و سؤال‌های کوتاه‌تر از ۸ کلمه، با این روش بررسی نمی‌شوند.

- **دقت RAG فارسی (Qwen2.5-7B-Instruct، k = 4):** پوشهٔ `kaggle/run_fa/`، تحلیل با `scripts/persian/analyze_fa_runs.py` و نتیجه در `results/persian/run_fa/summary.json`. از پیش‌بینی‌ها فقط شناسه و حرف گزینه در مخزن است. k = 4 به جای ۳۲ انتخاب شد، چون قطعه‌های فارسی حدود دو برابر توکن می‌گیرند و k = 32 در سهمیهٔ GPU این هفته جا نمی‌شد.
  - FA_MED_MMLU (۱٬۰۸۹ سؤال): بدون بازیابی **۵۹.۱۴٪**، با RRF-2 **۵۳.۳۵٪**. RAG در ۱۴۳ سؤال جواب را درست کرد و در ۲۰۶ سؤال خراب کرد (p < 0.001).
  - KOPP (۱٬۸۷۸ سؤال): بدون بازیابی **۵۱.۶۵٪**، با RRF-2 **۴۷.۵۵٪**. ۲۸۶ اصلاح در برابر ۳۶۳ خرابی (p = 0.003).
  - **شکاف زبانی:** همان ۱٬۰۸۹ سؤال به انگلیسی ۷۸.۴۲٪ و به فارسی ۵۹.۱۴٪، یعنی **۱۹.۳ واحد** افت. ۲۷۸ سؤال فقط به انگلیسی درست جواب داده شد و ۶۸ سؤال فقط به فارسی (p < 0.0001). با RAG در k = 4 این افت ۲۰.۹ واحد است.
  - **وابستگی به محتوای بازیابی‌شده:** در KOPP، وقتی متن گزینهٔ درست در ۴ قطعه هست (۱۴۷ سؤال)، RAG دقت را از ۵۸.۵٪ به **۷۱.۴٪** می‌رساند (p = 0.003). وقتی نیست (۱٬۶۹۶ سؤال)، دقت از ۵۰.۷٪ به ۴۵.۲٪ می‌افتد. در FA_MED_MMLU فقط ۶۱ سؤال چنین قطعه‌ای دارند و تفاوت معنادار نیست.
  - **محدودیت:** این معیار فقط تطابق عینی را می‌بیند. «نبودن» یعنی متن عیناً پیدا نشد، نه اینکه هیچ اطلاعات مفیدی نبود.
  - **برداشت:** با این منبع، جواب به‌ندرت بازیابی می‌شود و بازیابی بیشتر حواس مدل را پرت می‌کند. این با نتیجهٔ آزمایش ۲×۲ هم‌خوان است که منبع دانش را عامل اصلی نشان داد.

- **آزمایش ۲×۲ دقت (زبان سؤال × منبع؛ BGE-M3، k = 4):** همان آزمونِ مستقیمی که بالاتر پیشنهاد شده بود، این بار در کل خط RAG. کد در `kaggle/xling_rag/` و `scripts/persian/analyze_xling_rag.py`، نتیجه در `results/persian/xling_rag/summary.json`. هزینه ۲.۳ ساعت GPU بود.
  - سؤال انگلیسی: با Textbooks **۷۳.۹۲٪**، با منبع فارسی ۷۰.۴۳٪ (p = 0.015). بدون بازیابی: ۷۸.۴۲٪.
  - سؤال فارسی: با Textbooks ۴۷.۷۵٪، با منبع فارسی **۵۲.۹۸٪** (p = 0.004). بدون بازیابی: ۵۹.۱۴٪.
  - **اثر متقابل:** هیچ منبعی برای هر دو زبان بهترین نیست. سؤال انگلیسی با کتاب انگلیسی بهتر است و سؤال فارسی با منبع فارسی. بدترین حالت، سؤال فارسی با کتاب انگلیسی است، با اینکه بازیابی در این حالت جواب را بیشتر پیدا می‌کند (۱۰۲ سؤال در برابر ۶۵). یعنی هم‌زبان نبودن سؤال و متن بازیابی‌شده بیشتر از کیفیت بهتر منبع هزینه دارد.
  - **اصلاح برداشت قبلی:** آزمایش ۲×۲ پوشش نشان داد منبع عامل اصلی *پوشش بازیابی* است. اما برای *دقت*، هم‌زبان بودن سؤال و متن به همان اندازه مهم است. آزمایش طبیعی بعدی ترجمهٔ سؤال یا ترجمهٔ قطعه‌هاست.
  - **در هر چهار حالت، RAG با k = 4 از حالت بدون بازیابی بدتر است.** وقتی متن گزینهٔ درست در ۴ قطعه هست، دقت نزدیک یا بالاتر از حالت بدون بازیابی است (مثلاً فارسی با فارسی: ۷۸.۵٪ به ۸۴.۶٪ در ۶۵ سؤال). وقتی نیست، ۵ تا ۱۲ واحد افت دارد. چون جواب فقط در ۵ تا ۱۱٪ سؤال‌ها بازیابی می‌شود، اثر منفی غالب است.
  - **بررسی سازگاری:** حالت ۱ (۷۳.۹۲٪) به نتیجهٔ انگلیسی RRF-2 با k = 4 (۷۴.۲۹٪) نزدیک است.

- **تحلیل خطا (بدون GPU):** کد `scripts/persian/error_analysis.py`، نتیجه در `results/persian/error_analysis.json`.
  - **شکاف زبانی به تفکیک موضوع** (بدون بازیابی، همان سؤال‌ها): آناتومی −۲۶.۷، ژنتیک پزشکی −۲۶.۰، زیست‌شناسی −۲۵.۷، پزشکی حرفه‌ای −۱۸.۴، دانش بالینی −۱۵.۱، پزشکی عمومی −۱۲.۱ واحد. شکاف جایی بیشتر است که جواب به اصطلاح تخصصی وابسته است.
  - **زبان استدلال مدل:** Qwen با سؤال فارسی تقریباً هیچ‌وقت به فارسی استدلال نمی‌کند (حداکثر ۷ پاسخ در هر اجرا). یا به انگلیسی استدلال می‌کند یا **به چینی**. سهم چینی در FA_MED_MMLU: ۲۴٪ بدون بازیابی، ۶۱٪ با منبع فارسی، ۷۵٪ با Textbooks. در KOPP: ۴۵٪ و ۵۹٪. با سؤال انگلیسی هرگز به چینی نمی‌رود.
    - حرف جواب همچنان درست خوانده می‌شود.
    - دقت در دو حالت نزدیک است. مثلاً بدون بازیابی: استدلال انگلیسی ۵۷.۵٪، چینی ۶۴.۳٪.
    - چون زبان را خود مدل انتخاب می‌کند، این فقط یک توصیف است و اثر علّی نیست.
    - آزمایش ارزان بعدی: دستوری که زبان استدلال را ثابت کند، مثلاً «به انگلیسی فکر کن».
  - **بررسی دستی ۱۲ سؤال تصادفی که فقط به انگلیسی درست جواب گرفتند:** ترجمه در ۱۰ سؤال درست بود. در ۲ سؤال چیزی از دست رفته بود، مثلاً «Li-Fraumeni» با کلمه‌ای به معنی «عضویت» ترجمه شده بود. پس در این نمونهٔ کوچک، شکاف بیشتر از ضعف فارسیِ مدل است تا خطای ترجمه.

- **اول ترجمه، بعد جواب:** Qwen خودش سؤال فارسی و گزینه‌ها را به انگلیسی ترجمه می‌کند و بعد خط انگلیسی اجرا می‌شود. کد در `kaggle/translate/`، `scripts/persian/translate_questions.py` و `scripts/persian/analyze_translate.py`، نتیجه در `results/persian/translate/summary.json`. هزینه ۰.۷۵ ساعت GPU بود. ترجمه ۰.۲۶ ثانیه برای هر سؤال طول کشید. ۱۳ ترجمه از ۱٬۰۸۹ قابل خواندن نبود و همان متن فارسی نگه داشته شد، یعنی هیچ سؤالی حذف نشد.
  - بدون بازیابی: انگلیسی اصلی ۷۸.۴۲٪، **ترجمه‌شده ۶۶.۸۵٪**، فارسی ۵۹.۱۴٪.
  - با Textbooks و k = 4: انگلیسی اصلی ۷۳.۹۲٪، **ترجمه‌شده ۶۳.۵۴٪**، فارسی ۴۷.۷۵٪. با منبع فارسی: ۵۲.۹۸٪.
  - **بهترین روش برای سؤال فارسی، اول ترجمه کردن است.** نسبت به پرسیدن به فارسی، بدون بازیابی ۷.۷ واحد و با Textbooks ۱۵.۸ واحد بهتر است. نسبت به منبع فارسی هم ۱۰.۶ واحد بهتر است. همهٔ این تفاوت‌ها معنادارند (p < 0.0001). بدون بازیابی، حدود ۴۰٪ شکاف زبانی جبران می‌شود.
  - **بازیابی کامل جبران شد.** متن گزینهٔ درست با سؤال ترجمه‌شده در ۱۲۳ سؤال بازیابی شد و با سؤال انگلیسی اصلی در ۱۲۴ سؤال. پس افت باقی‌مانده در مرحلهٔ فهم و جواب دادن است، نه در بازیابی.
  - **کیفیت ترجمه مهم است، ولی همهٔ شکاف را توضیح نمی‌دهد.** شباهت ترجمهٔ برگشتی با متن اصلی انگلیسی (chrF) برای سؤال ۶۷.۰ و برای گزینه‌ها ۶۸.۳ است. در سه گروه سؤال، از کیفیت پایین تا بالا، فاصله با انگلیسی ۱۴.۹، ۱۰.۵ و ۹.۳ واحد است. بخشی از افت از ترجمهٔ خود دیتاست فارسی است و ترجمهٔ برگشتی آن را جبران نمی‌کند. سؤال‌های طولانی‌تر و سخت‌تر هم chrF پایین‌تری دارند، پس این فقط یک همبستگی است.
  - RAG در این حالت هم ۳.۳ واحد از حالت بدون بازیابی بدتر است (p = 0.015). وقتی جواب بازیابی شده (۱۲۳ سؤال)، دو حالت برابرند (۸۱.۳٪ و ۸۲.۱٪).

</div>
