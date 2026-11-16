# Related work and novelty check (2026-11-16)

A first search, not a systematic review. Each claim below was checked against the paper's arXiv
abstract page, and, where marked, against the full text. Search terms: Persian / Farsi medical RAG,
Persian medical QA benchmark, multilingual medical RAG, cross-lingual medical evidence,
translate-test.

## Closest prior work

| Work | What it does | RAG? | Persian? | Difference from ours |
|---|---|---|---|---|
| **PersianMedQA**, Ranjbar Kalahroodi et al., arXiv:2506.00250 (v4, 2026-05), LREC 2026 | 20,785 Iranian exam MCQs; 41 LLMs, zero-shot and CoT; Persian vs. English (Gemini translation) | **No** (full text checked; its conclusion names "retrieval-augmented systems" as future work) | Yes | Closed-book only. Reports Qwen2.5-7B and Aya-Expanse-8B, and that Persian fine-tunes (Dorna) are weak; 3–10% of questions are answered correctly only in Persian. |
| **Gaokerena-V**, Ghassabi et al., arXiv:2505.16000 | Persian medical corpus + MF3QA; LoRA on Aya-Expanse-8B | No (mentions RAG only as related work) | Yes | We reuse its corpus as the retrieval source and reproduce its MCQ numbers (within 0.6 points). |
| **MedRAG / MIRAGE**, Xiong et al., ACL Findings 2024 | Medical RAG benchmark and toolkit | Yes | No (English) | The study we reproduce and extend. |
| **MedExpQA**, Alonso, Oronoz, Agerri, arXiv:2404.05590 (AI in Medicine 2024) | Medical MCQ in EN/ES/FR/IT with doctors' explanations; RAG with MedRAG-style RRF snippets | Yes | No | Finds RAG far below gold explanations, and lower scores outside English. |
| **Effects of Cross-lingual Evidence in Multilingual Medical QA**, Yeginbergen, Oronoz, Agerri, arXiv:2604.20531 (2026-04) | Evidence source (curated / web / LLM-generated) × retrieval language, for EN, ES, FR, IT, Basque, Kazakh | Yes | **No** | Closest design to our 2×2. For low-resource languages, mixing English and target-language retrieval works best. Persian is a new language for this question. |
| **MKG-Rank**, arXiv:2503.16131 | Word-level translation + UMLS knowledge-graph retrieval for ZH, JA, KO, SW | KG retrieval | No | Different retrieval type; argues against full-text translation. |
| **Med-CoReasoner**, arXiv:2601.08267 | Parallel English + local-language reasoning to cut language gaps | Partly | (not checked) | Related to our "think in English" ablation. |
| **Is Translation All You Need?**, Liu et al., NAACL 2025 (arXiv:2403.10258) | Translate-test vs. native prompting, general tasks | No | Partly | General-domain evidence that translate-first helps English-centric models, except for culture-bound tasks. |
| **Persian RAG** (MatinaSRoberta), arXiv:2501.04858; **Persian university RAG**, arXiv:2411.06237 | General-domain Persian RAG | Yes | Yes | Not medical. |

## What looks new here (to be re-checked before submission)

1. A MedRAG/MIRAGE-style RAG benchmark for **Persian medical** MCQ, item-aligned with the English
   MMLU-Med subset, so that the language gap is measured on the same questions. No such Persian
   study was found.
2. The **question-language × corpus-language** 2×2 for Persian (prior 2×2-style work: 2604.20531,
   without Persian).
3. **Retrieval vs. Persian medical fine-tuning** on the same questions (RQ2), including the finding
   that the fine-tune's advantage depends on the prompt.
4. A leakage check of benchmarks against the fine-tuning data (RQ4).

## Still to check

- Google Scholar / Semantic Scholar for 2026 Persian medical RAG papers (only web search so far).
- ACL Anthology 2026 (LREC, ACL, EMNLP workshops: BioNLP, ClinicalNLP, MRL, LoResLM).
- Whether PersianMedQA's v4 added any retrieval experiment after the version read here (v3 text).
