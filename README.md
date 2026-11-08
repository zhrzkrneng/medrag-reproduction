# medrag-reproduction

Reproduction of “Benchmarking Retrieval-Augmented Generation for Medicine” (Xiong et al., 2024): the MedRAG toolkit on the MIRAGE benchmark, scaled down to run on free hardware (CPU, plus the free Kaggle T4 GPU tier).

[English](#english) · [فارسی](#فارسی)

---

## English

### What is reproduced

| Component | Paper | This repo, phase 1 |
|---|---|---|
| Benchmark | MIRAGE, 7,663 questions in 5 datasets | MedQA-US (1,273) + MMLU-Med (1,089) |
| Corpus | PubMed, StatPearls, Textbooks, Wikipedia, MedCorp | **Textbooks** (18 books, 125,829 snippets; same as paper Table 3) |
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
scripts/            retrieve.py · embed_corpus.py · generate.py · evaluate.py
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

*Pending. Filled in after the Kaggle runs.*

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
- **منبع دانش:** کتاب‌های درسی (Textbooks)، شامل ۱۸ کتاب و ۱۲۵٬۸۲۹ قطعه. این عددها دقیقاً با جدول ۳ مقاله یکی است.
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
*بعد از اجرا روی کاگل اضافه می‌شود.*

</div>
