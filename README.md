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

**Reading.** We do **not** reproduce the gain the paper reports for GPT-3.5 with Textbooks + RRF-2. With Qwen2.5-7B, RAG gives no change on MedQA-US and a significant drop on MMLU-Med. The paper's own Table 6 shows that on these two exam-style datasets the effect of RAG depends on the model and is negative for several (GPT-4, Llama2-70B, and PMC-LLaMA and Mixtral on MedQA), so our result is consistent with that part of the paper. Possible reasons, **not yet tested**: Qwen2.5 is a newer and stronger model whose CoT accuracy is already high (possible exposure to these public benchmarks during pre-training), and a 7B model may be distracted by 7,000 tokens of context. Planned checks: the number-of-snippets sweep (k = 1…16) and PMC-LLaMA-13B (the paper's model).

**Cost.** On 2×T4 (Kaggle free tier), CoT takes about 0.3–0.5 s per question, and RAG with 32 snippets about 13.4 s per question (~8.8 GPU hours for both datasets). T4 GPUs support only the Triton/Flex attention backends in vLLM 0.33; a diagnosis of other settings is in `results/speed_diagnosis/`.

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

**دلایل احتمالی (هنوز آزمایش نشده):** Qwen2.5 مدل جدیدتر و قوی‌تری است و دقت CoT آن از قبل بالاست. ممکن است این بنچمارک‌های عمومی را در پیش‌آموزش دیده باشد. همچنین ممکن است یک مدل 7B با ۷ هزار توکن context حواسش پرت شود. آزمایش تعداد قطعه‌ها (k = 1 تا 16) و اجرای PMC-LLaMA-13B (مدل خود مقاله) برای بررسی همین برنامه‌ریزی شده‌اند.

**هزینه:** روی دو T4 رایگان کاگل، CoT حدود ۰.۳ تا ۰.۵ ثانیه برای هر سؤال طول می‌کشد و RAG با ۳۲ قطعه حدود ۱۳.۴ ثانیه، یعنی حدود ۸.۸ ساعت GPU برای هر دو دیتاست.

</div>
