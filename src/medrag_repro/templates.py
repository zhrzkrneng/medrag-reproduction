"""Prompt templates, copied verbatim from the official MedRAG code.

Source: https://github.com/Teddy-XiongGZ/MedRAG/blob/7599a728a28789fd601728c08d313b1148051f41/src/template.py
(public-domain NCBI software). Only the two templates used for MIRAGE
(chain-of-thought and MedRAG) are kept; the Liquid templates are rewritten as
plain ``str.format`` strings with identical text.
"""

COT_SYSTEM = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

COT_PROMPT = '''
Here is the question:
{question}

Here are the potential choices:
{options}

Please think step-by-step and generate your output in json:
'''

MEDRAG_SYSTEM = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

MEDRAG_PROMPT = '''
Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
{options}

Please think step-by-step and generate your output in json:
'''


def format_options(options: dict) -> str:
    """Same as MedRAG: 'A. ...' lines sorted by option key."""
    return "\n".join(f"{k}. {options[k]}" for k in sorted(options))


def format_snippet(idx: int, snippet: dict) -> str:
    """Same as MedRAG: 'Document [i] (Title: t) content'."""
    return "Document [{:d}] (Title: {:s}) {:s}".format(idx, snippet["title"], snippet["content"])


def build_messages(question: str, options: dict, context: str | None, system_suffix: str = "") -> list[dict]:
    """system_suffix: optional extra instruction appended to the system prompt (an ablation; empty = MedRAG)."""
    opts = format_options(options)
    if context is None:
        return [
            {"role": "system", "content": COT_SYSTEM + system_suffix},
            {"role": "user", "content": COT_PROMPT.format(question=question, options=opts)},
        ]
    return [
        {"role": "system", "content": MEDRAG_SYSTEM + system_suffix},
        {"role": "user", "content": MEDRAG_PROMPT.format(context=context, question=question, options=opts)},
    ]


# --- Gaokerena's own multiple-choice prompt (RQ2 robustness check) ---------------------------------
# Copied from github.com/Mehrdadghassabi/Gaokerena-V (commit 13eb1f7), evaluation/multiple_choice_qa/
# */Untitled2.ipynb, `prompt_eval_bare` (the prompt used for every reported Gaokerena / Aya / Qwen
# result there, NSHOT = 0). It is a single user message, no system prompt; options as "[A] : text".
# Their f-string with `{{question}} \n` gives the exact newlines below; checked against the
# `model_prompt` column of their result files.
GK_PROMPT = "\nبه پرسش زیر به پاسخ بدهید و تنها از زبان فارسی استفاده کنید!\n## پرسش\n{question} \n\n\n## گزینه ها\n{choices} \n\n\n\n\n"
# RAG variant (ours; Gaokerena reports no RAG result). The context section and its heading
# "## متن" follow their unused `prompt_eval_with_context_and_examples`; snippets are formatted as in MedRAG.
GK_PROMPT_RAG = "\nبه پرسش زیر به پاسخ بدهید و تنها از زبان فارسی استفاده کنید!\n## متن\n{context} \n\n## پرسش\n{question} \n\n\n## گزینه ها\n{choices} \n\n\n\n\n"


def gk_format_choices(options: dict) -> str:
    """Their format_choices: '[A] : text' lines in A-D order."""
    return "\n".join(f"[{k}] : {options[k]}" for k in options)


def build_messages_gk(question: str, options: dict, context: str | None) -> list[dict]:
    ch = gk_format_choices(options)
    if context is None:
        return [{"role": "user", "content": GK_PROMPT.format(question=question, choices=ch)}]
    return [{"role": "user", "content": GK_PROMPT_RAG.format(context=context, question=question, choices=ch)}]
