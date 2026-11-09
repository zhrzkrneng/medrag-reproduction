"""Text clean-up for Persian benchmark text.

Some KOPP questions (copied from PDFs) wrap every word in Unicode bidirectional control marks
(U+202A-U+202E, U+2066-U+2069, LRM/RLM). Lucene treats them as part of the token, so BM25 finds
nothing for these questions. Words are sometimes separated only by these marks, so each mark is
replaced by a space (not deleted), then whitespace is collapsed.
"""
import re

BIDI = re.compile("[‎‏‪-‮⁦-⁩]")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", BIDI.sub(" ", text or "")).strip()
