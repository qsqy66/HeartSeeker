"""Local, dependency-optional tokenization for Chinese + English retrieval.

SQLite's ``unicode61`` tokenizer treats a run of Chinese characters as one
token, so FTS5 alone cannot rank Chinese queries.  This module produces a
space-separated token stream stored in ``chunks.search_text`` and indexed by
``chunks_fts_seg``; the same function is used for queries, so document and
query tokens always agree.

``jieba`` is used when installed.  Without it a deterministic CJK bigram
fallback keeps retrieval useful and makes tests reproducible.
"""
from __future__ import annotations

import re
from typing import Iterable

try:  # optional: pure-Python, small, but not required
    import jieba as _jieba  # type: ignore

    _jieba.setLogLevel(60)
except Exception:  # pragma: no cover - depends on the environment
    _jieba = None

CJK_RE = re.compile(r"[㐀-鿿]")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|\d+(?:\.\d+)?|[㐀-鿿]+")


def _cjk_run_tokens(run: str) -> Iterable[str]:
    if len(run) == 1:
        yield run
        return
    # The whole run supports exact phrase matches; bigrams support partial
    # Chinese queries such as "心衰" against "心力衰竭".
    yield run
    for index in range(len(run) - 1):
        yield run[index:index + 2]


def _fallback_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text or ""):
        value = match.group(0)
        if CJK_RE.search(value):
            tokens.extend(_cjk_run_tokens(value))
        else:
            tokens.append(value.lower())
    return tokens


def tokenize(text: str) -> list[str]:
    """Return normalized, deduplicated tokens in first-seen order."""
    text = text or ""
    if _jieba is not None:
        raw = [str(token) for token in _jieba.cut(text) if str(token).strip()]
        tokens: list[str] = []
        for token in raw:
            if CJK_RE.search(token) and len(token) > 1:
                tokens.extend(_cjk_run_tokens(token))
            elif not CJK_RE.search(token):
                tokens.append(token.lower())
            else:
                tokens.append(token)
    else:
        tokens = _fallback_tokens(text)
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def segment(text: str) -> str:
    """Render text as a whitespace-delimited token stream for FTS5."""
    return " ".join(tokenize(text))


def terms(text: str) -> set[str]:
    return set(tokenize(text))
