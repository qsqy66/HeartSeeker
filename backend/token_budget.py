"""Conservative prompt token budgeting for long agent transcripts.

Exact tokenizers differ by provider, so this module uses a deliberately cheap
estimate: CJK characters cost roughly one token each, other text about one
token per four characters.  The budget is enforced by truncating older
observations and, if needed, dropping the oldest middle turns while always
keeping the system prompt and the original user question.
"""
from __future__ import annotations

import re
from typing import Any

CJK_RE = re.compile(r"[㐀-鿿]")
DEFAULT_MAX_TOKENS = 24000
MIN_KEEP_TOKENS = 1200


def estimate_tokens(text: Any) -> int:
    value = str(text or "")
    if not value:
        return 0
    cjk = len(CJK_RE.findall(value))
    return cjk + max(0, (len(value) - cjk + 3) // 4)


def estimate_messages(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(message.get("content")) + 8 for message in messages)


def _truncate(message: dict[str, str], limit_chars: int) -> dict[str, str]:
    content = str(message.get("content") or "")
    if len(content) <= limit_chars:
        return message
    return {**message, "content": content[:limit_chars] + "…[已截断]"}


def fit_messages(messages: list[dict[str, str]], max_tokens: int = DEFAULT_MAX_TOKENS,
                 keep_last: int = 3) -> list[dict[str, str]]:
    """Return a copy of ``messages`` within ``max_tokens`` (best effort).

    The first system message and the first user message are never dropped.
    Recent messages stay intact; older assistant JSON and observations are
    truncated first, then the oldest middle turns are removed.
    """
    if max_tokens <= 0:
        return [dict(message) for message in messages]
    copied = [dict(message) for message in messages]
    if estimate_messages(copied) <= max_tokens:
        return copied

    head = copied[:2]
    tail = copied[2:]
    # Truncate old turns from the front of the tail, preserving recent context.
    for index, message in enumerate(tail):
        if index >= max(0, len(tail) - keep_last):
            break
        limit = 600 if message.get("role") == "assistant" else 1200
        tail[index] = _truncate(message, limit)
        if estimate_messages(head + tail) <= max_tokens:
            return head + tail

    while len(tail) > keep_last and estimate_messages(head + tail) > max_tokens:
        tail.pop(0)

    # Last resort: hard-truncate the most recent contexts, never the question.
    if estimate_messages(head + tail) > max_tokens:
        for index in range(len(tail)):
            tail[index] = _truncate(tail[index], 400)
    if estimate_messages(head + tail) > max_tokens:
        budget = max(MIN_KEEP_TOKENS, max_tokens - estimate_messages(head))
        kept: list[dict[str, str]] = []
        for message in tail:
            if estimate_messages(kept + [message]) > budget:
                break
            kept.append(message)
        tail = kept or tail[-1:]
    return head + tail
