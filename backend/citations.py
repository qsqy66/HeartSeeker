"""Citation parsing and quote-level verification.

A page number alone is a weak claim: it proves the page exists, not that the
answer is supported by it.  The agent is asked to return an exact quote for
every citation, and this module checks that the quote actually occurs in the
retrieved evidence for that page.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

PAGE_RE = re.compile(r"\[Page\s+(\d+)\]", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def normalize_quote(text: str) -> str:
    """Normalize text for quote matching without changing its wording."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"-\s*\n\s*", "", text)  # join hyphenated line breaks
    text = SPACE_RE.sub("", text)
    return re.sub(r"[^\w㐀-鿿]+", "", text)


def extract_pages(text: str) -> list[int]:
    return sorted({int(number) for number in PAGE_RE.findall(text or "")})


@dataclass
class Citation:
    page: int
    quote: str = ""
    paper_id: Optional[int] = None
    claim: str = ""
    verified: bool = False
    level: str = "page"  # "quote" | "page"


@dataclass
class ValidationResult:
    ok: bool
    citations: list[Citation] = field(default_factory=list)
    invalid: list[dict[str, Any]] = field(default_factory=list)
    missing_pages: list[int] = field(default_factory=list)
    level: str = "none"  # "quote" | "page" | "none"
    warnings: list[str] = field(default_factory=list)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {"page": item.page, "paper_id": item.paper_id, "quote": item.quote,
             "claim": item.claim, "verified": item.verified, "level": item.level}
            for item in self.citations
        ]


def _evidence_index(evidence: Iterable[dict[str, Any]]) -> dict[tuple[Optional[int], int], list[str]]:
    index: dict[tuple[Optional[int], int], list[str]] = {}
    for row in evidence:
        page = row.get("page_number", row.get("page"))
        if page is None:
            continue
        try:
            page = int(page)
        except (TypeError, ValueError):
            continue
        paper_id = row.get("paper_id")
        try:
            paper_id = int(paper_id) if paper_id is not None else None
        except (TypeError, ValueError):
            paper_id = None
        index.setdefault((paper_id, page), []).append(row.get("text") or "")
    return index


def _pages_by_paper(index: dict[tuple[Optional[int], int], list[str]]) -> dict[Optional[int], set[int]]:
    pages: dict[Optional[int], set[int]] = {}
    for (paper_id, page) in index:
        pages.setdefault(paper_id, set()).add(page)
    return pages


def _available_pages(pages_by_paper: dict[Optional[int], set[int]],
                     paper_id: Optional[int]) -> set[int]:
    if paper_id in pages_by_paper:
        return pages_by_paper[paper_id]
    # Single-paper evidence often omits paper_id; it is still unambiguous.
    if len(pages_by_paper) == 1:
        return next(iter(pages_by_paper.values()))
    return set()


def validate_citations(answer: str, raw_citations: Optional[list[dict[str, Any]]],
                       evidence: list[dict[str, Any]],
                       default_paper_id: Optional[int] = None,
                       require_citation: bool = False) -> ValidationResult:
    index = _evidence_index(evidence)
    pages_by_paper = _pages_by_paper(index)
    citations: list[Citation] = []
    invalid: list[dict[str, Any]] = []
    missing: list[int] = []

    for raw in raw_citations or []:
        if not isinstance(raw, dict):
            invalid.append({"reason": "引用格式无效", "raw": raw})
            continue
        page = raw.get("page", raw.get("page_number"))
        try:
            page = int(page)
        except (TypeError, ValueError):
            invalid.append({"reason": "引用缺少页码", "raw": raw})
            continue
        paper_id = raw.get("paper_id", default_paper_id)
        try:
            paper_id = int(paper_id) if paper_id is not None else None
        except (TypeError, ValueError):
            paper_id = default_paper_id
        quote = str(raw.get("quote") or "").strip()
        claim = str(raw.get("claim") or "").strip()
        citation = Citation(page=page, quote=quote, paper_id=paper_id, claim=claim)
        available = _available_pages(pages_by_paper, paper_id)

        if page not in available:
            citation.verified = False
            missing.append(page)
            invalid.append({"page": page, "paper_id": paper_id, "reason": "页码不在检索证据中",
                            "quote": quote})
            citations.append(citation)
            continue

        if quote and len(normalize_quote(quote)) >= 6:
            page_text = "\n".join(index.get((paper_id, page), []))
            if normalize_quote(quote) in normalize_quote(page_text):
                citation.verified = True
                citation.level = "quote"
            else:
                # Quotes occasionally span two adjacent chunks from the same page.
                paper_text = "\n".join(
                    text for (row_paper, _), texts in index.items()
                    if row_paper in (paper_id, None) for text in texts
                )
                if normalize_quote(quote) in normalize_quote(paper_text):
                    citation.verified = True
                    citation.level = "quote"
                else:
                    invalid.append({"page": page, "paper_id": paper_id,
                                    "reason": "引文未在证据原文中找到", "quote": quote})
        else:
            citation.verified = True
            citation.level = "page"
        citations.append(citation)

    if not citations:
        answer_pages = extract_pages(answer)
        for page in answer_pages:
            paper_id = default_paper_id
            available = _available_pages(pages_by_paper, paper_id)
            citation = Citation(page=page, paper_id=paper_id, verified=page in available)
            if not citation.verified:
                missing.append(page)
                invalid.append({"page": page, "paper_id": paper_id, "reason": "页码不在检索证据中"})
            citations.append(citation)

    verified = [item for item in citations if item.verified]
    level = "none"
    if any(item.level == "quote" for item in verified):
        level = "quote"
    elif verified:
        level = "page"

    ok = not invalid and (bool(verified) or not require_citation)
    warnings: list[str] = []
    if require_citation and not verified:
        warnings.append("回答没有可核验的论文引用。")
    if invalid:
        warnings.append("部分引用未能通过原文核验，已标记。")

    return ValidationResult(ok=ok, citations=citations, invalid=invalid,
                            missing_pages=sorted(set(missing)), level=level,
                            warnings=warnings)


def citation_footer(citations: list[Citation], include_quote: bool = True) -> str:
    lines: list[str] = []
    for citation in citations:
        if not citation.verified:
            continue
        label = f"[Page {citation.page}]"
        if citation.paper_id is not None:
            label = f"[Paper {citation.paper_id} · Page {citation.page}]"
        if include_quote and citation.quote:
            lines.append(f"{label} “{citation.quote[:160]}”")
        else:
            lines.append(label)
    return "\n".join(lines)
