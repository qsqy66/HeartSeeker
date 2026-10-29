"""Hybrid retrieval over one paper or the whole library.

The application stores two complementary indexes:

* SQLite FTS5 for literal terms (strong for English medical jargon)
* LanceDB vectors for semantic recall (strong for paraphrases and Chinese)

This module fuses both rankings with Reciprocal Rank Fusion (RRF), keeps the
source rank for each hit, and returns enough metadata for page/bbox provenance.
It deliberately has no FastAPI dependency so it can be unit tested directly.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from . import rerank as rerank_module
from . import tokenization
from .credentials import safe_key
from .embedding import DEFAULT_EMBED_MODEL, embed_texts, normalize, search_chunks
from .medical_terms import expand_query

RRF_K = 60
CJK_RE = re.compile(r"[㐀-鿿]")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|\d+(?:\.\d+)?")
TERM_SPLIT_RE = re.compile(r"[\s,;，；。！？、/|()（）\[\]【】]+")
# Chinese questions have no word delimiters.  Removing common interrogative and
# function phrases yields useful LIKE terms ("主要结局是什么" -> "主要结局"),
# and bigrams provide a fallback when the phrase is longer.
CJK_STOP_RE = re.compile(
    r"(?:是什么|什么|是多少|多少|哪些|哪个|如何|怎么|为什么|是否|有没有|"
    r"患者|病人|研究|试验|论文|文章|请问|告诉|一下)"
)


def connect(db_path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def _like(term: str) -> str:
    return "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _cjk_like_terms(query: str, limit: int = 24) -> list[str]:
    """Split a Chinese question into searchable phrases and short n-grams."""
    terms: list[str] = []
    for piece in TERM_SPLIT_RE.split(query):
        for sub in CJK_STOP_RE.split(piece):
            sub = sub.strip().lstrip("的了")
            if len(sub) < 2:
                continue
            terms.append(sub)
            if len(sub) <= 8:
                terms.extend(sub[index:index + 2] for index in range(len(sub) - 1))
    return _dedupe(terms, limit=limit)


def _dedupe(values: Iterable[str], limit: int = 24) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = value.strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _fts_rows(connection: sqlite3.Connection, fts_query: str,
              paper_id: Optional[int], limit: int) -> list[sqlite3.Row]:
    clause = "AND c.paper_id = ?" if paper_id is not None else ""
    params: list[Any] = [fts_query]
    if paper_id is not None:
        params.append(paper_id)
    try:
        return connection.execute(
            "SELECT c.id, c.paper_id, c.page_id, c.chunk_index, c.text, c.section, c.heading, "
            "c.x0, c.y0, c.x1, c.y1, p.page_number, papers.title AS paper_title "
            "FROM chunks_fts JOIN chunks c ON chunks_fts.rowid = c.id "
            "JOIN pages p ON p.id = c.page_id JOIN papers ON papers.id = c.paper_id "
            f"WHERE chunks_fts MATCH ? {clause} "
            "ORDER BY bm25(chunks_fts), p.page_number, c.chunk_index LIMIT ?",
            [*params, limit],
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def _seg_fts_rows(connection: sqlite3.Connection, tokens: list[str],
                  paper_id: Optional[int], limit: int) -> list[sqlite3.Row]:
    """BM25 over the segmented index (real ranking for Chinese queries)."""
    if not tokens:
        return []
    escaped = [token.replace('"', '""') for token in tokens]
    seg_query = " OR ".join(f'"{token}"' for token in escaped)
    clause = "AND c.paper_id = ?" if paper_id is not None else ""
    params: list[Any] = [seg_query]
    if paper_id is not None:
        params.append(paper_id)
    try:
        return connection.execute(
            "SELECT c.id, c.paper_id, c.page_id, c.chunk_index, c.text, c.section, c.heading, "
            "c.x0, c.y0, c.x1, c.y1, p.page_number, papers.title AS paper_title "
            "FROM chunks_fts_seg JOIN chunks c ON chunks_fts_seg.rowid = c.id "
            "JOIN pages p ON p.id = c.page_id JOIN papers ON papers.id = c.paper_id "
            f"WHERE chunks_fts_seg MATCH ? {clause} "
            "ORDER BY bm25(chunks_fts_seg), p.page_number, c.chunk_index LIMIT ?",
            [*params, limit],
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def _like_rows(connection: sqlite3.Connection, terms: list[str],
               paper_id: Optional[int], limit: int) -> list[sqlite3.Row]:
    if not terms:
        return []
    like_clause = " OR ".join(["c.text LIKE ? ESCAPE '\\'"] * len(terms))
    where = f"({like_clause})"
    params: list[Any] = [_like(term) for term in terms]
    if paper_id is not None:
        where += " AND c.paper_id = ?"
        params.append(paper_id)
    return connection.execute(
        "SELECT c.id, c.paper_id, c.page_id, c.chunk_index, c.text, c.section, c.heading, "
        "c.x0, c.y0, c.x1, c.y1, p.page_number, papers.title AS paper_title "
        "FROM chunks c JOIN pages p ON p.id = c.page_id JOIN papers ON papers.id = c.paper_id "
        f"WHERE {where} ORDER BY p.page_number, c.chunk_index LIMIT ?",
        [*params, limit],
    ).fetchall()


def keyword_search(query: str, db_path: Path | str, paper_id: Optional[int] = None,
                   limit: int = 30) -> list[dict[str, Any]]:
    """Literal search with bilingual expansion and a CJK-friendly LIKE path.

    SQLite's unicode61 tokenizer treats a run of Chinese characters as one
    token, so FTS5 alone misses most Chinese queries.  English terms go through
    FTS5; Chinese terms and their synonyms go through LIKE.  Results from both
    paths are merged, which lets "心衰" find an English paper that says "heart
    failure" and vice versa.
    """
    query = query.strip()
    if not query:
        return []
    expansions = expand_query(query)
    all_terms = [query, *expansions]
    fts_tokens = _dedupe(TOKEN_RE.findall(" ".join(all_terms)))
    fts_query = " OR ".join(f'"{token}"' for token in fts_tokens) if fts_tokens else ""
    seg_tokens = _dedupe(tokenization.tokenize(" ".join(all_terms)))

    like_terms: list[str] = []
    if _has_cjk(query):
        like_terms.extend(_cjk_like_terms(query))
    like_terms.extend(term for term in all_terms if _has_cjk(term))
    like_terms = _dedupe(like_terms, limit=24)

    connection = connect(db_path)
    try:
        fts_rows = _fts_rows(connection, fts_query, paper_id, limit) if fts_query else []
        seg_rows = _seg_fts_rows(connection, seg_tokens, paper_id, limit)
        like_rows = _like_rows(connection, like_terms, paper_id, limit) if like_terms else []
    finally:
        connection.close()

    if _has_cjk(query):
        ordered = seg_rows + like_rows + fts_rows
    else:
        ordered = fts_rows + seg_rows + like_rows
    results: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in ordered:
        chunk_id = int(row["id"])
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        results.append(_row_to_chunk(row, source="keyword"))
        if len(results) >= limit:
            break
    return results


def vector_search(query: str, db_path: Path | str, vectors_dir: Path | str,
                  paper_id: Optional[int] = None, limit: int = 30,
                  api_key: Optional[str] = None, embedding_model: Optional[str] = None,
                  enable_vectors: bool = True) -> list[dict[str, Any]]:
    if not enable_vectors:
        return []
    key = api_key if api_key is not None else safe_key("qwen")
    if not key:
        return []
    if embedding_model is None:
        embedding_model = _embedding_model(db_path)
    try:
        vector = normalize(embed_texts([query], key, embedding_model)[0])
        rows = search_chunks(Path(vectors_dir), paper_id, vector, limit)
    except Exception:
        return []  # retrieval must degrade to keyword search, never break a request
    return [_row_to_chunk(row, source="vector") for row in rows]


def _embedding_model(db_path: Path | str) -> str:
    try:
        connection = connect(db_path)
        try:
            row = connection.execute(
                "SELECT value FROM settings WHERE key='embedding_model'"
            ).fetchone()
        finally:
            connection.close()
        value = row["value"].strip() if row and row["value"].strip() else ""
        return value or DEFAULT_EMBED_MODEL
    except sqlite3.Error:
        return DEFAULT_EMBED_MODEL


def _row_to_chunk(row: Any, source: str, score: Optional[float] = None) -> dict[str, Any]:
    get = row.get if hasattr(row, "get") else lambda key, default=None: row[key] if key in row.keys() else default
    chunk_id = get("id") if get("id") is not None else get("chunk_id")
    return {
        "id": int(chunk_id) if chunk_id is not None else None,
        "chunk_id": int(chunk_id) if chunk_id is not None else None,
        "paper_id": int(get("paper_id")) if get("paper_id") is not None else None,
        "page_id": int(get("page_id")) if get("page_id") is not None else None,
        "chunk_index": int(get("chunk_index")) if get("chunk_index") is not None else None,
        "page_number": int(get("page_number")) if get("page_number") is not None else None,
        "text": get("text") or "",
        "section": get("section") or "",
        "heading": get("heading") or "",
        "paper_title": get("paper_title") or "",
        "x0": get("x0"), "y0": get("y0"), "x1": get("x1"), "y1": get("y1"),
        "distance": get("_distance"),
        "score": score,
        "sources": [source],
    }


def rrf_fuse(result_lists: Iterable[tuple[str, list[dict[str, Any]]]],
             weights: Optional[dict[str, float]] = None, k: int = RRF_K) -> list[dict[str, Any]]:
    """Fuse ranked lists by reciprocal rank, deduplicating by chunk id."""
    weights = weights or {}
    merged: dict[int, dict[str, Any]] = {}
    text_keys: dict[tuple[Any, str], int] = {}
    for source, rows in result_lists:
        weight = weights.get(source, 1.0)
        for rank, row in enumerate(rows, start=1):
            chunk_id = row.get("id")
            if chunk_id is None:
                continue
            contribution = weight / (k + rank)
            normalized = re.sub(r"\s+", "", str(row.get("text") or "").lower())[:400]
            text_key = (row.get("paper_id"), normalized) if len(normalized) >= 40 else None
            if text_key is not None and text_key in text_keys:
                current = merged[text_keys[text_key]]
            else:
                current = merged.get(chunk_id)
            if current is None:
                current = dict(row)
                current["score"] = 0.0
                current["sources"] = []
                merged[chunk_id] = current
                if text_key is not None:
                    text_keys[text_key] = chunk_id
            current["score"] = float(current.get("score") or 0.0) + contribution
            if source not in current["sources"]:
                current["sources"].append(source)
            if row.get("distance") is not None:
                current["distance"] = row["distance"]
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)


def _canonical_chunks(db_path: Path | str, chunk_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    connection = connect(db_path)
    try:
        rows = connection.execute(
            "SELECT c.id, c.paper_id, c.page_id, c.chunk_index, c.text, c.section, c.heading, "
            "c.x0, c.y0, c.x1, c.y1, p.page_number, papers.title AS paper_title "
            "FROM chunks c JOIN pages p ON p.id = c.page_id JOIN papers ON papers.id = c.paper_id "
            f"WHERE c.id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
    finally:
        connection.close()
    return {int(row["id"]): _row_to_chunk(row, source="canonical") for row in rows}


def _retrieval_options(db_path: Path | str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "rerank_enabled": True,
        "rerank_engine": "lexical",
        "rerank_model": "",
        "rerank_base_url": "",
        "relevance_min_coverage": 0.0,
    }
    try:
        connection = connect(db_path)
        try:
            rows = connection.execute(
                "SELECT key, value FROM settings WHERE key IN "
                "('rerank_enabled','rerank_engine','rerank_model','rerank_base_url',"
                "'relevance_min_coverage')"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return options
    values = {row["key"]: row["value"] for row in rows}
    if "rerank_enabled" in values:
        options["rerank_enabled"] = values["rerank_enabled"].strip().lower() not in {
            "0", "false", "no", "off"}
    options["rerank_engine"] = values.get("rerank_engine", "").strip() or "lexical"
    options["rerank_model"] = values.get("rerank_model", "").strip()
    options["rerank_base_url"] = values.get("rerank_base_url", "").strip()
    try:
        options["relevance_min_coverage"] = float(values.get("relevance_min_coverage") or 0.0)
    except ValueError:
        options["relevance_min_coverage"] = 0.0
    return options


def _neighbor_context(connection: sqlite3.Connection, rows: list[dict[str, Any]],
                      max_chars: int = 2400) -> None:
    """Small-to-big: attach the same-page neighbouring chunks to each hit."""
    for row in rows:
        row["context_text"] = str(row.get("text") or "")
        row["context_chunk_ids"] = [row.get("id")] if row.get("id") is not None else []
        paper_id = row.get("paper_id")
        page_id = row.get("page_id")
        chunk_index = row.get("chunk_index")
        if paper_id is None or page_id is None or chunk_index is None:
            continue
        try:
            neighbours = connection.execute(
                "SELECT id, text FROM chunks WHERE paper_id=? AND page_id=? "
                "AND chunk_index BETWEEN ? AND ? ORDER BY chunk_index",
                (int(paper_id), int(page_id), int(chunk_index) - 1, int(chunk_index) + 1),
            ).fetchall()
        except sqlite3.Error:
            continue
        if len(neighbours) <= 1:
            continue
        pieces: list[str] = []
        used = 0
        for neighbour in neighbours:
            text = str(neighbour["text"] or "").strip()
            if not text:
                continue
            if used + len(text) > max_chars and pieces:
                break
            pieces.append(text)
            used += len(text) + 1
        row["context_text"] = " ".join(pieces)[:max_chars]
        row["context_chunk_ids"] = [int(item["id"]) for item in neighbours]


def _coverage(query: str, row: dict[str, Any]) -> float:
    query_terms = tokenization.terms(query)
    if not query_terms:
        return 0.0
    row_terms = tokenization.terms(" ".join(str(row.get(key) or "") for key in
                                           ("heading", "section", "text")))
    return len(query_terms & row_terms) / len(query_terms)


def hybrid_search(query: str, db_path: Path | str, vectors_dir: Path | str,
                  paper_id: Optional[int] = None, top_k: int = 8, candidate_k: int = 30,
                  api_key: Optional[str] = None, embedding_model: Optional[str] = None,
                  enable_vectors: bool = True,
                  rerank_engine: Optional[str] = None,
                  rerank_model: Optional[str] = None,
                  rerank_base_url: Optional[str] = None,
                  rerank_enabled: Optional[bool] = None,
                  min_coverage: Optional[float] = None,
                  expand_neighbors: bool = True) -> list[dict[str, Any]]:
    """Return the best chunks from both indexes, fused, reranked and enriched."""
    query = query.strip()
    if not query:
        return []
    options = _retrieval_options(db_path)
    if rerank_engine is None:
        rerank_engine = str(options["rerank_engine"])
    if rerank_model is None:
        rerank_model = str(options["rerank_model"])
    if rerank_base_url is None:
        rerank_base_url = str(options["rerank_base_url"])
    if rerank_enabled is None:
        rerank_enabled = bool(options["rerank_enabled"])
    if min_coverage is None:
        min_coverage = float(options["relevance_min_coverage"])

    keyword_rows = keyword_search(query, db_path, paper_id, candidate_k)
    vector_rows = vector_search(
        query, db_path, vectors_dir, paper_id, candidate_k,
        api_key=api_key, embedding_model=embedding_model, enable_vectors=enable_vectors,
    )
    fused = rrf_fuse([("vector", vector_rows), ("keyword", keyword_rows)],
                     weights={"vector": 1.0, "keyword": 0.9})
    fused = fused[:max(candidate_k, top_k)]
    canonical = _canonical_chunks(db_path, [row["id"] for row in fused if row.get("id")])
    results: list[dict[str, Any]] = []
    for row in fused:
        canonical_row = canonical.get(int(row["id"])) if row.get("id") is not None else None
        if canonical_row is None:
            continue
        canonical_row["score"] = round(float(row["score"]), 6)
        canonical_row["sources"] = row["sources"]
        if row.get("distance") is not None:
            canonical_row["distance"] = row["distance"]
        results.append(canonical_row)

    if expand_neighbors and results:
        connection = connect(db_path)
        try:
            _neighbor_context(connection, results)
        finally:
            connection.close()

    if rerank_enabled and results:
        results = rerank_module.rerank(query, results, max(candidate_k, top_k),
                                       engine=rerank_engine, model=rerank_model,
                                       base_url=rerank_base_url)

    if min_coverage and min_coverage > 0:
        gated: list[dict[str, Any]] = []
        for row in results:
            if _coverage(query, row) >= min_coverage:
                gated.append(row)
                continue
            distance = row.get("distance")
            try:
                if distance is not None and float(distance) <= 0.9:
                    gated.append(row)
            except (TypeError, ValueError):
                continue
        results = gated

    return results[:top_k]


def library_search(query: str, db_path: Path | str, vectors_dir: Path | str,
                   top_k: int = 8, **kwargs: Any) -> list[dict[str, Any]]:
    return hybrid_search(query, db_path, vectors_dir, paper_id=None, top_k=top_k, **kwargs)


def read_page(db_path: Path | str, paper_id: int, page_number: int) -> dict[str, Any]:
    connection = connect(db_path)
    try:
        row = connection.execute(
            "SELECT p.page_number, p.text, papers.title AS paper_title "
            "FROM pages p JOIN papers ON papers.id = p.paper_id "
            "WHERE p.paper_id = ? AND p.page_number = ?",
            (paper_id, page_number),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise LookupError("页码不存在")
    return {"paper_id": paper_id, "page_number": int(row["page_number"]),
            "paper_title": row["paper_title"], "text": row["text"]}


def read_section(db_path: Path | str, paper_id: int, heading: str,
                 limit: int = 8) -> list[dict[str, Any]]:
    connection = connect(db_path)
    try:
        rows = connection.execute(
            "SELECT c.id, c.paper_id, c.page_id, c.chunk_index, c.text, c.section, c.heading, "
            "c.x0, c.y0, c.x1, c.y1, p.page_number, papers.title AS paper_title "
            "FROM chunks c JOIN pages p ON p.id = c.page_id JOIN papers ON papers.id = c.paper_id "
            "WHERE c.paper_id = ? AND (c.heading LIKE ? OR c.section LIKE ?) "
            "ORDER BY p.page_number, c.chunk_index LIMIT ?",
            (paper_id, _like(heading), _like(heading), limit),
        ).fetchall()
        if not rows:
            rows = connection.execute(
                "SELECT c.id, c.paper_id, c.page_id, c.chunk_index, c.text, c.section, c.heading, "
                "c.x0, c.y0, c.x1, c.y1, p.page_number, papers.title AS paper_title "
                "FROM chunks c JOIN pages p ON p.id = c.page_id JOIN papers ON papers.id = c.paper_id "
                "WHERE c.paper_id = ? AND c.text LIKE ? ESCAPE '\\' "
                "ORDER BY p.page_number, c.chunk_index LIMIT ?",
                (paper_id, _like(heading), limit),
            ).fetchall()
    finally:
        connection.close()
    return [_row_to_chunk(row, source="keyword") for row in rows]


def compare_papers(db_path: Path | str, vectors_dir: Path | str,
                   paper_ids: list[int], aspects: list[str],
                   top_k_each: int = 4, **kwargs: Any) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    for paper_id in paper_ids[:5]:
        connection = connect(db_path)
        try:
            paper = connection.execute("SELECT title FROM papers WHERE id=?", (paper_id,)).fetchone()
        finally:
            connection.close()
        if paper is None:
            comparison.append({"paper_id": paper_id, "paper_title": "", "aspect": "", "chunks": [],
                               "error": "论文不存在"})
            continue
        for aspect in aspects[:3]:
            chunks = hybrid_search(aspect, db_path, vectors_dir, paper_id=paper_id,
                                   top_k=top_k_each, **kwargs)
            comparison.append({"paper_id": paper_id, "paper_title": paper["title"],
                               "aspect": aspect, "chunks": chunks})
    return comparison


def build_context(rows: list[dict[str, Any]], max_chars: int = 30000,
                  per_row_chars: int = 1800) -> str:
    """Render retrieved chunks into a bounded, clearly delimited context block."""
    parts: list[str] = []
    used = 0
    seen: set[int] = set()
    for row in rows:
        chunk_id = row.get("id")
        if chunk_id is not None and int(chunk_id) in seen:
            continue
        if chunk_id is not None:
            seen.add(int(chunk_id))
        text = (row.get("context_text") or row.get("text") or "").strip()
        if not text:
            continue
        header = f"[Paper: {row.get('paper_title') or '未命名'}] [Page {row.get('page_number')}] {row.get('section') or row.get('heading') or '正文'}"
        block = header + "\n" + text[:per_row_chars]
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def to_sources(rows: list[dict[str, Any]], text_chars: int = 1200) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for row in rows:
        sources.append({
            "id": row.get("id"),
            "chunk_id": row.get("id"),
            "paper_id": row.get("paper_id"),
            "paper_title": row.get("paper_title", ""),
            "page_number": row.get("page_number"),
            "text": (row.get("text") or "")[:text_chars],
            "section": row.get("section", ""),
            "heading": row.get("heading", ""),
            "x0": row.get("x0"), "y0": row.get("y0"),
            "x1": row.get("x1"), "y1": row.get("y1"),
            "score": row.get("score"),
            "sources": row.get("sources", []),
        })
    return sources


def paper_exists(db_path: Path | str, paper_id: int) -> bool:
    connection = connect(db_path)
    try:
        return connection.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone() is not None
    finally:
        connection.close()
