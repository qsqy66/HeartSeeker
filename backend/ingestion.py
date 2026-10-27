"""PDF ingestion, structure-aware chunking and vector index maintenance.

Parsing is deliberately kept separate from FastAPI so imports can run in a
background worker and tests can exercise a real PDF without starting a server.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Optional

import fitz

from . import ocr as ocr_module
from . import tokenization
from .embedding import (
    DEFAULT_EMBED_MODEL,
    delete_paper_vectors,
    embed_texts,
    normalize,
    upsert_chunks,
)
from .storage import get_connection

HEADING_RE = re.compile(
    r"^(?:abstract|introduction|background|methods?|materials?\s+and\s+methods|"
    r"results?|discussion|conclusions?|references?|limitations?|acknowledg?ments?|"
    r"outcomes?|study\s+design|statistical\s+analysis|participants|population|"
    r"interventions?|comparators?|ethics|funding|conflicts?\s+of\s+interest)\b",
    re.IGNORECASE,
)
CHINESE_HEADING_RE = re.compile(
    r"^(?:摘要|引言|背景|方法|材料与方法|结果|讨论|结论|参考文献|局限性|"
    r"研究设计|统计分析|研究对象|干预措施|对照组|结局指标|伦理|资助|利益冲突)"
)
NUMBERED_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")

ProgressCallback = Callable[[str, int, int], None]


def _page_blocks(page) -> list[dict[str, Any]]:
    """Extract text blocks with bbox, average font size and bold ratio."""
    data = page.get_text("dict")
    blocks: list[dict[str, Any]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # skip image blocks (type 1)
            continue
        line_texts: list[str] = []
        total_chars = 0
        bold_chars = 0
        size_sum = 0.0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_texts.append("".join(span.get("text", "") for span in spans))
            for span in spans:
                text = span.get("text", "")
                n = len(text)
                if n:
                    total_chars += n
                    if span.get("flags", 0) & 16:  # PyMuPDF flag bit 4 = bold
                        bold_chars += n
                    size_sum += float(span.get("size", 0)) * n
        text = " ".join(part.strip() for part in line_texts if part.strip())
        if not text:
            continue
        bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
        blocks.append({
            "x0": float(bbox[0]), "y0": float(bbox[1]),
            "x1": float(bbox[2]), "y1": float(bbox[3]),
            "text": text,
            "size": size_sum / total_chars if total_chars else 0.0,
            "bold_ratio": bold_chars / total_chars if total_chars else 0.0,
        })
    return blocks


def _table_to_markdown(rows: list[list[Any]]) -> str:
    cleaned = [[("" if cell is None else str(cell).strip()) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return ""
    ncol = max(len(row) for row in cleaned)
    padded = [row + [""] * (ncol - len(row)) for row in cleaned]
    header, body = padded[0], padded[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * ncol) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _page_tables(page) -> list[dict[str, Any]]:
    """Extract tables, falling back to text alignment for borderless tables."""
    found: list[Any] = []
    for strategy in ("lines", "text"):
        try:
            if strategy == "lines":
                found = list(page.find_tables().tables)
            else:
                found = list(page.find_tables(strategy="text").tables)
        except Exception:
            found = []
        if found:
            break
    tables: list[dict[str, Any]] = []
    for table in found:
        try:
            md = _table_to_markdown(table.extract())
        except Exception:
            continue
        if not md:
            continue
        bbox = table.bbox
        tables.append({"x0": float(bbox[0]), "y0": float(bbox[1]),
                       "x1": float(bbox[2]), "y1": float(bbox[3]), "text": md})
    return tables


def _reading_order(elements: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    """Column-aware reading order for two-column medical papers."""
    if len(elements) <= 1:
        return elements
    xs = sorted(el["x0"] for el in elements)
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    max_gap = max(gaps)
    if max_gap > max(40.0, page_width * 0.15):
        split = xs[gaps.index(max_gap)] + 1.0
        left = [el for el in elements if el["x0"] < split]
        right = [el for el in elements if el["x0"] >= split]
        if left and right:
            return (sorted(left, key=lambda el: (el["y0"], el["x0"]))
                    + sorted(right, key=lambda el: (el["y0"], el["x0"])))
    return sorted(elements, key=lambda el: (round(el["y0"] / 12.0), el["x0"]))


def is_heading(el: dict[str, Any], body_size: float) -> bool:
    text = el["text"].strip()
    if not text or len(text) > 150:
        return False
    if NUMBERED_RE.match(text) or HEADING_RE.match(text) or CHINESE_HEADING_RE.match(text):
        return True
    if body_size and el["size"] >= body_size + 0.8 and el["bold_ratio"] >= 0.5:
        return True
    return False


CHUNK_OVERLAP = 200


def _split_text(text: str, x0: float, y0: float, x1: float, y1: float,
                section: str, heading: str, max_len: int = 1200,
                overlap: int = CHUNK_OVERLAP) -> list[dict[str, Any]]:
    """Split a long block into overlapping sub-chunks with shared provenance.

    Overlap keeps definitions and methods that straddle a boundary retrievable
    by either side; all parts still point at the same page/bbox.
    """
    if len(text) <= max_len:
        return [{"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                 "section": section, "heading": heading}]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("。", start, end))
            if cut > start + max_len * 0.5:
                end = cut + 1
        parts.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - max(0, min(overlap, max_len // 2)))
    return [{"text": part, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
             "section": section, "heading": heading} for part in parts]


def extract_page(page, page_number: int) -> tuple[str, list[dict[str, Any]]]:
    """Return (plain_text, chunks) with structure-aware chunking for one page."""
    blocks = _page_blocks(page)
    tables = _page_tables(page)
    page_width = float(page.rect.width or 612.0)
    sizes = [b["size"] for b in blocks if b["text"].strip()]
    body_size = sorted(sizes)[len(sizes) // 2] if sizes else 0.0

    elements: list[dict[str, Any]] = [{**b, "kind": "text"} for b in blocks]
    for table in tables:
        elements.append({"x0": table["x0"], "y0": table["y0"], "x1": table["x1"], "y1": table["y1"],
                         "text": table["text"], "size": body_size or 10.0,
                         "bold_ratio": 0.0, "kind": "table"})
    ordered = _reading_order(elements, page_width)

    section = ""
    chunks: list[dict[str, Any]] = []
    parts: list[str] = []
    for el in ordered:
        text = el["text"].strip()
        if not text:
            continue
        if el["kind"] == "text" and is_heading(el, body_size):
            section = text
            chunks.extend(_split_text(text, el["x0"], el["y0"], el["x1"], el["y1"], text, text))
            parts.append(text)
        else:
            chunks.extend(_split_text(text, el["x0"], el["y0"], el["x1"], el["y1"], section, ""))
            parts.append(text)
    return "\n\n".join(parts), chunks


def _ocr_chunks(text: str, page_width: float, page_height: float,
                max_len: int = 1200) -> list[dict[str, Any]]:
    """Turn OCR output into provenance-preserving chunks.

    OCR has no per-line boxes we can trust, so all chunks share the page rect.
    Headings are still detected from the text so section-aware tools keep
    working on scanned papers.
    """
    chunks: list[dict[str, Any]] = []
    section = ""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        first = lines[0]
        heading = first if len(first) <= 150 and (
            HEADING_RE.match(first) or CHINESE_HEADING_RE.match(first)
        ) else ""
        if heading:
            section = heading
            chunks.extend(_split_text(heading, 0.0, 0.0, page_width, page_height,
                                      section, heading, max_len=max_len))
            lines = lines[1:]
        body = " ".join(lines).strip()
        if body:
            chunks.extend(_split_text(body, 0.0, 0.0, page_width, page_height,
                                      section, "", max_len=max_len))
    return chunks


def _setting(connection: sqlite3.Connection, key: str, default: str) -> str:
    row = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    value = row["value"].strip() if row and row["value"].strip() else ""
    return value or default


def _safe_delete_vectors(vectors_dir: Path | str, paper_id: int) -> None:
    try:
        delete_paper_vectors(Path(vectors_dir), paper_id)
    except Exception:
        pass


def _report(progress: Optional[ProgressCallback], stage: str, current: int, total: int) -> None:
    if progress is not None:
        try:
            progress(stage, current, total)
        except Exception:
            pass


def process_document(paper_id: int, document, db_path: Path | str,
                     vectors_dir: Path | str, embedding_key: Optional[str] = None,
                     embedding_model: Optional[str] = None,
                     progress: Optional[ProgressCallback] = None,
                     ocr_engine: Optional[Any] = None,
                     ocr_lang: str = ocr_module.DEFAULT_LANG,
                     ocr_dpi: int = ocr_module.DEFAULT_DPI,
                     ocr_min_chars: int = ocr_module.MIN_PAGE_CHARS,
                     enable_ocr: bool = True) -> dict[str, Any]:
    """Re-parse a document into pages/chunks/vectors and mark it ready.

    Pages without a usable text layer are rendered and passed to a local OCR
    engine.  When no engine is installed the page stays marked ``needs_ocr``
    but the import still succeeds.
    """
    _safe_delete_vectors(vectors_dir, paper_id)
    chunk_rows: list[dict[str, Any]] = []
    total_chars = 0
    page_count = len(document)
    engine = None
    if enable_ocr:
        try:
            if isinstance(ocr_engine, str):
                engine = ocr_module.get_engine(ocr_engine)
            else:
                engine = ocr_engine or ocr_module.get_engine()
        except ocr_module.OcrError:
            engine = None
    ocr_pages = 0
    ocr_errors: list[str] = []
    with get_connection(db_path) as connection:
        # Delete chunks explicitly (not via cascade) so the FTS cleanup trigger fires.
        connection.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        connection.execute("DELETE FROM pages WHERE paper_id=?", (paper_id,))
        connection.commit()
        for page_number, page in enumerate(document, start=1):
            full_text, chunks = extract_page(page, page_number)
            used_ocr = False
            if engine is not None and engine.available() and len(full_text.strip()) < ocr_min_chars:
                _report(progress, "ocr", page_number, page_count)
                try:
                    ocr_text = ocr_module.ocr_page(page, engine, ocr_lang, ocr_dpi)
                except Exception as error:  # OCR is best-effort per page
                    ocr_errors.append(f"第 {page_number} 页 OCR 失败：{error}")
                else:
                    if ocr_text.strip():
                        full_text = ocr_text
                        chunks = _ocr_chunks(ocr_text, float(page.rect.width or 612.0),
                                             float(page.rect.height or 792.0))
                        used_ocr = True
                        ocr_pages += 1
            total_chars += len(full_text)
            page_id = connection.execute(
                "INSERT INTO pages(paper_id, page_number, text) VALUES (?, ?, ?)",
                (paper_id, page_number, full_text),
            ).lastrowid
            blocks = list(page.get_text("blocks"))
            if used_ocr:
                blocks = [(0.0, 0.0, float(page.rect.width or 612.0),
                           float(page.rect.height or 792.0), full_text)]
            for block in blocks:
                if len(block) >= 5 and str(block[4]).strip():
                    connection.execute(
                        "INSERT INTO page_text_blocks(page_id, x0, y0, x1, y1, text) VALUES (?, ?, ?, ?, ?, ?)",
                        (page_id, *block[:4], str(block[4]).strip()),
                    )
            for chunk_index, chunk in enumerate(chunks):
                search_text = tokenization.segment(chunk["text"])
                cursor = connection.execute(
                    "INSERT INTO chunks(paper_id, page_id, chunk_index, text, search_text, "
                    "x0, y0, x1, y1, section, heading) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (paper_id, page_id, chunk_index, chunk["text"], search_text,
                     chunk["x0"], chunk["y0"], chunk["x1"], chunk["y1"],
                     chunk["section"], chunk["heading"]),
                )
                chunk_rows.append({
                    "id": cursor.lastrowid, "paper_id": paper_id, "page_id": page_id,
                    "page_number": page_number, "section": chunk["section"],
                    "heading": chunk["heading"], "text": chunk["text"],
                    "x0": chunk["x0"], "y0": chunk["y0"], "x1": chunk["x1"], "y1": chunk["y1"],
                })
            connection.commit()
            _report(progress, "parsing", page_number, page_count)

    vectorized = 0
    used_model = ""
    embedding_dim: Optional[int] = None
    embedding_error = ""
    if chunk_rows and embedding_key:
        try:
            model = embedding_model or ""
            if not model:
                with get_connection(db_path) as connection:
                    model = _setting(connection, "embedding_model", DEFAULT_EMBED_MODEL)
            vectors = embed_texts(
                [row["text"] for row in chunk_rows], embedding_key, model,
                on_progress=lambda done, total: _report(progress, "embedding", done, total),
            )
            for row, vector in zip(chunk_rows, vectors):
                row["vector"] = normalize(vector)
            upsert_chunks(Path(vectors_dir), chunk_rows)
            used_model = model
            embedding_dim = len(vectors[0]) if vectors else None
            vectorized = 1
        except Exception as error:
            vectorized = 0  # embedding failure must not block reading/FTS
            embedding_error = str(error)
            _report(progress, "embedding_failed", 0, len(chunk_rows))

    needs_ocr = 0 if total_chars else 1
    error_message = embedding_error or ("；".join(ocr_errors[:3]) if ocr_errors else "")
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE papers SET page_count=?, status='ready', vectorized=?, needs_ocr=?, "
            "embedding_model=?, embedding_dim=?, progress_stage=?, progress_current=?, progress_total=?, "
            "error_message=? WHERE id=?",
            (page_count, vectorized, needs_ocr, used_model, embedding_dim,
             "ready" if vectorized or not embedding_key else "ready_no_vectors",
             page_count, page_count, error_message or None, paper_id),
        )
        connection.commit()
    _report(progress, "ready", page_count, page_count)
    return {"paper_id": paper_id, "pages": page_count, "chunks": len(chunk_rows),
            "vectorized": bool(vectorized), "embedding_model": used_model,
            "embedding_dim": embedding_dim, "needs_ocr": bool(needs_ocr),
            "ocr_pages": ocr_pages, "ocr_engine": getattr(engine, "name", "none"),
            "ocr_errors": ocr_errors, "embedding_error": embedding_error}


def reindex_document(paper_id: int, db_path: Path | str, vectors_dir: Path | str,
                     embedding_key: str, embedding_model: Optional[str] = None,
                     progress: Optional[ProgressCallback] = None) -> dict[str, Any]:
    """Re-embed existing chunks without re-parsing the PDF."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT id, paper_id, page_id, text FROM chunks WHERE paper_id=? ORDER BY id",
            (paper_id,),
        ).fetchall()
        paper = connection.execute("SELECT id FROM papers WHERE id=?", (paper_id,)).fetchone()
    if paper is None:
        raise ValueError("论文不存在")
    if not rows:
        raise ValueError("论文没有可重建索引的切片")
    model = embedding_model or ""
    if not model:
        with get_connection(db_path) as connection:
            model = _setting(connection, "embedding_model", DEFAULT_EMBED_MODEL)
    _safe_delete_vectors(vectors_dir, paper_id)
    chunk_rows = [dict(row) for row in rows]
    vectors = embed_texts(
        [row["text"] for row in chunk_rows], embedding_key, model,
        on_progress=lambda done, total: _report(progress, "embedding", done, total),
    )
    for row, vector in zip(chunk_rows, vectors):
        row["vector"] = normalize(vector)
    upsert_chunks(Path(vectors_dir), chunk_rows)
    dim = len(vectors[0]) if vectors else None
    with get_connection(db_path) as connection:
        connection.execute(
            "UPDATE papers SET vectorized=1, embedding_model=?, embedding_dim=?, "
            "progress_stage='ready', progress_current=?, progress_total=?, error_message=NULL WHERE id=?",
            (model, dim, len(chunk_rows), len(chunk_rows), paper_id),
        )
        connection.commit()
    _report(progress, "ready", len(chunk_rows), len(chunk_rows))
    return {"paper_id": paper_id, "chunks": len(chunk_rows), "vectorized": True,
            "embedding_model": model, "embedding_dim": dim}
