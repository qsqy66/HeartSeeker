from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .embedding import (
    DEFAULT_EMBED_MODEL,
    EmbeddingError,
    delete_paper_vectors,
    embed_texts,
    normalize,
    search_chunks,
    upsert_chunks,
)


APP_HOME = Path(os.environ.get("PAPER_READER_HOME", Path.home() / "PaperReader"))
PAPERS_DIR = APP_HOME / "papers"
DB_PATH = APP_HOME / "database" / "paper_reader.sqlite3"
VECTORS_DIR = APP_HOME / "vectors"
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_PAGES = 300
PROVIDERS = {"deepseek": "https://api.deepseek.com/chat/completions", "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"}
MODEL_ENDPOINTS = {"deepseek": "https://api.deepseek.com/models", "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/models"}

# --- Structured extraction heuristics -------------------------------------------------

HEADING_RE = re.compile(
    r"^(?:abstract|introduction|background|methods?|materials?\s+and\s+methods|"
    r"results?|discussion|conclusions?|references?|limitations?|acknowledg?ments?|"
    r"outcomes?|study\s+design|statistical\s+analysis|participants|population|"
    r"interventions?|comparators?|ethics|funding|conflicts?\s+of\s+interest)\b",
    re.IGNORECASE,
)
NUMBERED_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def initialize_storage() -> None:
    for directory in (PAPERS_DIR, DB_PATH.parent, VECTORS_DIR, APP_HOME / "cache", APP_HOME / "exports", APP_HOME / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                file_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                page_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT,
                imported_at TEXT NOT NULL,
                vectorized INTEGER NOT NULL DEFAULT 0,
                needs_ocr INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                x0 REAL, y0 REAL, x1 REAL, y1 REAL,
                section TEXT NOT NULL DEFAULT '',
                heading TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS paper_metadata (
                paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
                authors TEXT NOT NULL DEFAULT '[]', abstract TEXT NOT NULL DEFAULT '',
                year INTEGER, doi TEXT, keywords TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
                quote TEXT NOT NULL, page_number INTEGER, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL, provider TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS page_text_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
                x0 REAL NOT NULL, y0 REAL NOT NULL, x1 REAL NOT NULL, y1 REAL NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS api_keys (
                provider TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, content='chunks', content_rowid='id'
            );
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
            END;
            """
        )
        # Migrate databases created by earlier V1 versions without making startup fragile.
        _ensure_column(connection, "papers", "metadata_json", "metadata_json TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(connection, "papers", "vectorized", "vectorized INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "papers", "needs_ocr", "needs_ocr INTEGER NOT NULL DEFAULT 0")
        _ensure_column(connection, "chunks", "x0", "x0 REAL")
        _ensure_column(connection, "chunks", "y0", "y0 REAL")
        _ensure_column(connection, "chunks", "x1", "x1 REAL")
        _ensure_column(connection, "chunks", "y1", "y1 REAL")
        _ensure_column(connection, "chunks", "section", "section TEXT NOT NULL DEFAULT ''")
        _ensure_column(connection, "chunks", "heading", "heading TEXT NOT NULL DEFAULT ''")


# --- Structured PDF extraction --------------------------------------------------------

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
    tables: list[dict[str, Any]] = []
    try:
        for table in page.find_tables().tables:
            md = _table_to_markdown(table.extract())
            if not md:
                continue
            bbox = table.bbox
            tables.append({"x0": float(bbox[0]), "y0": float(bbox[1]),
                           "x1": float(bbox[2]), "y1": float(bbox[3]), "text": md})
    except Exception:
        pass  # table detection is best-effort; a single bad page must not break import
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


def _is_heading(el: dict[str, Any], body_size: float) -> bool:
    text = el["text"].strip()
    if not text or len(text) > 150:
        return False
    if NUMBERED_RE.match(text) or HEADING_RE.match(text):
        return True
    if body_size and el["size"] >= body_size + 0.8 and el["bold_ratio"] >= 0.5:
        return True
    return False


def _split_text(text: str, x0: float, y0: float, x1: float, y1: float,
                section: str, heading: str, max_len: int = 1200) -> list[dict[str, Any]]:
    """Split a long block into sub-chunks that all share the same provenance."""
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
        start = end
    return [{"text": part, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
             "section": section, "heading": heading} for part in parts]


def _extract_page(page, page_number: int) -> tuple[str, list[dict[str, Any]]]:
    """Return (plain_text, chunks) with structure-aware chunking for one page."""
    blocks = _page_blocks(page)
    tables = _page_tables(page)
    page_width = float(page.rect.width or 612.0)
    sizes = [b["size"] for b in blocks if b["text"].strip()]
    body_size = sorted(sizes)[len(sizes) // 2] if sizes else 0.0

    elements: list[dict[str, Any]] = [{**b, "kind": "text"} for b in blocks]
    for table in tables:
        elements.append({"x0": table["x0"], "y0": table["y0"], "x1": table["x1"], "y1": table["y1"],
                         "text": table["text"], "size": body_size or 10.0, "bold_ratio": 0.0, "kind": "table"})
    ordered = _reading_order(elements, page_width)

    section = ""
    chunks: list[dict[str, Any]] = []
    parts: list[str] = []
    for el in ordered:
        text = el["text"].strip()
        if not text:
            continue
        if el["kind"] == "text" and _is_heading(el, body_size):
            section = text
            chunks.extend(_split_text(text, el["x0"], el["y0"], el["x1"], el["y1"], text, text))
            parts.append(text)
        else:
            chunks.extend(_split_text(text, el["x0"], el["y0"], el["x1"], el["y1"], section, ""))
            parts.append(text)
    return "\n\n".join(parts), chunks


# --- Storage + embedding helpers ------------------------------------------------------

def _safe_qwen_key() -> Optional[str]:
    try:
        return _key_store("qwen")
    except Exception:
        return None


def _embedding_model() -> str:
    with get_connection() as connection:
        row = connection.execute("SELECT value FROM settings WHERE key='embedding_model'").fetchone()
    value = (row["value"].strip() if row and row["value"].strip() else DEFAULT_EMBED_MODEL)
    return value


def _safe_delete_vectors(paper_id: int) -> None:
    try:
        delete_paper_vectors(VECTORS_DIR, paper_id)
    except Exception:
        pass


def _clean_chunks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        cleaned.append({
            "id": row["id"], "text": row["text"], "page_number": row["page_number"],
            "section": row.get("section", "") or "", "heading": row.get("heading", "") or "",
            "x0": row.get("x0"), "y0": row.get("y0"), "x1": row.get("x1"), "y1": row.get("y1"),
            "score": round(float(row["_distance"]), 4) if row.get("_distance") is not None else None,
        })
    return cleaned


def _retrieve_chunks(paper_id: int, query: str, top_k: int = 8) -> list[dict[str, Any]]:
    key = _safe_qwen_key()
    if not key:
        return []
    try:
        model = _embedding_model()
        vector = embed_texts([query], key, model)[0]
        return search_chunks(VECTORS_DIR, paper_id, normalize(vector), top_k)
    except Exception:
        return []


def _rag_context(paper_id: int, message: str) -> tuple[str, list[dict[str, Any]]]:
    rows = _retrieve_chunks(paper_id, message, 10)
    if rows:
        return "\n".join(f"[Page {row['page_number']}] {row['text'][:1500]}" for row in rows)[:30000], rows
    return _paper_context(paper_id), []


def _process_document(paper_id: int, document) -> None:
    """Re-parse a document into pages/chunks/vectors and mark it ready."""
    chunk_rows: list[dict[str, Any]] = []
    total_chars = 0
    _safe_delete_vectors(paper_id)
    with get_connection() as connection:
        # Delete chunks explicitly (not via cascade) so the FTS cleanup trigger fires.
        connection.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        connection.execute("DELETE FROM pages WHERE paper_id=?", (paper_id,))
        for page_number, page in enumerate(document, start=1):
            full_text, chunks = _extract_page(page, page_number)
            total_chars += len(full_text)
            page_id = connection.execute(
                "INSERT INTO pages(paper_id, page_number, text) VALUES (?, ?, ?)",
                (paper_id, page_number, full_text),
            ).lastrowid
            for block in page.get_text("blocks"):
                if len(block) >= 5 and str(block[4]).strip():
                    connection.execute(
                        "INSERT INTO page_text_blocks(page_id, x0, y0, x1, y1, text) VALUES (?, ?, ?, ?, ?, ?)",
                        (page_id, *block[:4], str(block[4]).strip()),
                    )
            for chunk_index, chunk in enumerate(chunks):
                cursor = connection.execute(
                    "INSERT INTO chunks(paper_id, page_id, chunk_index, text, x0, y0, x1, y1, section, heading) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (paper_id, page_id, chunk_index, chunk["text"], chunk["x0"], chunk["y0"],
                     chunk["x1"], chunk["y1"], chunk["section"], chunk["heading"]),
                )
                chunk_rows.append({
                    "id": cursor.lastrowid, "paper_id": paper_id, "page_id": page_id,
                    "page_number": page_number, "section": chunk["section"],
                    "heading": chunk["heading"], "text": chunk["text"],
                    "x0": chunk["x0"], "y0": chunk["y0"], "x1": chunk["x1"], "y1": chunk["y1"],
                })

    vectorized = 0
    if chunk_rows:
        key = _safe_qwen_key()
        if key:
            try:
                model = _embedding_model()
                vectors = embed_texts([row["text"] for row in chunk_rows], key, model)
                for row, vector in zip(chunk_rows, vectors):
                    row["vector"] = normalize(vector)
                upsert_chunks(VECTORS_DIR, chunk_rows)
                vectorized = 1
            except Exception:
                vectorized = 0  # embedding failure must not block reading/FTS

    needs_ocr = 0 if total_chars else 1
    with get_connection() as connection:
        connection.execute(
            "UPDATE papers SET page_count=?, status='ready', vectorized=?, needs_ocr=? WHERE id=?",
            (len(document), vectorized, needs_ocr, paper_id),
        )


# --- Pydantic payloads ----------------------------------------------------------------

class SettingsPayload(BaseModel):
    values: dict[str, str] = Field(default_factory=dict)


class ApiKeyPayload(BaseModel):
    key: str = Field(min_length=1, max_length=4096)


class EvidencePayload(BaseModel):
    quote: str = Field(min_length=1, max_length=12000)
    chunk_id: Optional[int] = None
    page_number: Optional[int] = None


class MetadataPayload(BaseModel):
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)


class ChatPayload(BaseModel):
    message: str = Field(default="", max_length=12000)
    question: str = Field(default="", max_length=12000)
    provider: Optional[str] = None


class ModelsPayload(BaseModel):
    provider: str
    api_key: Optional[str] = None


def _key_store(provider: str, value: Optional[str] = None) -> Optional[str]:
    """Store API keys in the operating system credential manager."""
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        raise HTTPException(400, "仅支持 deepseek 或 qwen")
    try:
        import keyring  # type: ignore
        if value is None:
            return keyring.get_password("paper-reader", provider)
        else:
            keyring.set_password("paper-reader", provider, value)
            return value
    except Exception as error:
        raise HTTPException(503, "系统安全凭据库不可用，请安装并启用 keyring 后再配置 API Key") from error


def _paper_context(paper_id: int) -> str:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT page_number,text FROM pages WHERE paper_id=? ORDER BY page_number", (paper_id,)
        ).fetchall()
    if not rows:
        raise HTTPException(404, "论文不存在或没有可用文本")
    return "\n".join(f"[Page {row['page_number']}] {row['text'][:6000]}" for row in rows)[:30000]


def _grounded_response(message: str, answer: str, context: str) -> str:
    """Reject provider output that invents citations outside the supplied paper context."""
    if not answer.strip():
        raise HTTPException(502, "模型返回了空响应")
    if not re.search(r"\[Page\s+\d+\]", answer, re.IGNORECASE):
        raise HTTPException(502, "模型响应缺少论文页码证据")
    pages = {int(n) for n in re.findall(r"\[Page\s+(\d+)\]", context, re.IGNORECASE)}
    cited = {int(n) for n in re.findall(r"\[Page\s+(\d+)\]", answer, re.IGNORECASE)}
    if not cited.intersection(pages):
        raise HTTPException(502, "模型引用了不存在的论文页码")
    return answer.strip()


def _provider_chat(provider: str, key: str, prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }).encode()
    request = urllib.request.Request(PROVIDERS[provider], data=payload, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"
    })
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(502, f"{provider} 服务暂不可用") from error


app = FastAPI(title="Paper Reader Local API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    initialize_storage()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "storage": str(APP_HOME)}


@app.get("/api/papers")
def list_papers() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, title, file_name, page_count, status, error_message, imported_at, vectorized, needs_ocr "
            "FROM papers ORDER BY imported_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        paper = connection.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if paper is None:
            raise HTTPException(404, "论文不存在")
        pages = connection.execute(
            "SELECT id, page_number, text FROM pages WHERE paper_id = ? ORDER BY page_number",
            (paper_id,),
        ).fetchall()
        metadata = connection.execute("SELECT * FROM paper_metadata WHERE paper_id=?", (paper_id,)).fetchone()
        blocks = connection.execute(
            "SELECT b.id,b.page_id,b.x0,b.y0,b.x1,b.y1,b.text FROM page_text_blocks b "
            "JOIN pages p ON p.id=b.page_id WHERE p.paper_id=? ORDER BY p.page_number,b.id", (paper_id,)
        ).fetchall() if connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='page_text_blocks'"
        ).fetchone() else []
    metadata_value = dict(metadata) if metadata else None
    if metadata_value:
        for field in ("authors", "keywords"):
            try:
                metadata_value[field] = json.loads(metadata_value[field])
            except (TypeError, json.JSONDecodeError):
                metadata_value[field] = []
    return {"paper": dict(paper), "metadata": metadata_value,
            "pages": [dict(page) for page in pages], "text_blocks": [dict(block) for block in blocks]}


@app.get("/api/papers/{paper_id}/structured")
def structured_paper(paper_id: int) -> dict[str, Any]:
    _paper_context(paper_id)
    fields: dict[str, dict[str, Any]] = {}
    patterns = {
        "研究设计": r"\b(randomized controlled trial|randomized clinical trial|cohort study|case-control study|cross-sectional study|systematic review|meta-analysis)\b",
        "样本量": r"\b(?:n\s*=\s*|sample size of\s+)(\d[\d,]*)",
        "统计方法": r"\b(odds ratio|hazard ratio|relative risk|confidence interval|p\s*[<=>])\b",
        "关键结果": r"\b(primary outcome|primary endpoint|main outcome)\b",
        "局限性": r"\b(limitations?|limitation of this study)\b",
    }
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT chunks.id, chunks.text, pages.page_number, chunks.x0, chunks.y0, chunks.x1, chunks.y1, chunks.section "
            "FROM chunks JOIN pages ON pages.id=chunks.page_id WHERE chunks.paper_id=? ORDER BY pages.page_number,chunks.id",
            (paper_id,),
        ).fetchall()
    for name, pattern in patterns.items():
        matches = [dict(row) for row in rows if re.search(pattern, row["text"], re.IGNORECASE)]
        fields[name] = {
            "value": matches[0]["text"][:500] if matches else "论文未明确报告",
            "evidence": [{"id": row["id"], "text": row["text"], "page_number": row["page_number"],
                          "x0": row["x0"], "y0": row["y0"], "x1": row["x1"], "y1": row["y1"]} for row in matches[:3]],
        }
    for name in ("一句话总结", "Population", "Intervention / Exposure", "Comparator", "Outcomes"):
        fields.setdefault(name, {"value": "论文未明确报告", "evidence": []})
    return {"fields": fields}


@app.post("/api/papers/import")
async def import_paper(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "仅支持 PDF 文件")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "单个 PDF 不得超过 50 MB")
    digest = hashlib.sha256(content).hexdigest()
    with get_connection() as connection:
        existing = connection.execute("SELECT id, title FROM papers WHERE sha256 = ?", (digest,)).fetchone()
        if existing:
            raise HTTPException(409, f"论文已导入：{existing['title']}")

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename)
    stored_path = PAPERS_DIR / f"{digest[:16]}_{safe_name}"
    stored_path.write_bytes(content)
    paper_id: Optional[int] = None
    document: Optional[fitz.Document] = None
    try:
        document = fitz.open(stored_path)
        if len(document) > MAX_PAGES:
            raise ValueError("单个 PDF 不得超过 300 页")
        title = (document.metadata.get("title") or Path(file.filename).stem).strip() or "未命名论文"
        with get_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO papers(title, file_name, stored_path, sha256, page_count, status, imported_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (title, file.filename, str(stored_path), digest, len(document), "processing", utc_now()),
            )
            paper_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO paper_metadata(paper_id, authors, abstract, keywords) VALUES(?,?,?,?)",
                (paper_id, "[]", "", "[]"),
            )
        _process_document(paper_id, document)
        return {"id": paper_id, "title": title, "status": "ready"}
    except Exception as error:
        if paper_id is not None:
            with get_connection() as connection:
                connection.execute(
                    "UPDATE papers SET status = 'failed', error_message = ? WHERE id = ?",
                    (str(error), paper_id),
                )
        if paper_id is None:
            stored_path.unlink(missing_ok=True)
        raise HTTPException(422, f"解析 PDF 失败：{error}") from error
    finally:
        if document is not None:
            document.close()


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: int) -> dict[str, str]:
    with get_connection() as connection:
        paper = connection.execute("SELECT stored_path FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if paper is None:
            raise HTTPException(404, "论文不存在")
        connection.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        connection.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    _safe_delete_vectors(paper_id)
    Path(paper["stored_path"]).unlink(missing_ok=True)
    return {"status": "deleted"}


@app.get("/api/papers/{paper_id}/search")
def search_paper(paper_id: int, q: str) -> list[dict[str, Any]]:
    query = q.strip()
    if not query:
        return []
    with get_connection() as connection:
        if connection.execute("SELECT 1 FROM papers WHERE id = ?", (paper_id,)).fetchone() is None:
            raise HTTPException(404, "论文不存在")
        tokens = " ".join(re.findall(r"[\w]+", query))
        rows = connection.execute(
            "SELECT chunks.id, chunks.text, pages.page_number, chunks.section, chunks.heading, "
            "chunks.x0, chunks.y0, chunks.x1, chunks.y1 "
            "FROM chunks_fts JOIN chunks ON chunks_fts.rowid = chunks.id "
            "JOIN pages ON chunks.page_id = pages.id "
            "WHERE chunks.paper_id = ? AND chunks_fts MATCH ? LIMIT 20",
            (paper_id, tokens),
        ).fetchall()
    return [{"id": row["id"], "text": row["text"], "page_number": row["page_number"],
             "section": row["section"], "heading": row["heading"],
             "x0": row["x0"], "y0": row["y0"], "x1": row["x1"], "y1": row["y1"],
             "source": "fts"} for row in rows]


@app.get("/api/papers/{paper_id}/retrieve")
def retrieve_paper(paper_id: int, q: str, top_k: int = 8) -> list[dict[str, Any]]:
    query = q.strip()
    if not query:
        return []
    _paper_context(paper_id)
    top_k = max(1, min(top_k, 50))
    rows = _retrieve_chunks(paper_id, query, top_k)
    if rows:
        return [{**chunk, "source": "vector"} for chunk in _clean_chunks(rows)]
    return search_paper(paper_id, query)


@app.get("/api/papers/{paper_id}/file")
def paper_file(paper_id: int):
    from fastapi.responses import FileResponse

    with get_connection() as connection:
        paper = connection.execute("SELECT stored_path, file_name FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if paper is None or not Path(paper["stored_path"]).is_file():
        raise HTTPException(404, "论文文件不存在")
    return FileResponse(paper["stored_path"], media_type="application/pdf", filename=paper["file_name"])


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    with get_connection() as connection:
        rows = connection.execute("SELECT key,value FROM settings").fetchall()
    values = {row["key"]: row["value"] for row in rows}
    provider = values.get("provider", "")
    configured = False
    if provider:
        try:
            configured = bool(_key_store(provider))
        except HTTPException:
            configured = False
    return {
        "provider": provider,
        "model": values.get("model", ""),
        "embedding_model": values.get("embedding_model", DEFAULT_EMBED_MODEL),
        "api_key_configured": configured,
        "data_dir": str(APP_HOME),
    }


@app.put("/api/settings")
def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider", "")).lower()
    model = str(payload.get("model", ""))
    embedding_model = str(payload.get("embedding_model", "")).strip() or DEFAULT_EMBED_MODEL
    api_key = str(payload.get("api_key", ""))
    if provider and provider not in PROVIDERS:
        raise HTTPException(400, "仅支持 deepseek 或 qwen")
    if api_key:
        _key_store(provider, api_key)
    with get_connection() as connection:
        for key, value in {"provider": provider, "model": model, "embedding_model": embedding_model}.items():
            if len(key) > 100 or len(value) > 10000:
                raise HTTPException(400, "设置键或值过长")
            connection.execute(
                "INSERT INTO settings(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (key, value, utc_now()),
            )
    configured = False
    if provider:
        try:
            configured = bool(_key_store(provider))
        except HTTPException:
            configured = False
    return {
        "provider": provider, "model": model, "embedding_model": embedding_model,
        "api_key_configured": configured, "data_dir": str(APP_HOME),
    }


@app.post("/api/models")
def list_remote_models(payload: ModelsPayload) -> dict[str, Any]:
    provider = payload.provider.lower().strip()
    if provider not in MODEL_ENDPOINTS:
        raise HTTPException(400, "仅支持 deepseek 或 qwen")
    key = payload.api_key.strip() if payload.api_key else _key_store(provider)
    if not key:
        raise HTTPException(400, "请先输入或配置该供应商的 API Key")
    request = urllib.request.Request(
        MODEL_ENDPOINTS[provider],
        headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise HTTPException(401, "API Key 无效或没有读取模型列表的权限") from error
        raise HTTPException(502, f"{provider} 模型列表服务返回 HTTP {error.code}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise HTTPException(502, f"{provider} 模型列表暂时不可用") from error
    models = []
    for item in data.get("data", []):
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            models.append(item["id"])
    if not models:
        raise HTTPException(502, "供应商未返回可用模型列表")
    return {"provider": provider, "models": sorted(set(models))}


@app.put("/api/settings/api-key/{provider}")
def save_api_key(provider: str, payload: ApiKeyPayload) -> dict[str, str]:
    _key_store(provider, payload.key)
    return {"provider": provider.lower(), "configured": "true"}


@app.get("/api/settings/api-key/{provider}")
def api_key_status(provider: str) -> dict[str, Any]:
    return {"provider": provider.lower(), "configured": bool(_key_store(provider))}


@app.post("/api/papers/{paper_id}/evidence")
def add_evidence(paper_id: int, payload: EvidencePayload) -> dict[str, Any]:
    context = _paper_context(paper_id)
    if payload.quote not in context:
        raise HTTPException(422, "证据摘录不在论文文本中")
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO evidence(paper_id,chunk_id,quote,page_number,created_at) VALUES(?,?,?,?,?)",
            (paper_id, payload.chunk_id, payload.quote, payload.page_number, utc_now()),
        )
    return {"id": cursor.lastrowid, "paper_id": paper_id, "quote": payload.quote, "page_number": payload.page_number}


@app.get("/api/papers/{paper_id}/evidence")
def list_evidence(paper_id: int) -> list[dict[str, Any]]:
    _paper_context(paper_id)
    with get_connection() as connection:
        rows = connection.execute("SELECT * FROM evidence WHERE paper_id=? ORDER BY created_at", (paper_id,)).fetchall()
    return [dict(row) for row in rows]


@app.put("/api/papers/{paper_id}/metadata")
def update_metadata(paper_id: int, payload: MetadataPayload) -> dict[str, Any]:
    _paper_context(paper_id)
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO paper_metadata(paper_id,authors,abstract,year,doi,keywords) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(paper_id) DO UPDATE SET authors=excluded.authors,abstract=excluded.abstract,"
            "year=excluded.year,doi=excluded.doi,keywords=excluded.keywords",
            (paper_id, json.dumps(payload.authors, ensure_ascii=False), payload.abstract,
             payload.year, payload.doi, json.dumps(payload.keywords, ensure_ascii=False)),
        )
    return payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()


@app.post("/api/papers/{paper_id}/chat")
def chat(paper_id: int, payload: ChatPayload) -> dict[str, Any]:
    message = (payload.message or payload.question).strip()
    if not message:
        raise HTTPException(400, "问题不能为空")
    provider = (payload.provider or "deepseek").lower()
    if provider not in PROVIDERS:
        raise HTTPException(400, "provider 必须是 deepseek 或 qwen")
    key = _key_store(provider)
    if not key:
        raise HTTPException(400, f"{provider} 尚未配置 API key")
    context, retrieved = _rag_context(paper_id, message)
    with get_connection() as connection:
        setting = connection.execute("SELECT value FROM settings WHERE key='model'").fetchone()
    model = setting["value"].strip() if setting and setting["value"].strip() else (
        "deepseek-chat" if provider == "deepseek" else "qwen-plus"
    )
    prompt = (
        "仅依据以下论文内容回答问题。必须在回答中引用一个或多个 [Page N] 页码；"
        "不确定时明确说无法从论文判断。\n论文内容:\n" + context + "\n问题:\n" + message
    )
    with get_connection() as connection:
        connection.execute("INSERT INTO chat_messages(paper_id,role,content,provider,created_at) VALUES(?,?,?,?,?)",
                           (paper_id, "user", message, provider, utc_now()))
    answer = _grounded_response(message, _provider_chat(provider, key, prompt, model), context)
    with get_connection() as connection:
        cursor = connection.execute("INSERT INTO chat_messages(paper_id,role,content,provider,created_at) VALUES(?,?,?,?,?)",
                                     (paper_id, "assistant", answer, provider, utc_now()))
    citations = []
    for page_number in sorted({int(number) for number in re.findall(r"\[Page\s+(\d+)\]", answer)}):
        citations.append({"id": page_number, "page_number": page_number, "text": f"[Page {page_number}]"})
    sources = _clean_chunks(retrieved)
    return {"id": cursor.lastrowid, "content": answer, "answer": answer, "provider": provider,
            "citations": citations, "sources": sources}


@app.get("/api/papers/{paper_id}/chat")
def chat_history(paper_id: int) -> list[dict[str, Any]]:
    _paper_context(paper_id)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id,role,content,provider,created_at FROM chat_messages WHERE paper_id=? ORDER BY id", (paper_id,)
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/papers/{paper_id}/retry")
def retry_paper(paper_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        paper = connection.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not paper:
            raise HTTPException(404, "论文不存在")
        path = Path(paper["stored_path"])
        connection.execute("UPDATE papers SET status='processing', error_message=NULL WHERE id=?", (paper_id,))
    try:
        document = fitz.open(path)
        try:
            _process_document(paper_id, document)
        finally:
            document.close()
        return {"id": paper_id, "status": "ready"}
    except Exception as error:
        with get_connection() as connection:
            connection.execute("UPDATE papers SET status='failed',error_message=? WHERE id=?", (str(error), paper_id))
        raise HTTPException(422, f"重试解析失败：{error}") from error
