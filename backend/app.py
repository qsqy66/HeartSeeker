from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


APP_HOME = Path(os.environ.get("PAPER_READER_HOME", Path.home() / "PaperReader"))
PAPERS_DIR = APP_HOME / "papers"
DB_PATH = APP_HOME / "database" / "paper_reader.sqlite3"
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_PAGES = 300
PROVIDERS = {"deepseek": "https://api.deepseek.com/chat/completions", "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"}
MODEL_ENDPOINTS = {"deepseek": "https://api.deepseek.com/models", "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1/models"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_storage() -> None:
    for directory in (PAPERS_DIR, DB_PATH.parent, APP_HOME / "vectors", APP_HOME / "cache", APP_HOME / "exports", APP_HOME / "logs"):
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
                imported_at TEXT NOT NULL
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
                text TEXT NOT NULL
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
        # Migrate databases created by the first V1 without making startup fragile.
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(papers)")}
        if "metadata_json" not in columns:
            connection.execute("ALTER TABLE papers ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")


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
    # Responses must cite at least one supplied page marker for evidence-grounded mode.
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


app = FastAPI(title="Paper Reader Local API", version="0.1.0")
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
            "SELECT id, title, file_name, page_count, status, error_message, imported_at "
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
            "SELECT chunks.id, chunks.text, pages.page_number FROM chunks "
            "JOIN pages ON pages.id=chunks.page_id WHERE chunks.paper_id=? ORDER BY pages.page_number,chunks.id",
            (paper_id,),
        ).fetchall()
    for name, pattern in patterns.items():
        matches = [dict(row) for row in rows if re.search(pattern, row["text"], re.IGNORECASE)]
        fields[name] = {
            "value": matches[0]["text"][:500] if matches else "论文未明确报告",
            "evidence": [{"id": row["id"], "text": row["text"], "page_number": row["page_number"]} for row in matches[:3]],
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
    paper_id: int | None = None
    document: fitz.Document | None = None
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
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                page_cursor = connection.execute(
                    "INSERT INTO pages(paper_id, page_number, text) VALUES (?, ?, ?)",
                    (paper_id, page_number, text),
                )
                page_id = page_cursor.lastrowid
                for block in page.get_text("blocks"):
                    if len(block) >= 5 and str(block[4]).strip():
                        connection.execute(
                            "INSERT INTO page_text_blocks(page_id,x0,y0,x1,y1,text) VALUES(?,?,?,?,?,?)",
                            (page_id, *block[:4], str(block[4]).strip()),
                        )
                paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
                if not paragraphs and text:
                    paragraphs = [text]
                for chunk_index, paragraph in enumerate(paragraphs):
                    connection.execute(
                        "INSERT INTO chunks(paper_id, page_id, chunk_index, text) VALUES (?, ?, ?, ?)",
                        (paper_id, page_id, chunk_index, paragraph[:4000]),
                    )
            connection.execute("UPDATE papers SET status = 'ready' WHERE id = ?", (paper_id,))
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
        connection.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
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
            "SELECT chunks.id, chunks.text, pages.page_number "
            "FROM chunks_fts JOIN chunks ON chunks_fts.rowid = chunks.id "
            "JOIN pages ON chunks.page_id = pages.id "
            "WHERE chunks.paper_id = ? AND chunks_fts MATCH ? LIMIT 20",
            (paper_id, tokens),
        ).fetchall()
    return [dict(row) for row in rows]


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
        "api_key_configured": configured,
        "data_dir": str(APP_HOME),
    }


@app.put("/api/settings")
def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider", "")).lower()
    model = str(payload.get("model", ""))
    api_key = str(payload.get("api_key", ""))
    if provider and provider not in PROVIDERS:
        raise HTTPException(400, "仅支持 deepseek 或 qwen")
    if api_key:
        _key_store(provider, api_key)
    with get_connection() as connection:
        for key, value in {"provider": provider, "model": model}.items():
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
        "provider": provider,
        "model": model,
        "api_key_configured": configured,
        "data_dir": str(APP_HOME),
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
    context = _paper_context(paper_id)
    message = (payload.message or payload.question).strip()
    if not message:
        raise HTTPException(400, "问题不能为空")
    provider = (payload.provider or "deepseek").lower()
    if provider not in PROVIDERS:
        raise HTTPException(400, "provider 必须是 deepseek 或 qwen")
    key = _key_store(provider)
    if not key:
        raise HTTPException(400, f"{provider} 尚未配置 API key")
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
    return {"id": cursor.lastrowid, "content": answer, "answer": answer, "provider": provider, "citations": citations}


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
        connection.execute("DELETE FROM pages WHERE paper_id=?", (paper_id,))
        connection.execute("UPDATE papers SET status='processing', error_message=NULL WHERE id=?", (paper_id,))
    try:
        document = fitz.open(path)
        with get_connection() as connection:
            for number, page in enumerate(document, 1):
                text = page.get_text("text").strip()
                page_id = connection.execute("INSERT INTO pages(paper_id,page_number,text) VALUES(?,?,?)",
                                             (paper_id, number, text)).lastrowid
                for block in page.get_text("blocks"):
                    if len(block) >= 5 and str(block[4]).strip():
                        connection.execute("INSERT INTO page_text_blocks(page_id,x0,y0,x1,y1,text) VALUES(?,?,?,?,?,?)",
                                           (page_id, *block[:4], str(block[4]).strip()))
                for index, paragraph in enumerate([p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]):
                    connection.execute("INSERT INTO chunks(paper_id,page_id,chunk_index,text) VALUES(?,?,?,?)",
                                       (paper_id, page_id, index, paragraph[:4000]))
            connection.execute("UPDATE papers SET page_count=?,status='ready' WHERE id=?", (len(document), paper_id))
        document.close()
        return {"id": paper_id, "status": "ready"}
    except Exception as error:
        with get_connection() as connection:
            connection.execute("UPDATE papers SET status='failed',error_message=? WHERE id=?", (str(error), paper_id))
        raise HTTPException(422, f"重试解析失败：{error}") from error
