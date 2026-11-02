from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

import fitz
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import clinical, evidence, memory, observability, ocr as ocr_module, ontology
from .agent import AgentConfig, AgentError, AgentRunner
from .credentials import (
    EXTRA_KEY_SOURCES, CredentialError, key_store as _credential_key_store, safe_key,
)
from .embedding import DEFAULT_EMBED_MODEL, delete_paper_vectors
from .ingestion import (
    is_heading as _is_heading,
    process_document,
    reindex_document,
)
from .safety import MEDICAL_SYSTEM_RULES, apply_safety, wrap_untrusted
from .storage import APP_HOME, DB_PATH, PAPERS_DIR, VECTORS_DIR, get_connection, initialize_storage, utc_now
from . import providers, rerank, retrieval
from .structured import extract_fields


MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_PAGES = 300
RECENT_TURNS = 10  # 最近 10 轮对话原文保留，更早的历史压缩为摘要
LOGGER = logging.getLogger("heartseeker_med.ingestion")

# One worker keeps CPU/embedding-heavy imports serialized while the API stays
# responsive; imports are durable in SQLite, so a crash is recoverable via retry.
_IMPORT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-import")

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
        score = row.get("_distance")
        if score is None:
            score = row.get("score")
        cleaned.append({
            "id": row["id"], "text": row["text"], "page_number": row["page_number"],
            "section": row.get("section", "") or "", "heading": row.get("heading", "") or "",
            "x0": row.get("x0"), "y0": row.get("y0"), "x1": row.get("x1"), "y1": row.get("y1"),
            "score": round(float(score), 4) if score is not None else None,
            "paper_id": row.get("paper_id"), "paper_title": row.get("paper_title", ""),
            "sources": row.get("sources", []),
        })
    return cleaned


def _rag_context(paper_id: int, message: str) -> tuple[str, list[dict[str, Any]]]:
    rows = retrieval.hybrid_search(message, DB_PATH, VECTORS_DIR, paper_id=paper_id, top_k=10)
    if rows:
        return retrieval.build_context(rows, max_chars=30000, per_row_chars=1800), rows
    return _paper_context(paper_id), []


# --- Background import jobs -----------------------------------------------------------

def _set_progress(paper_id: int, stage: str, current: int = 0, total: int = 0,
                  status: Optional[str] = None, error: Optional[str] = None) -> None:
    assignments = ["progress_stage=?", "progress_current=?", "progress_total=?"]
    params: list[Any] = [stage, current, total]
    if status is not None:
        assignments.append("status=?")
        params.append(status)
    if error is not None:
        assignments.append("error_message=?")
        params.append(error)
    params.append(paper_id)
    with get_connection() as connection:
        connection.execute(f"UPDATE papers SET {', '.join(assignments)} WHERE id=?", params)
        connection.commit()


def _progress_callback(paper_id: int):
    last_stage = {"value": ""}

    def report(stage: str, current: int, total: int) -> None:
        try:
            _set_progress(paper_id, stage, current, total)
        except Exception:
            LOGGER.warning("failed to persist progress for paper %s", paper_id, exc_info=True)
        if stage != last_stage["value"]:
            last_stage["value"] = stage
            observability.log_event("paper.progress", paper_id=paper_id, stage=stage,
                                    current=current, total=total)
    return report


def _ocr_options() -> dict[str, Any]:
    """Read local OCR settings without letting a bad value abort an import."""
    enabled = _setting("ocr_enabled", "true").lower() not in {"0", "false", "no", "off"}
    try:
        dpi = int(_setting("ocr_dpi", str(ocr_module.DEFAULT_DPI)))
    except ValueError:
        dpi = ocr_module.DEFAULT_DPI
    engine_name = _setting("ocr_engine", "auto")
    engine: Any = engine_name
    if engine_name.strip().lower() == "aliyun":
        engine = ocr_module.AliyunOcrEngine(
            access_key_id=_setting("ocr_access_key_id", ""),
            access_key_secret=safe_key(ocr_module.ALIYUN_OCR_KEY) or "",
            endpoint=_setting("ocr_endpoint", ocr_module.DEFAULT_ALIYUN_ENDPOINT),
            ocr_type=_setting("ocr_type", ocr_module.DEFAULT_ALIYUN_OCR_TYPE),
        )
    return {
        "enable_ocr": enabled,
        "ocr_engine": engine,
        "ocr_lang": _setting("ocr_lang", ocr_module.DEFAULT_LANG),
        "ocr_dpi": max(72, min(dpi, 600)),
    }


def _run_import_job(paper_id: int, stored_path: str) -> None:
    started = time.time()
    observability.log_event("paper.import.start", paper_id=paper_id, path=stored_path)
    try:
        _set_progress(paper_id, "parsing", status="processing", error=None)
        document = fitz.open(stored_path)
        try:
            if len(document) > MAX_PAGES:
                raise ValueError("单个 PDF 不得超过 300 页")
            process_document(
                paper_id, document, DB_PATH, VECTORS_DIR,
                embedding_key=_safe_qwen_key(), embedding_model=_embedding_model(),
                progress=_progress_callback(paper_id),
                **_ocr_options(),
            )
        finally:
            document.close()
    except Exception as error:
        LOGGER.exception("import failed for paper %s", paper_id)
        _set_progress(paper_id, "failed", status="failed", error=str(error))
        observability.log_event("paper.import.failed", level=logging.ERROR,
                                paper_id=paper_id, error=str(error),
                                elapsed_ms=int((time.time() - started) * 1000))
    else:
        observability.log_event("paper.import.done", paper_id=paper_id,
                                elapsed_ms=int((time.time() - started) * 1000))


def _run_reindex_job(paper_id: int, key: str) -> None:
    started = time.time()
    observability.log_event("paper.reindex.start", paper_id=paper_id)
    try:
        _set_progress(paper_id, "queued")
        reindex_document(paper_id, DB_PATH, VECTORS_DIR, key,
                         embedding_model=_embedding_model(),
                         progress=_progress_callback(paper_id))
    except Exception as error:
        LOGGER.exception("reindex failed for paper %s", paper_id)
        # The extracted text and FTS index are still usable, so keep the paper
        # readable and surface the failure through progress/error_message.
        _set_progress(paper_id, "index_failed", error=str(error))
        observability.log_event("paper.reindex.failed", level=logging.ERROR,
                                paper_id=paper_id, error=str(error),
                                elapsed_ms=int((time.time() - started) * 1000))
    else:
        observability.log_event("paper.reindex.done", paper_id=paper_id,
                                elapsed_ms=int((time.time() - started) * 1000))


def _enqueue_import(paper_id: int, stored_path: Path) -> None:
    _set_progress(paper_id, "queued", status="processing", error=None)
    _IMPORT_EXECUTOR.submit(_run_import_job, paper_id, str(stored_path))


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
    stream: bool = False
    retry: bool = False
    session_id: Optional[int] = None


class AgentChatPayload(BaseModel):
    message: str = Field(default="", max_length=12000)
    question: str = Field(default="", max_length=12000)
    paper_id: Optional[int] = None
    mode: str = "auto"  # auto | paper | library | general
    provider: Optional[str] = None
    stream: bool = True
    retry: bool = False
    session_id: Optional[int] = None


class SessionTitlePayload(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class ModelsPayload(BaseModel):
    provider: str
    api_key: Optional[str] = None
    protocol: Optional[str] = None
    base_url: Optional[str] = None


class ClinicalCalculatePayload(BaseModel):
    calculator: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class EvidenceAppraisePayload(BaseModel):
    text: str = ""
    paper_id: Optional[int] = None
    title: str = ""


class MemoryPayload(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    tags: list[str] = Field(default_factory=list)
    source: str = Field(default="user", max_length=80)


def _key_store(provider: str, value: Optional[str] = None) -> Optional[str]:
    """Store API keys in the operating system credential manager."""
    try:
        provider = provider.lower().strip()
        # OCR and rerank use dedicated credential slots rather than chat
        # providers, so they must be accepted here too.
        if provider not in providers.PROVIDERS and provider not in EXTRA_KEY_SOURCES:
            raise HTTPException(400, "未知的模型供应商")
        return _credential_key_store(provider, value)
    except CredentialError as error:
        raise HTTPException(503, str(error)) from error


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


def _setting(key: str, default: str = "") -> str:
    with get_connection() as connection:
        row = connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    value = row["value"].strip() if row and row["value"] else ""
    return value or default


def _provider_config(provider: Optional[str] = None) -> providers.ProviderConfig:
    """Resolve the selected provider, including custom protocol/base URL."""
    saved = (_setting("provider") or "deepseek").lower().strip()
    selected = (provider or saved).lower().strip()
    same_as_saved = selected == saved
    protocol = (_setting("protocol") or None) if same_as_saved else None
    base_url = (_setting("base_url") or None) if same_as_saved else None
    try:
        return providers.resolve_config(selected, protocol=protocol, base_url=base_url)
    except providers.ProviderError as error:
        raise HTTPException(400, str(error)) from error


def _provider_chat(provider: str, key: str, prompt: str, model: str) -> str:
    config = _provider_config(provider)
    try:
        return providers.chat(config, key, model, [{"role": "user", "content": prompt}],
                              timeout=45)
    except providers.ProviderError as error:
        raise HTTPException(502, str(error)) from error


def _stream_provider_chat(provider: str, key: str, prompt: str, model: str):
    config = _provider_config(provider)
    try:
        yield from providers.stream(config, key, model, [{"role": "user", "content": prompt}],
                                     timeout=120)
    except providers.ProviderError as error:
        raise HTTPException(502, str(error)) from error


def _resolve_model(provider: str | providers.ProviderConfig) -> str:
    config = provider if isinstance(provider, providers.ProviderConfig) else _provider_config(provider)
    saved = (_setting("provider") or "deepseek").lower().strip()
    model = _setting("model") if config.id == saved else ""
    return model or config.default_model


def _chat_target(provider: Optional[str] = None) -> tuple[providers.ProviderConfig, str, str]:
    """Resolve provider, credential and model, with consistent 400 messages."""
    config = _provider_config(provider)
    key = ""
    try:
        key = _key_store(config.id) or ""
    except HTTPException:
        if config.requires_key:
            raise
    if config.requires_key and not key:
        raise HTTPException(400, f"{config.label} 尚未配置 API Key，请先在设置中填写")
    model = _resolve_model(config)
    if not model:
        raise HTTPException(400, "请先在设置中填写模型名称")
    return config, key, model


def _format_turns(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}" for m in messages)


def _split_history(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(messages) <= RECENT_TURNS * 2:
        return [], messages
    return messages[:-RECENT_TURNS * 2], messages[-RECENT_TURNS * 2:]


def _summarize_history(provider: str, key: str, model: str, existing_summary: str, uncovered: list[dict[str, Any]]) -> str:
    base = (existing_summary + "\n\n" if existing_summary else "") + _format_turns(uncovered)
    prompt = ("将以下对话历史压缩成简洁摘要（不超过300字），保留关键事实、用户关注点、"
              "重要结论和指代关系，供后续追问参考。只输出摘要正文。\n\n对话历史：\n" + base)
    try:
        return _provider_chat(provider, key, prompt, model).strip()
    except Exception:
        return existing_summary or ""


def _paper_conversation_state(paper_id: int, provider: str, key: str, model: str) -> tuple[str, list[dict[str, Any]]]:
    with get_connection() as connection:
        messages = [dict(row) for row in connection.execute(
            "SELECT id, role, content FROM chat_messages WHERE paper_id=? ORDER BY id", (paper_id,)
        ).fetchall()]
        row = connection.execute("SELECT summary, summarized_upto FROM chat_summaries WHERE paper_id=?", (paper_id,)).fetchone()
    older, recent = _split_history(messages)
    if not older:
        return "", recent
    last_older_id = older[-1]["id"]
    summary = row["summary"] if row else ""
    covered = row["summarized_upto"] if row else 0
    if covered >= last_older_id:
        return summary, recent
    uncovered = [m for m in older if m["id"] > covered]
    summary = _summarize_history(provider, key, model, summary, uncovered)
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO chat_summaries(paper_id, summary, summarized_upto, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(paper_id) DO UPDATE SET summary=excluded.summary, summarized_upto=excluded.summarized_upto, updated_at=excluded.updated_at",
            (paper_id, summary, last_older_id, utc_now()),
        )
    return summary, recent


def _resolve_chat_session(session_id: Optional[int] = None,
                          create: bool = False) -> Optional[int]:
    """Resolve a general-chat session, falling back to the most recent one.

    ``create`` lets first-time or legacy clients start a session implicitly.
    An explicit id that does not exist is a client error, not a new session.
    """
    with get_connection() as connection:
        if session_id is not None:
            row = connection.execute(
                "SELECT id FROM general_sessions WHERE id=?", (session_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(404, "会话不存在")
            return int(row["id"])
        row = connection.execute(
            "SELECT id FROM general_sessions ORDER BY updated_at DESC, id DESC LIMIT 1"
        ).fetchone()
        if row is not None:
            return int(row["id"])
        if not create:
            return None
        now = utc_now()
        cursor = connection.execute(
            "INSERT INTO general_sessions(title,created_at,updated_at) VALUES('',?,?)",
            (now, now),
        )
        return int(cursor.lastrowid)


def _session_title(message: str) -> str:
    line = next((item.strip() for item in message.splitlines() if item.strip()), "")
    return line[:40] + "..." if len(line) > 40 else line


def _touch_chat_session(connection, session_id: int, title: str = "") -> None:
    if title:
        connection.execute(
            "UPDATE general_sessions SET title=CASE WHEN title='' THEN ? ELSE title END,"
            " updated_at=? WHERE id=?",
            (title, utc_now(), session_id),
        )
    else:
        connection.execute(
            "UPDATE general_sessions SET updated_at=? WHERE id=?", (utc_now(), session_id))


def _general_conversation_state(provider: str, key: str, model: str,
                                session_id: int) -> tuple[str, list[dict[str, Any]]]:
    with get_connection() as connection:
        messages = [dict(row) for row in connection.execute(
            "SELECT id, role, content FROM general_messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()]
        row = connection.execute(
            "SELECT summary, summarized_upto FROM general_session_summaries WHERE session_id=?",
            (session_id,),
        ).fetchone()
    older, recent = _split_history(messages)
    if not older:
        return "", recent
    last_older_id = older[-1]["id"]
    summary = row["summary"] if row else ""
    covered = row["summarized_upto"] if row else 0
    if covered >= last_older_id:
        return summary, recent
    uncovered = [m for m in older if m["id"] > covered]
    summary = _summarize_history(provider, key, model, summary, uncovered)
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO general_session_summaries(session_id, summary, summarized_upto, updated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, "
            "summarized_upto=excluded.summarized_upto, updated_at=excluded.updated_at",
            (session_id, summary, last_older_id, utc_now()),
        )
    return summary, recent


def _build_paper_prompt(summary: str, recent: list[dict[str, Any]], context: str, message: str) -> str:
    parts = ["你是医学论文精读助手。用户正在针对同一篇论文连续提问，请结合对话历史理解用户问题中的指代和上下文。仅依据下面给出的论文内容回答；必须在回答中引用一个或多个 [Page N] 页码；不确定时明确说无法从论文判断。\n" + MEDICAL_SYSTEM_RULES]
    if summary:
        parts.append("【历史对话摘要】\n" + summary)
    if recent:
        parts.append("【最近对话】\n" + _format_turns(recent))
    parts.append("【论文内容】\n" + wrap_untrusted(context))
    parts.append("【当前问题】\n" + message)
    return "\n\n".join(parts)


def _build_general_prompt(summary: str, recent: list[dict[str, Any]], message: str) -> str:
    parts = ["你是面向医学生的临床医学学习助手，用清晰、准确、易懂的语言回答；涉及医学事实时尽量严谨，不确定时如实说明。请结合对话历史理解用户的追问和指代。\n" + MEDICAL_SYSTEM_RULES]
    if summary:
        parts.append("【历史对话摘要】\n" + summary)
    if recent:
        parts.append("【最近对话】\n" + _format_turns(recent))
    parts.append("【当前问题】\n" + message)
    return "\n\n".join(parts)


def _paper_chat_stream(paper_id: int, provider: str, key: str, model: str, prompt: str,
                       message: str, grounding: str, retrieved: list[dict[str, Any]]):
    parts: list[str] = []
    try:
        for delta in _stream_provider_chat(provider, key, prompt, model):
            parts.append(delta)
            yield json.dumps({"delta": delta}, ensure_ascii=False) + "\n"
    except HTTPException as error:
        yield json.dumps({"error": error.detail}, ensure_ascii=False) + "\n"
        return
    raw = "".join(parts).strip()
    try:
        answer = _grounded_response(message, raw, grounding)
    except HTTPException as error:
        yield json.dumps({"error": error.detail}, ensure_ascii=False) + "\n"
        return
    answer, _ = apply_safety(answer, message)
    with get_connection() as connection:
        cursor = connection.execute("INSERT INTO chat_messages(paper_id,role,content,provider,created_at) VALUES(?,?,?,?,?)",
                                     (paper_id, "assistant", answer, provider, utc_now()))
    citations = []
    for page_number in sorted({int(number) for number in re.findall(r"\[Page\s+(\d+)\]", answer)}):
        citations.append({"id": page_number, "page_number": page_number, "text": f"[Page {page_number}]"})
    yield json.dumps({"done": True, "id": cursor.lastrowid, "content": answer,
                      "citations": citations, "sources": _clean_chunks(retrieved),
                      "citation_level": "page",
                      "warnings": ["该对话接口只校验页码，未做逐字引文与论断核验；请优先使用 Agent 问答。"]},
                     ensure_ascii=False) + "\n"


def _general_chat_stream(provider: str, key: str, model: str, prompt: str,
                         question: str = "", session_id: Optional[int] = None):
    parts: list[str] = []
    try:
        for delta in _stream_provider_chat(provider, key, prompt, model):
            parts.append(delta)
            yield json.dumps({"delta": delta}, ensure_ascii=False) + "\n"
    except HTTPException as error:
        yield json.dumps({"error": error.detail}, ensure_ascii=False) + "\n"
        return
    answer = "".join(parts).strip()
    if not answer:
        yield json.dumps({"error": "模型返回了空响应"}, ensure_ascii=False) + "\n"
        return
    answer, _ = apply_safety(answer, question)
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO general_messages(session_id,role,content,provider,created_at) "
            "VALUES(?,?,?,?,?)",
            (session_id, "assistant", answer, provider, utc_now()))
        if session_id is not None:
            _touch_chat_session(connection, session_id)
    yield json.dumps({"done": True, "id": cursor.lastrowid, "content": answer}, ensure_ascii=False) + "\n"


app = FastAPI(title="觅心 HeartSeeker_Med Local API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    initialize_storage()
    observability.setup_logging()
    # Bind telemetry to this process's database only once the real server runs;
    # importing the app (for tests) must never touch the live usage table.
    providers.set_usage_recorder(
        lambda usage: observability.record_usage(usage, DB_PATH))
    observability.log_event("app.started", version=app.version, storage=str(APP_HOME))


@app.on_event("shutdown")
def shutdown() -> None:
    _IMPORT_EXECUTOR.shutdown(wait=False, cancel_futures=True)


DIST_DIR = Path(__file__).resolve().parents[1] / "dist"


def _frontend_built() -> bool:
    return (DIST_DIR / "index.html").is_file()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "storage": str(APP_HOME),
            "frontend": "built" if _frontend_built() else "source"}


@app.get("/api/stats")
def usage_stats() -> dict[str, Any]:
    """Aggregate token, agent, tool and library usage for the stats page."""
    day_start = datetime.now().astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0
    ).astimezone(timezone.utc).isoformat()
    with get_connection() as connection:
        tokens = dict(connection.execute(
            "SELECT COUNT(*) AS calls,"
            " COALESCE(SUM(prompt_tokens),0) AS prompt,"
            " COALESCE(SUM(completion_tokens),0) AS completion,"
            " COALESCE(SUM(total_tokens),0) AS total,"
            " COALESCE(SUM(estimated),0) AS estimated_calls,"
            " COALESCE(SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END),0) AS today_calls,"
            " COALESCE(SUM(CASE WHEN created_at >= ? THEN total_tokens ELSE 0 END),0)"
            " AS today_total"
            " FROM llm_usage",
            (day_start, day_start),
        ).fetchone())
        runs = dict(connection.execute(
            "SELECT COUNT(*) AS total,"
            " COALESCE(SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END),0) AS today_runs,"
            " COALESCE(SUM(verified),0) AS verified"
            " FROM agent_runs",
            (day_start,),
        ).fetchone())
        average = connection.execute(
            "SELECT AVG(steps) FROM (SELECT COUNT(*) AS steps FROM agent_steps GROUP BY run_id)"
        ).fetchone()
        runs["avg_steps"] = round(float(average[0] or 0), 2)
        maps = {
            "modes": "SELECT mode, COUNT(*) AS count FROM agent_runs"
                     " GROUP BY mode ORDER BY count DESC, mode",
            "tools": "SELECT tool, COUNT(*) AS count FROM agent_steps"
                     " GROUP BY tool ORDER BY count DESC, tool",
            "models": "SELECT provider, model, COUNT(*) AS calls,"
                      " COALESCE(SUM(total_tokens),0) AS total_tokens,"
                      " COALESCE(SUM(estimated),0) AS estimated_calls"
                      " FROM llm_usage GROUP BY provider, model"
                      " ORDER BY total_tokens DESC, calls DESC LIMIT 12",
        }
        grouped = {name: [dict(row) for row in connection.execute(sql)]
                   for name, sql in maps.items()}
        sessions = dict(connection.execute(
            "SELECT COUNT(*) AS total FROM general_sessions").fetchone())
        sessions["messages"] = connection.execute(
            "SELECT COUNT(*) FROM general_messages").fetchone()[0]
        papers = dict(connection.execute(
            "SELECT COUNT(*) AS total,"
            " COALESCE(SUM(needs_ocr),0) AS needs_ocr,"
            " COALESCE(SUM(vectorized),0) AS vectorized"
            " FROM papers").fetchone())
    return {
        "tokens": tokens,
        "runs": {**runs, "modes": grouped["modes"]},
        "tools": {
            "total": sum(item["count"] for item in grouped["tools"]),
            "by_tool": grouped["tools"],
        },
        "models": grouped["models"],
        "sessions": sessions,
        "papers": papers,
    }


@app.get("/api/logs")
def list_logs(after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
    """Structured events newer than ``after`` (bounded in-memory ring buffer)."""
    return observability.recent(after, limit)


@app.get("/api/papers")
def list_papers() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, title, file_name, page_count, status, error_message, imported_at, vectorized, needs_ocr, "
            "embedding_model, embedding_dim, progress_stage, progress_current, progress_total "
            "FROM papers ORDER BY imported_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def _search_payload(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**chunk, "source": "hybrid"} for chunk in _clean_chunks(rows)]


@app.get("/api/library/search")
def library_search(q: str, top_k: int = 8) -> list[dict[str, Any]]:
    """Hybrid search across every imported paper."""
    query = q.strip()
    if not query:
        return []
    top_k = max(1, min(top_k, 50))
    rows = retrieval.library_search(query, DB_PATH, VECTORS_DIR, top_k=top_k)
    return _search_payload(rows)


# --- local medical knowledge ---------------------------------------------------------

@app.get("/api/clinical/calculators")
def list_clinical_calculators() -> list[dict[str, Any]]:
    return clinical.list_calculators()


@app.post("/api/clinical/calculate")
def run_clinical_calculator(payload: ClinicalCalculatePayload) -> dict[str, Any]:
    try:
        return clinical.calculate(payload.calculator, payload.inputs)
    except clinical.CalculatorError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/ontology/lookup")
def lookup_medical_ontology(q: str, limit: int = 5) -> dict[str, Any]:
    query = q.strip()
    if not query:
        return {"query": "", "matches": [], "version": ontology.get_ontology().version}
    try:
        matches = ontology.lookup(query, limit=max(1, min(limit, 20)))
    except ontology.OntologyError as error:
        raise HTTPException(503, str(error)) from error
    return {"query": query, "matches": matches,
            "version": ontology.get_ontology().version}


@app.get("/api/ontology/stats")
def ontology_stats() -> dict[str, Any]:
    try:
        return ontology.stats()
    except ontology.OntologyError as error:
        raise HTTPException(503, str(error)) from error


@app.post("/api/evidence/appraise")
def appraise_evidence(payload: EvidenceAppraisePayload) -> dict[str, Any]:
    text = payload.text.strip()
    try:
        if text:
            return evidence.appraise(text, title=payload.title)
        if payload.paper_id is None:
            raise HTTPException(422, "请提供 text 或 paper_id")
        return evidence.appraise_paper(payload.paper_id, DB_PATH)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def _store_agent_message(mode: str, paper_id: Optional[int], role: str,
                         content: str, provider: str,
                         session_id: Optional[int] = None) -> None:
    if not content.strip():
        return
    with get_connection() as connection:
        if mode == "paper":
            if paper_id is None:
                return
            connection.execute(
                "INSERT INTO chat_messages(paper_id,role,content,provider,created_at) VALUES(?,?,?,?,?)",
                (paper_id, role, content, provider, utc_now()),
            )
        else:
            connection.execute(
                "INSERT INTO general_messages(session_id,role,content,provider,created_at) "
                "VALUES(?,?,?,?,?)",
                (session_id, role, content, provider, utc_now()),
            )
            if session_id is not None:
                _touch_chat_session(
                    connection, session_id,
                    _session_title(content) if role == "user" else "",
                )


def _agent_history(mode: str, paper_id: Optional[int], retry: bool,
                   message: str, session_id: Optional[int] = None) -> list[dict[str, Any]]:
    with get_connection() as connection:
        if mode == "paper" and paper_id is not None:
            rows = connection.execute(
                "SELECT role,content FROM chat_messages WHERE paper_id=? ORDER BY id DESC LIMIT 20",
                (paper_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT role,content FROM general_messages WHERE session_id=? "
                "ORDER BY id DESC LIMIT 20",
                (session_id,),
            ).fetchall()
    history = [dict(row) for row in reversed(rows)]
    if retry and history and history[-1]["role"] == "user" \
            and history[-1]["content"].strip() == message:
        history = history[:-1]
    return history


@app.post("/api/agent/chat")
def agent_chat(payload: AgentChatPayload):
    """Run the tool-using agent over a paper, the library, or general knowledge."""
    message = (payload.message or payload.question).strip()
    if not message:
        raise HTTPException(400, "问题不能为空")
    endpoint, key, model = _chat_target(payload.provider)
    provider = endpoint.id

    mode = (payload.mode or "auto").lower().strip()
    if mode not in {"auto", "paper", "library", "general"}:
        raise HTTPException(400, "mode 必须是 auto / paper / library / general")
    paper_id = payload.paper_id
    if mode == "auto":
        if paper_id is not None:
            mode = "paper"
        else:
            with get_connection() as connection:
                has_papers = connection.execute("SELECT 1 FROM papers LIMIT 1").fetchone() is not None
            mode = "library" if has_papers else "general"
    if mode == "paper":
        if paper_id is None:
            raise HTTPException(400, "论文内问答需要 paper_id")
        if not retrieval.paper_exists(DB_PATH, paper_id):
            raise HTTPException(404, "论文不存在")
    elif paper_id is not None:
        mode = "paper"

    # General/library conversations are grouped into sessions so a new chat
    # starts with fresh recent-turn memory instead of inheriting an old one.
    session_id = None
    if mode != "paper":
        session_id = _resolve_chat_session(payload.session_id, create=True)
    history = _agent_history(mode, paper_id, payload.retry, message, session_id)
    try:
        token_budget_value = int(_setting("context_token_budget", "24000"))
    except ValueError:
        token_budget_value = 24000
    agent_config = AgentConfig(provider=provider, key=key, model=model, mode=mode,
                               protocol=endpoint.protocol, base_url=endpoint.base_url,
                               paper_id=paper_id, history=history,
                               max_context_tokens=max(2000, min(token_budget_value, 200000)))

    stream_fn = None
    if payload.stream:
        def stream_fn(messages):
            emitted = False
            try:
                for piece in providers.stream(endpoint, key, model, messages,
                                              temperature=agent_config.temperature):
                    emitted = True
                    yield piece
            except providers.ProviderError:
                if emitted:
                    raise
                # Some OpenAI-compatible servers do not implement SSE; fall
                # back to a normal completion instead of losing the answer.
                yield providers.chat(endpoint, key, model, messages,
                                     temperature=agent_config.temperature)

    runner = AgentRunner(agent_config, DB_PATH, VECTORS_DIR, stream_fn=stream_fn)
    observability.log_event(
        "agent.chat.start", mode=mode, provider=provider, model=model,
        paper_id=paper_id, session_id=session_id, stream=payload.stream,
        retry=payload.retry, question_chars=len(message),
    )

    if not payload.stream:
        started = time.time()
        try:
            done = runner.run(message)
        except AgentError as error:
            observability.log_event("agent.chat.error", level=logging.WARNING,
                                    mode=mode, error=str(error))
            raise HTTPException(502, str(error)) from error
        if not payload.retry:
            _store_agent_message(mode, paper_id, "user", message, provider, session_id)
        _store_agent_message(mode, paper_id, "assistant", done.get("content", ""),
                             provider, session_id)
        observability.log_event(
            "agent.chat.done", mode=mode, run_id=done.get("id"),
            verified=bool(done.get("verified")), steps=len(done.get("trace") or []),
            warnings=len(done.get("warnings") or []),
            elapsed_ms=int((time.time() - started) * 1000),
        )
        return done

    if not payload.retry:
        _store_agent_message(mode, paper_id, "user", message, provider, session_id)

    def generate():
        started = time.time()
        for event in runner.run_iter(message):
            kind = event.get("type")
            if kind == "step":
                observation = event.get("observation") or {}
                observability.log_event(
                    "agent.step", step=event.get("index"), tool=event.get("tool"),
                    ok=bool(observation.get("ok")) if isinstance(observation, dict) else False,
                    duration_ms=event.get("duration_ms"),
                )
            elif kind == "done":
                _store_agent_message(mode, paper_id, "assistant", event.get("content", ""),
                                     provider, session_id)
                observability.log_event(
                    "agent.chat.done", mode=mode, run_id=event.get("id"),
                    verified=bool(event.get("verified")),
                    steps=len(event.get("trace") or []),
                    warnings=len(event.get("warnings") or []),
                    elapsed_ms=int((time.time() - started) * 1000),
                )
            elif kind == "error":
                observability.log_event("agent.chat.error", level=logging.WARNING,
                                        mode=mode, error=str(event.get("error") or ""))
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/api/agent/runs/{run_id}")
def get_agent_run(run_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        run = connection.execute("SELECT * FROM agent_runs WHERE id=?", (run_id,)).fetchone()
        if run is None:
            raise HTTPException(404, "Agent 运行记录不存在")
        steps = connection.execute(
            "SELECT step_index,thought,tool,args_json,observation_json,duration_ms,created_at "
            "FROM agent_steps WHERE run_id=? ORDER BY step_index", (run_id,)
        ).fetchall()
    data = dict(run)
    for key in ("trace_json", "citations_json", "warnings_json"):
        try:
            data[key] = json.loads(data[key])
        except (TypeError, ValueError):
            data[key] = []
    step_rows = []
    for step in steps:
        item = dict(step)
        for key in ("args_json", "observation_json"):
            try:
                item[key] = json.loads(item[key])
            except (TypeError, ValueError):
                item[key] = {}
        step_rows.append(item)
    return {"run": data, "steps": step_rows}


def _markdown_cell(value: Any) -> str:
    text = str(value or "").replace("|", "\\|").replace("\n", " ")
    return text[:500]


@app.get("/api/agent/runs/{run_id}/export")
def export_agent_run(run_id: int, format: str = "md") -> Response:
    """Export one run as Markdown, JSON or CSV (citations only)."""
    with get_connection() as connection:
        run = connection.execute("SELECT * FROM agent_runs WHERE id=?", (run_id,)).fetchone()
        if run is None:
            raise HTTPException(404, "Agent 运行记录不存在")
        steps = [dict(step) for step in connection.execute(
            "SELECT step_index,thought,tool,args_json,observation_json,duration_ms "
            "FROM agent_steps WHERE run_id=? ORDER BY step_index", (run_id,)
        ).fetchall()]
    data = dict(run)
    for key in ("trace_json", "citations_json", "warnings_json"):
        try:
            data[key] = json.loads(data[key])
        except (TypeError, ValueError):
            data[key] = []
    fmt = (format or "md").strip().lower()
    if fmt not in {"md", "markdown", "json", "csv"}:
        raise HTTPException(400, "format 必须是 md、json 或 csv")
    filename = f"heartseeker-med-run-{run_id}"

    if fmt in {"md", "markdown"}:
        citations = data["citations_json"]
        lines = [
            "# 觅心 回答", "",
            f"- 问题：{data['question']}",
            f"- 模式：{data['mode']}",
            f"- 模型：{data.get('provider') or '—'} / {data.get('model') or '—'}",
            f"- 引用核验：{'通过' if data.get('verified') else '未通过'}"
            f"（{data.get('citation_level') or 'none'}）",
            f"- 时间：{data.get('created_at') or '—'}", "",
            "## 回答", "", str(data.get("answer") or ""), "",
            "## 引用", "",
            "| 论文 | 页码 | 论断 | 原文 |",
            "| --- | --- | --- | --- |",
        ]
        for item in citations:
            lines.append(
                f"| {_markdown_cell(item.get('paper_id'))} "
                f"| {_markdown_cell(item.get('page'))} "
                f"| {_markdown_cell(item.get('claim'))} "
                f"| {_markdown_cell(item.get('quote'))} |"
            )
        if data["warnings_json"]:
            lines.extend(["", "## 警告", ""])
            lines.extend(f"- {warning}" for warning in data["warnings_json"])
        if steps:
            lines.extend(["", "## 执行轨迹", ""])
            for step in steps:
                lines.append(f"{step['step_index']}. **{step['tool']}** "
                             f"({step.get('duration_ms') or 0} ms)")
                if step.get("thought"):
                    lines.append(f"   - 思考：{step['thought']}")
        body = "\n".join(lines) + "\n"
        return Response(body, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{filename}.md"'})

    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["paper_id", "page", "claim", "quote", "verified",
                         "claim_supported", "support_score"])
        for item in data["citations_json"]:
            writer.writerow([item.get("paper_id"), item.get("page"),
                             item.get("claim"), item.get("quote"),
                             item.get("verified"), item.get("claim_supported"),
                             item.get("support_score")])
        return Response("\ufeff" + buffer.getvalue(),
                        media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{filename}.csv"'})

    payload = {"run": data,
               "steps": [dict(step) for step in steps]}
    for step in payload["steps"]:
        for key in ("args_json", "observation_json"):
            try:
                step[key] = json.loads(step[key])
            except (TypeError, ValueError):
                step[key] = {}
    return Response(json.dumps(payload, ensure_ascii=False, indent=2),
                    media_type="application/json; charset=utf-8",
                    headers={"Content-Disposition":
                             f'attachment; filename="{filename}.json"'})


@app.get("/api/memory")
def list_memory(limit: int = 50) -> list[dict[str, Any]]:
    return memory.list_memories(DB_PATH, limit=limit)


@app.post("/api/memory")
def add_memory(payload: MemoryPayload) -> dict[str, Any]:
    try:
        return memory.remember(DB_PATH, payload.text, tags=payload.tags,
                               source=payload.source or "user")
    except memory.MemoryError as error:
        raise HTTPException(422, str(error)) from error


@app.delete("/api/memory/{memory_id}")
def delete_memory(memory_id: int) -> dict[str, Any]:
    if not memory.forget(DB_PATH, memory_id):
        raise HTTPException(404, "笔记不存在")
    return {"id": memory_id, "status": "deleted"}


class FeedbackPayload(BaseModel):
    agent_run_id: Optional[int] = None
    paper_id: Optional[int] = None
    rating: int = Field(ge=-1, le=1)
    comment: str = Field(default="", max_length=4000)


@app.post("/api/feedback")
def add_feedback(payload: FeedbackPayload) -> dict[str, Any]:
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO feedback(agent_run_id,paper_id,rating,comment,created_at) VALUES(?,?,?,?,?)",
            (payload.agent_run_id, payload.paper_id, payload.rating,
             payload.comment.strip(), utc_now()),
        )
    return {"id": cursor.lastrowid, "status": "saved"}


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


def _structured_analysis(paper_id: int, force_llm: bool) -> dict[str, Any]:
    _paper_context(paper_id)
    endpoint = _provider_config()
    provider = endpoint.id
    key = safe_key(provider) or ""
    model = _resolve_model(endpoint)
    analysis = extract_fields(paper_id, DB_PATH, provider, key, model,
                              force=force_llm, use_llm=force_llm,
                              protocol=endpoint.protocol, base_url=endpoint.base_url)
    analysis["provider"] = provider
    analysis["model"] = model
    analysis["llm_available"] = bool(key) or not endpoint.requires_key
    return analysis


@app.get("/api/papers/{paper_id}/structured")
def structured_paper(paper_id: int, refresh: bool = False) -> dict[str, Any]:
    """Return cached LLM extraction when available, otherwise heuristic fields.

    A normal page load never spends an LLM call.  ``POST`` or ``?refresh=true``
    explicitly asks for a fresh structured reading.
    """
    try:
        return _structured_analysis(paper_id, force_llm=refresh)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/papers/{paper_id}/structured")
def refresh_structured_paper(paper_id: int) -> dict[str, Any]:
    endpoint = _provider_config()
    if endpoint.requires_key and not safe_key(endpoint.id):
        raise HTTPException(400, "请先在设置中配置问答模型 API Key")
    if not _resolve_model(endpoint):
        raise HTTPException(400, "请先在设置中填写模型名称")
    try:
        return _structured_analysis(paper_id, force_llm=True)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/papers/import")
async def import_paper(file: UploadFile = File(...)) -> dict[str, Any]:
    """Validate and persist the PDF, then parse/embed it on a background worker.

    The response returns as soon as the file is durable; the client polls
    ``/api/papers/{id}/status`` for progress.
    """
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
    try:
        with fitz.open(stored_path) as document:
            if len(document) > MAX_PAGES:
                raise ValueError("单个 PDF 不得超过 300 页")
            title = (document.metadata.get("title") or Path(file.filename).stem).strip() or "未命名论文"
            page_count = len(document)
    except Exception as error:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(422, f"解析 PDF 失败：{error}") from error

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                "INSERT INTO papers(title, file_name, stored_path, sha256, page_count, status, imported_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (title, file.filename, str(stored_path), digest, page_count, "processing", utc_now()),
            )
            paper_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO paper_metadata(paper_id, authors, abstract, keywords) VALUES(?,?,?,?)",
                (paper_id, "[]", "", "[]"),
            )
            connection.commit()
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
    _enqueue_import(paper_id, stored_path)
    return {"id": paper_id, "title": title, "status": "processing", "page_count": page_count}


@app.get("/api/papers/{paper_id}/status")
def paper_status(paper_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT id, status, error_message, page_count, vectorized, needs_ocr, embedding_model, "
            "embedding_dim, progress_stage, progress_current, progress_total FROM papers WHERE id=?",
            (paper_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, "论文不存在")
    result = dict(row)
    total = int(result.get("progress_total") or 0)
    current = int(result.get("progress_current") or 0)
    result["progress_ratio"] = round(current / total, 3) if total > 0 else 0.0
    return result


@app.post("/api/papers/{paper_id}/reindex")
def reindex_paper(paper_id: int) -> dict[str, Any]:
    """Re-embed existing chunks, e.g. after switching the embedding model."""
    key = _safe_qwen_key()
    if not key:
        raise HTTPException(400, "请先在设置中配置 Qwen（DashScope）嵌入 Key")
    with get_connection() as connection:
        paper = connection.execute(
            "SELECT status, (SELECT COUNT(*) FROM chunks WHERE paper_id=?) AS chunks "
            "FROM papers WHERE id=?",
            (paper_id, paper_id),
        ).fetchone()
    if paper is None:
        raise HTTPException(404, "论文不存在")
    if paper["status"] == "processing":
        raise HTTPException(409, "论文仍在导入中")
    if int(paper["chunks"] or 0) == 0:
        raise HTTPException(409, "论文没有可重建索引的切片，请先重试解析")
    _set_progress(paper_id, "queued", error=None)
    with get_connection() as connection:
        connection.execute("UPDATE papers SET error_message=NULL WHERE id=?", (paper_id,))
        connection.commit()
    _IMPORT_EXECUTOR.submit(_run_reindex_job, paper_id, key)
    return {"id": paper_id, "status": "processing", "chunks": int(paper["chunks"] or 0)}


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
    rows = retrieval.hybrid_search(query, DB_PATH, VECTORS_DIR, paper_id=paper_id, top_k=top_k)
    if rows:
        return [{**chunk, "source": "hybrid", "sources": chunk.get("sources", [])}
                for chunk in _clean_chunks(rows)]
    return search_paper(paper_id, query)


@app.get("/api/papers/{paper_id}/file")
def paper_file(paper_id: int):
    from fastapi.responses import FileResponse

    with get_connection() as connection:
        paper = connection.execute("SELECT stored_path, file_name FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if paper is None or not Path(paper["stored_path"]).is_file():
        raise HTTPException(404, "论文文件不存在")
    return FileResponse(paper["stored_path"], media_type="application/pdf", filename=paper["file_name"])


def _ocr_available(engine_name: str, values: Optional[dict[str, str]] = None) -> str:
    """Return the resolved OCR engine name, or ``"none"`` when unusable."""
    values = values or {}
    try:
        if (engine_name or "").strip().lower() == "aliyun":
            engine = ocr_module.AliyunOcrEngine(
                access_key_id=values.get("ocr_access_key_id", ""),
                access_key_secret=safe_key(ocr_module.ALIYUN_OCR_KEY) or "",
                endpoint=values.get("ocr_endpoint", ocr_module.DEFAULT_ALIYUN_ENDPOINT),
                ocr_type=values.get("ocr_type", ocr_module.DEFAULT_ALIYUN_OCR_TYPE),
            )
            return engine.name if engine.available() else "none"
        return ocr_module.get_engine(engine_name).name
    except Exception:
        return "none"


def _settings_payload(values: dict[str, str]) -> dict[str, Any]:
    provider = values.get("provider", "").lower().strip()
    spec = providers.PROVIDERS.get(provider)
    protocol = values.get("protocol", "")
    base_url = values.get("base_url", "")
    if spec is not None and provider != "custom":
        protocol = spec.protocol
    try:
        endpoint = providers.resolve_config(provider, protocol or None, base_url or None)
        resolved_protocol = endpoint.protocol
        default_base_url = spec.base_url if spec else ""
    except providers.ProviderError:
        resolved_protocol = protocol
        default_base_url = spec.base_url if spec else ""
    return {
        "provider": provider,
        "provider_label": spec.label if spec else "",
        "protocol": resolved_protocol,
        "default_protocol": spec.protocol if spec else "",
        "base_url": base_url,
        "default_base_url": default_base_url,
        "model": values.get("model", ""),
        "default_model": spec.default_model if spec else "",
        "embedding_provider": "qwen",
        "embedding_model": values.get("embedding_model", DEFAULT_EMBED_MODEL),
        "ocr_enabled": values.get("ocr_enabled", "true"),
        "ocr_engine": values.get("ocr_engine", "auto"),
        "ocr_lang": values.get("ocr_lang", ocr_module.DEFAULT_LANG),
        "ocr_dpi": values.get("ocr_dpi", str(ocr_module.DEFAULT_DPI)),
        "ocr_endpoint": values.get("ocr_endpoint", ocr_module.DEFAULT_ALIYUN_ENDPOINT),
        "ocr_type": values.get("ocr_type", ocr_module.DEFAULT_ALIYUN_OCR_TYPE),
        "ocr_access_key_id": values.get("ocr_access_key_id", ""),
        "ocr_key_configured": bool(safe_key(ocr_module.ALIYUN_OCR_KEY)),
        "ocr_env_configured": bool(
            os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
            and os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
        ),
        "ocr_available": _ocr_available(values.get("ocr_engine", "auto"), values),
        "rerank_enabled": values.get("rerank_enabled", "true"),
        "rerank_engine": values.get("rerank_engine", "lexical"),
        "rerank_model": values.get("rerank_model", ""),
        "rerank_base_url": values.get("rerank_base_url", rerank.DEFAULT_QWEN_RERANK_URL),
        "rerank_key_configured": bool(safe_key(rerank.QWEN_RERANK_KEY) or safe_key("qwen")),
        "relevance_min_coverage": values.get("relevance_min_coverage", "0"),
        "nli_enabled": values.get("nli_enabled", "true"),
        "nli_engine": values.get("nli_engine", "none"),
        "nli_model": values.get("nli_model", ""),
        "context_token_budget": values.get("context_token_budget", "24000"),
        "api_key_configured": bool(safe_key(provider)) if spec else False,
        "embedding_key_configured": bool(safe_key("qwen")),
        "providers": providers.public_catalog(),
        "protocols": [{"id": key, "label": label} for key, label in providers.PROTOCOLS.items()],
        "data_dir": str(APP_HOME),
    }


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    with get_connection() as connection:
        rows = connection.execute("SELECT key,value FROM settings").fetchall()
    return _settings_payload({row["key"]: row["value"] for row in rows})


@app.put("/api/settings")
def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider", "")).lower().strip()
    protocol = str(payload.get("protocol", "")).lower().strip()
    base_url = str(payload.get("base_url", "")).strip().rstrip("/")
    model = str(payload.get("model", "")).strip()
    embedding_model = str(payload.get("embedding_model", "")).strip() or DEFAULT_EMBED_MODEL
    api_key = str(payload.get("api_key", ""))
    embedding_api_key = str(payload.get("embedding_api_key", ""))
    rerank_api_key = str(payload.get("rerank_api_key", ""))
    ocr_access_key_secret = str(payload.get("ocr_access_key_secret", ""))

    if provider and provider not in providers.PROVIDERS:
        raise HTTPException(400, "未知的模型供应商")
    if provider:
        saved_protocol = protocol or providers.PROVIDERS[provider].protocol
        try:
            resolved = providers.resolve_config(provider, saved_protocol, base_url or None)
        except providers.ProviderError as error:
            raise HTTPException(400, str(error)) from error
        protocol = resolved.protocol
        base_url = resolved.base_url
    else:
        protocol = ""
        base_url = ""

    values = {
        "provider": provider,
        "protocol": protocol,
        "base_url": base_url,
        "model": model,
        "embedding_model": embedding_model,
    }
    # Optional local-pipeline settings are only written when supplied.
    for key in ("ocr_enabled", "ocr_engine", "ocr_lang", "ocr_dpi",
                "ocr_endpoint", "ocr_type", "ocr_access_key_id",
                "rerank_enabled", "rerank_engine", "rerank_model", "rerank_base_url",
                "relevance_min_coverage", "nli_enabled", "nli_engine", "nli_model",
                "context_token_budget"):
        if key in payload and str(payload[key]).strip() != "":
            values[key] = str(payload[key]).strip()
    if values.get("ocr_endpoint"):
        endpoint_value = values["ocr_endpoint"].split("://", 1)[-1]
        values["ocr_endpoint"] = endpoint_value.rstrip("/")
    for key, value in values.items():
        if len(key) > 100 or len(value) > 10000:
            raise HTTPException(400, "设置键或值过长")

    if api_key:
        if not provider:
            raise HTTPException(400, "请先选择模型供应商")
        _key_store(provider, api_key)
    if embedding_api_key:
        _key_store("qwen", embedding_api_key)
    if rerank_api_key:
        _key_store(rerank.QWEN_RERANK_KEY, rerank_api_key)
    if ocr_access_key_secret:
        _key_store(ocr_module.ALIYUN_OCR_KEY, ocr_access_key_secret)

    with get_connection() as connection:
        for key, value in values.items():
            connection.execute(
                "INSERT INTO settings(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (key, value, utc_now()),
            )
    return _settings_payload(values)


class OcrTestPayload(BaseModel):
    access_key_id: str = ""
    access_key_secret: str = ""
    endpoint: str = ""
    ocr_type: str = ""


@app.post("/api/ocr/test")
def test_ocr_connection(payload: OcrTestPayload) -> dict[str, Any]:
    """Render a small text image and verify the configured Aliyun OCR credentials."""
    access_key_id = payload.access_key_id.strip() or _setting("ocr_access_key_id", "")
    access_key_secret = (
        payload.access_key_secret.strip()
        or safe_key(ocr_module.ALIYUN_OCR_KEY)
        or ""
    )
    endpoint = payload.endpoint.strip() or _setting(
        "ocr_endpoint", ocr_module.DEFAULT_ALIYUN_ENDPOINT)
    ocr_type = payload.ocr_type.strip() or _setting(
        "ocr_type", ocr_module.DEFAULT_ALIYUN_OCR_TYPE)
    if not access_key_id or not access_key_secret:
        raise HTTPException(400, "请填写阿里云 AccessKey ID 和 AccessKey Secret")
    engine = ocr_module.AliyunOcrEngine(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        endpoint=endpoint,
        ocr_type=ocr_type,
        timeout=30.0,
    )
    if not engine.available():
        raise HTTPException(400, "阿里云 OCR SDK 不可用，请先安装 alibabacloud_ocr_api20210707")

    document = fitz.open()
    try:
        page = document.new_page(width=720, height=220)
        page.insert_text((60, 110), "HeartSeeker_Med OCR test 12345", fontsize=28)
        image = ocr_module.render_page_png(page, dpi=150)
    finally:
        document.close()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(image)
        image_path = Path(handle.name)
    started = time.monotonic()
    try:
        text = engine.ocr_image(image_path, "")
    except ocr_module.OcrError as error:
        raise HTTPException(400, str(error)) from error
    finally:
        image_path.unlink(missing_ok=True)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    preview = text.strip()[:200]
    if not preview:
        raise HTTPException(400, "连接成功，但测试图片未识别出文字")
    return {
        "ok": True,
        "engine": "aliyun",
        "text": preview,
        "elapsed_ms": elapsed_ms,
    }


@app.get("/api/providers")
def list_providers() -> dict[str, Any]:
    return {
        "providers": providers.public_catalog(),
        "protocols": [{"id": key, "label": label} for key, label in providers.PROTOCOLS.items()],
    }


@app.post("/api/models")
def list_remote_models(payload: ModelsPayload) -> dict[str, Any]:
    provider = payload.provider.lower().strip()
    if provider not in providers.PROVIDERS:
        raise HTTPException(400, "未知的模型供应商")
    saved_provider = (_setting("provider") or "").lower().strip()
    protocol = payload.protocol
    base_url = (payload.base_url or "").strip().rstrip("/") or None
    if provider == saved_provider:
        protocol = protocol or _setting("protocol") or None
        base_url = base_url or _setting("base_url") or None
    if provider == "custom" and not base_url:
        raise HTTPException(400, "自定义供应商需要填写 Base URL")
    try:
        endpoint = providers.resolve_config(provider, protocol, base_url)
        key = payload.api_key.strip() if payload.api_key else (safe_key(provider) or "")
        models = providers.list_models(endpoint, key)
    except providers.ProviderError as error:
        if error.status in (401, 403):
            raise HTTPException(401, str(error)) from error
        message = str(error)
        if error.status is None and any(
            token in message for token in ("Base URL", "API Key", "模型", "供应商")
        ):
            raise HTTPException(400, message) from error
        raise HTTPException(502, message) from error
    return {"provider": provider, "protocol": endpoint.protocol,
            "base_url": endpoint.base_url, "models": models}


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
    endpoint, key, model = _chat_target(payload.provider)
    provider = endpoint.id
    context, retrieved = _rag_context(paper_id, message)
    summary, recent = _paper_conversation_state(paper_id, provider, key, model)
    if payload.retry:
        # 重试：当前问题已在上次落库，避免在 prompt 里重复出现
        if recent and recent[-1]["role"] == "user":
            recent = recent[:-1]
    else:
        with get_connection() as connection:
            connection.execute("INSERT INTO chat_messages(paper_id,role,content,provider,created_at) VALUES(?,?,?,?,?)",
                               (paper_id, "user", message, provider, utc_now()))
    prompt = _build_paper_prompt(summary, recent, context, message)
    # Citations must be grounded in the retrieved paper context, never in the
    # conversation itself; otherwise an old page marker could validate a new claim.
    grounding = context
    if payload.stream:
        return StreamingResponse(
            _paper_chat_stream(paper_id, provider, key, model, prompt, message, grounding, retrieved),
            media_type="application/x-ndjson",
        )
    raw = _provider_chat(provider, key, prompt, model)
    answer = _grounded_response(message, raw, grounding)
    answer, _ = apply_safety(answer, message)
    with get_connection() as connection:
        cursor = connection.execute("INSERT INTO chat_messages(paper_id,role,content,provider,created_at) VALUES(?,?,?,?,?)",
                                     (paper_id, "assistant", answer, provider, utc_now()))
    citations = []
    for page_number in sorted({int(number) for number in re.findall(r"\[Page\s+(\d+)\]", answer)}):
        citations.append({"id": page_number, "page_number": page_number, "text": f"[Page {page_number}]"})
    sources = _clean_chunks(retrieved)
    return {"id": cursor.lastrowid, "content": answer, "answer": answer, "provider": provider,
            "citations": citations, "sources": sources, "citation_level": "page",
            "warnings": ["该对话接口只校验页码，未做逐字引文与论断核验；请优先使用 Agent 问答。"]}


@app.get("/api/papers/{paper_id}/chat")
def chat_history(paper_id: int) -> list[dict[str, Any]]:
    _paper_context(paper_id)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id,role,content,provider,created_at FROM chat_messages WHERE paper_id=? ORDER BY id", (paper_id,)
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/chat")
def general_chat(payload: ChatPayload) -> dict[str, Any]:
    message = (payload.message or payload.question).strip()
    if not message:
        raise HTTPException(400, "问题不能为空")
    endpoint, key, model = _chat_target(payload.provider)
    provider = endpoint.id
    session_id = _resolve_chat_session(payload.session_id, create=True)
    summary, recent = _general_conversation_state(provider, key, model, session_id)
    if payload.retry:
        # 重试：当前问题已在上次落库，避免在 prompt 里重复出现
        if recent and recent[-1]["role"] == "user":
            recent = recent[:-1]
    else:
        with get_connection() as connection:
            connection.execute(
                "INSERT INTO general_messages(session_id,role,content,provider,created_at) "
                "VALUES(?,?,?,?,?)",
                (session_id, "user", message, provider, utc_now()))
            _touch_chat_session(connection, session_id, _session_title(message))
    prompt = _build_general_prompt(summary, recent, message)
    if payload.stream:
        return StreamingResponse(
            _general_chat_stream(provider, key, model, prompt, message, session_id),
            media_type="application/x-ndjson",
        )
    answer = _provider_chat(provider, key, prompt, model).strip()
    if not answer:
        raise HTTPException(502, "模型返回了空响应")
    answer, _ = apply_safety(answer, message)
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO general_messages(session_id,role,content,provider,created_at) "
            "VALUES(?,?,?,?,?)",
            (session_id, "assistant", answer, provider, utc_now()))
        _touch_chat_session(connection, session_id)
    return {"id": cursor.lastrowid, "content": answer, "answer": answer, "provider": provider,
            "session_id": session_id}


@app.get("/api/chat/sessions")
def general_chat_sessions() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(m.id) AS message_count "
            "FROM general_sessions s "
            "LEFT JOIN general_messages m ON m.session_id = s.id "
            "GROUP BY s.id ORDER BY s.updated_at DESC, s.id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/chat/sessions")
def create_general_chat_session() -> dict[str, Any]:
    now = utc_now()
    with get_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO general_sessions(title,created_at,updated_at) VALUES('',?,?)",
            (now, now),
        )
    return {"id": cursor.lastrowid, "title": "", "created_at": now,
            "updated_at": now, "message_count": 0}


@app.patch("/api/chat/sessions/{session_id}")
def rename_general_chat_session(session_id: int,
                                payload: SessionTitlePayload) -> dict[str, Any]:
    title = payload.title.strip()
    if not title:
        raise HTTPException(400, "会话标题不能为空")
    with get_connection() as connection:
        if connection.execute(
                "SELECT 1 FROM general_sessions WHERE id=?", (session_id,)
        ).fetchone() is None:
            raise HTTPException(404, "会话不存在")
        connection.execute(
            "UPDATE general_sessions SET title=? WHERE id=?", (title, session_id))
        row = connection.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, COUNT(m.id) AS message_count "
            "FROM general_sessions s "
            "LEFT JOIN general_messages m ON m.session_id = s.id "
            "WHERE s.id=? GROUP BY s.id",
            (session_id,),
        ).fetchone()
    return dict(row)


@app.delete("/api/chat/sessions/{session_id}")
def delete_general_chat_session(session_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        if connection.execute(
                "SELECT 1 FROM general_sessions WHERE id=?", (session_id,)
        ).fetchone() is None:
            raise HTTPException(404, "会话不存在")
        connection.execute(
            "DELETE FROM general_messages WHERE session_id=?", (session_id,))
        connection.execute(
            "DELETE FROM general_session_summaries WHERE session_id=?", (session_id,))
        connection.execute("DELETE FROM general_sessions WHERE id=?", (session_id,))
    return {"id": session_id, "status": "deleted"}


@app.get("/api/chat")
def general_chat_history(session_id: Optional[int] = None) -> list[dict[str, Any]]:
    resolved = _resolve_chat_session(session_id)
    if resolved is None:
        return []
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id,role,content,provider,created_at FROM general_messages "
            "WHERE session_id=? ORDER BY id",
            (resolved,),
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/papers/{paper_id}/retry")
def retry_paper(paper_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        paper = connection.execute(
            "SELECT stored_path, status FROM papers WHERE id=?", (paper_id,)
        ).fetchone()
    if not paper:
        raise HTTPException(404, "论文不存在")
    if paper["status"] == "processing":
        raise HTTPException(409, "论文正在处理中")
    path = Path(paper["stored_path"])
    if not path.is_file():
        raise HTTPException(410, "原始 PDF 已丢失，请重新导入")
    _enqueue_import(paper_id, path)
    return {"id": paper_id, "status": "processing"}


# --- Built frontend (single-process production mode) ---------------------------------
# The API routes above are registered first, so this SPA fallback never shadows
# them.  In dev mode Vite serves the frontend and these routes are unused.

def _dist_file(relative: str) -> Optional[Path]:
    if not DIST_DIR.is_dir():
        return None
    candidate = (DIST_DIR / relative).resolve()
    try:
        candidate.relative_to(DIST_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _index_response():
    from fastapi.responses import FileResponse

    index = _dist_file("index.html")
    if index is None:
        raise HTTPException(404, "前端尚未构建，请运行 npm run build，或用 --dev 启动 Vite")
    return FileResponse(index, media_type="text/html",
                        headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False)
def frontend_index():
    return _index_response()


@app.get("/{path:path}", include_in_schema=False)
def frontend_assets(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "接口不存在")
    file = _dist_file(path)
    if file is not None:
        cache = ("public, max-age=31536000, immutable"
                 if path.startswith("assets/") else "no-cache")
        from fastapi.responses import FileResponse
        return FileResponse(file, headers={"Cache-Control": cache})
    return _index_response()
