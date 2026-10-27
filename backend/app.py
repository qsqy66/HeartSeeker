from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import ocr as ocr_module
from .agent import AgentConfig, AgentError, AgentRunner
from .credentials import CredentialError, key_store as _credential_key_store, safe_key
from .embedding import DEFAULT_EMBED_MODEL, delete_paper_vectors
from .ingestion import (
    is_heading as _is_heading,
    process_document,
    reindex_document,
)
from .safety import MEDICAL_SYSTEM_RULES, apply_safety, wrap_untrusted
from .storage import APP_HOME, DB_PATH, PAPERS_DIR, VECTORS_DIR, get_connection, initialize_storage, utc_now
from . import providers, retrieval
from .structured import extract_fields


MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_PAGES = 300
RECENT_TURNS = 10  # 最近 10 轮对话原文保留，更早的历史压缩为摘要
LOGGER = logging.getLogger("paper_reader.ingestion")

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
    def report(stage: str, current: int, total: int) -> None:
        try:
            _set_progress(paper_id, stage, current, total)
        except Exception:
            LOGGER.warning("failed to persist progress for paper %s", paper_id, exc_info=True)
    return report


def _ocr_options() -> dict[str, Any]:
    """Read local OCR settings without letting a bad value abort an import."""
    enabled = _setting("ocr_enabled", "true").lower() not in {"0", "false", "no", "off"}
    try:
        dpi = int(_setting("ocr_dpi", str(ocr_module.DEFAULT_DPI)))
    except ValueError:
        dpi = ocr_module.DEFAULT_DPI
    return {
        "enable_ocr": enabled,
        "ocr_engine": _setting("ocr_engine", "auto"),
        "ocr_lang": _setting("ocr_lang", ocr_module.DEFAULT_LANG),
        "ocr_dpi": max(72, min(dpi, 600)),
    }


def _run_import_job(paper_id: int, stored_path: str) -> None:
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


def _run_reindex_job(paper_id: int, key: str) -> None:
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


class AgentChatPayload(BaseModel):
    message: str = Field(default="", max_length=12000)
    question: str = Field(default="", max_length=12000)
    paper_id: Optional[int] = None
    mode: str = "auto"  # auto | paper | library | general
    provider: Optional[str] = None
    stream: bool = True
    retry: bool = False


class ModelsPayload(BaseModel):
    provider: str
    api_key: Optional[str] = None
    protocol: Optional[str] = None
    base_url: Optional[str] = None


def _key_store(provider: str, value: Optional[str] = None) -> Optional[str]:
    """Store API keys in the operating system credential manager."""
    try:
        provider = provider.lower().strip()
        if provider not in providers.PROVIDERS:
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


def _general_conversation_state(provider: str, key: str, model: str) -> tuple[str, list[dict[str, Any]]]:
    with get_connection() as connection:
        messages = [dict(row) for row in connection.execute(
            "SELECT id, role, content FROM general_messages ORDER BY id"
        ).fetchall()]
        row = connection.execute("SELECT summary, summarized_upto FROM general_summary WHERE id=1").fetchone()
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
            "INSERT INTO general_summary(id, summary, summarized_upto, updated_at) VALUES(1,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, summarized_upto=excluded.summarized_upto, updated_at=excluded.updated_at",
            (summary, last_older_id, utc_now()),
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
                      "citations": citations, "sources": _clean_chunks(retrieved)}, ensure_ascii=False) + "\n"


def _general_chat_stream(provider: str, key: str, model: str, prompt: str, question: str = ""):
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
        cursor = connection.execute("INSERT INTO general_messages(role,content,provider,created_at) VALUES(?,?,?,?)",
                                     ("assistant", answer, provider, utc_now()))
    yield json.dumps({"done": True, "id": cursor.lastrowid, "content": answer}, ensure_ascii=False) + "\n"


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


def _store_agent_message(mode: str, paper_id: Optional[int], role: str,
                         content: str, provider: str) -> None:
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
                "INSERT INTO general_messages(role,content,provider,created_at) VALUES(?,?,?,?,?)",
                (role, content, provider, utc_now()),
            )


def _agent_history(mode: str, paper_id: Optional[int], retry: bool,
                   message: str) -> list[dict[str, Any]]:
    with get_connection() as connection:
        if mode == "paper" and paper_id is not None:
            rows = connection.execute(
                "SELECT role,content FROM chat_messages WHERE paper_id=? ORDER BY id DESC LIMIT 20",
                (paper_id,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT role,content FROM general_messages ORDER BY id DESC LIMIT 20"
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

    history = _agent_history(mode, paper_id, payload.retry, message)
    agent_config = AgentConfig(provider=provider, key=key, model=model, mode=mode,
                               protocol=endpoint.protocol, base_url=endpoint.base_url,
                               paper_id=paper_id, history=history)
    runner = AgentRunner(agent_config, DB_PATH, VECTORS_DIR)

    if not payload.stream:
        try:
            done = runner.run(message)
        except AgentError as error:
            raise HTTPException(502, str(error)) from error
        if not payload.retry:
            _store_agent_message(mode, paper_id, "user", message, provider)
        _store_agent_message(mode, paper_id, "assistant", done.get("content", ""), provider)
        return done

    if not payload.retry:
        _store_agent_message(mode, paper_id, "user", message, provider)

    def generate():
        for event in runner.run_iter(message):
            if event.get("type") == "done":
                _store_agent_message(mode, paper_id, "assistant", event.get("content", ""), provider)
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


def _ocr_available(engine_name: str) -> str:
    """Return the resolved local OCR engine name, or ``"none"``."""
    try:
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
        "ocr_available": _ocr_available(values.get("ocr_engine", "auto")),
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
    for key in ("ocr_enabled", "ocr_engine", "ocr_lang", "ocr_dpi"):
        if key in payload and str(payload[key]).strip():
            values[key] = str(payload[key]).strip()
    for key, value in values.items():
        if len(key) > 100 or len(value) > 10000:
            raise HTTPException(400, "设置键或值过长")

    if api_key:
        if not provider:
            raise HTTPException(400, "请先选择模型供应商")
        _key_store(provider, api_key)
    if embedding_api_key:
        _key_store("qwen", embedding_api_key)

    with get_connection() as connection:
        for key, value in values.items():
            connection.execute(
                "INSERT INTO settings(key,value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                (key, value, utc_now()),
            )
    return _settings_payload(values)


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
            "citations": citations, "sources": sources}


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
    summary, recent = _general_conversation_state(provider, key, model)
    if payload.retry:
        # 重试：当前问题已在上次落库，避免在 prompt 里重复出现
        if recent and recent[-1]["role"] == "user":
            recent = recent[:-1]
    else:
        with get_connection() as connection:
            connection.execute("INSERT INTO general_messages(role,content,provider,created_at) VALUES(?,?,?,?)",
                               ("user", message, provider, utc_now()))
    prompt = _build_general_prompt(summary, recent, message)
    if payload.stream:
        return StreamingResponse(
            _general_chat_stream(provider, key, model, prompt, message),
            media_type="application/x-ndjson",
        )
    answer = _provider_chat(provider, key, prompt, model).strip()
    if not answer:
        raise HTTPException(502, "模型返回了空响应")
    answer, _ = apply_safety(answer, message)
    with get_connection() as connection:
        cursor = connection.execute("INSERT INTO general_messages(role,content,provider,created_at) VALUES(?,?,?,?)",
                                     ("assistant", answer, provider, utc_now()))
    return {"id": cursor.lastrowid, "content": answer, "answer": answer, "provider": provider}


@app.get("/api/chat")
def general_chat_history() -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id,role,content,provider,created_at FROM general_messages ORDER BY id"
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
