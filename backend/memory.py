"""Small, local, user-controlled long-term memory for the agent.

Memories are plain notes stored in SQLite next to the rest of the local data.
They are retrieved deterministically by term overlap; no cloud service or
embedding call is involved.  Only notes the user explicitly asks to remember
are written (the agent tool spec says so), and every note can be deleted.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

from . import tokenization
from .storage import utc_now

MAX_TEXT = 2000
MAX_TAGS = 10
SPACE_RE = re.compile(r"\s+")


class MemoryError(ValueError):
    """Raised for invalid memory operations."""


def _connect(db_path: Path | str) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE IF NOT EXISTS agent_memories ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,"
        "tags TEXT NOT NULL DEFAULT '[]', source TEXT NOT NULL DEFAULT '',"
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    return connection


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [item for item in re.split(r"[,，\s]+", tags) if item]
    if not isinstance(tags, list):
        raise MemoryError("tags 必须是字符串或字符串列表")
    cleaned: list[str] = []
    for tag in tags:
        value = str(tag).strip()
        if value and value not in cleaned:
            cleaned.append(value[:40])
    return cleaned[:MAX_TAGS]


def _row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        tags = json.loads(row["tags"])
    except (TypeError, ValueError):
        tags = []
    return {"id": row["id"], "text": row["text"], "tags": tags,
            "source": row["source"], "created_at": row["created_at"],
            "updated_at": row["updated_at"]}


def remember(db_path: Path | str, text: str, tags: Any = None,
             source: str = "user") -> dict[str, Any]:
    value = (text or "").strip()
    if not value:
        raise MemoryError("记忆内容不能为空")
    if len(value) > MAX_TEXT:
        raise MemoryError(f"记忆内容不能超过 {MAX_TEXT} 字")
    tag_values = _normalize_tags(tags)
    now = utc_now()
    connection = _connect(db_path)
    try:
        cursor = connection.execute(
            "INSERT INTO agent_memories(text,tags,source,created_at,updated_at) "
            "VALUES(?,?,?,?,?)",
            (value, json.dumps(tag_values, ensure_ascii=False), source[:80], now, now),
        )
        connection.commit()
        memory_id = int(cursor.lastrowid)
    finally:
        connection.close()
    return {"id": memory_id, "text": value, "tags": tag_values,
            "source": source, "created_at": now, "updated_at": now}


def list_memories(db_path: Path | str, limit: int = 50) -> list[dict[str, Any]]:
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM agent_memories ORDER BY updated_at DESC, id DESC LIMIT ?",
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    finally:
        connection.close()
    return [_row(row) for row in rows]


def forget(db_path: Path | str, memory_id: int) -> bool:
    connection = _connect(db_path)
    try:
        cursor = connection.execute("DELETE FROM agent_memories WHERE id=?",
                                    (int(memory_id),))
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


def _score(query_terms: set[str], text: str, tags: list[str], query: str) -> float:
    if not query_terms:
        return 0.0
    text_terms = set(tokenization.terms(text))
    for tag in tags:
        text_terms.update(tokenization.terms(tag))
    overlap = len(query_terms & text_terms) / len(query_terms)
    if SPACE_RE.sub("", query).lower() in SPACE_RE.sub("", text).lower():
        overlap += 0.5
    return overlap


def recall(db_path: Path | str, query: str = "", limit: int = 5) -> list[dict[str, Any]]:
    query = (query or "").strip()
    limit = max(1, min(int(limit), 20))
    memories = list_memories(db_path, limit=200)
    if not query:
        return memories[:limit]
    query_terms = set(tokenization.terms(query))
    scored: list[tuple[float, dict[str, Any]]] = []
    for memory in memories:
        score = _score(query_terms, memory["text"], memory["tags"], query)
        if score > 0:
            scored.append((score, memory))
    scored.sort(key=lambda item: item[1]["updated_at"], reverse=True)
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{**memory, "score": round(score, 3)}
            for score, memory in scored[:limit]]


def get(db_path: Path | str, memory_id: int) -> Optional[dict[str, Any]]:
    connection = _connect(db_path)
    try:
        row = connection.execute("SELECT * FROM agent_memories WHERE id=?",
                                 (int(memory_id),)).fetchone()
    finally:
        connection.close()
    return _row(row) if row else None
