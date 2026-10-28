"""SQLite storage, schema migrations and local data paths."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import tokenization

APP_HOME = Path(os.environ.get("PAPER_READER_HOME", Path.home() / "PaperReader"))
PAPERS_DIR = APP_HOME / "papers"
DB_PATH = APP_HOME / "database" / "paper_reader.sqlite3"
VECTORS_DIR = APP_HOME / "vectors"

SCHEMA = """
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
    needs_ocr INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_dim INTEGER,
    progress_stage TEXT NOT NULL DEFAULT '',
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 0
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
    search_text TEXT NOT NULL DEFAULT '',
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
CREATE TABLE IF NOT EXISTS chat_summaries (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    summarized_upto INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS general_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    provider TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS general_summary (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    summary TEXT NOT NULL,
    summarized_upto INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_analysis (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    prompt_version TEXT NOT NULL,
    fields_json TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER,
    mode TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    citation_level TEXT,
    trace_json TEXT NOT NULL DEFAULT '[]',
    citations_json TEXT NOT NULL DEFAULT '[]',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    thought TEXT,
    tool TEXT,
    args_json TEXT,
    observation_json TEXT,
    duration_ms INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_run_id INTEGER,
    paper_id INTEGER,
    rating INTEGER NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path or DB_PATH), timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def ensure_column(connection: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _ensure_search_index(connection: sqlite3.Connection) -> None:
    """Create/backfill the segmented FTS index used for Chinese BM25.

    Runs after column migrations so legacy databases without ``search_text``
    are upgraded in place.  ``jieba`` is optional; the bigram fallback keeps
    the index deterministic.
    """
    connection.executescript(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_seg USING fts5("
        "search_text, content='chunks', content_rowid='id');"
        "CREATE TRIGGER IF NOT EXISTS chunks_search_ai AFTER INSERT ON chunks BEGIN "
        "INSERT INTO chunks_fts_seg(rowid, search_text) VALUES (new.id, new.search_text); "
        "END;"
        "CREATE TRIGGER IF NOT EXISTS chunks_search_ad AFTER DELETE ON chunks BEGIN "
        "INSERT INTO chunks_fts_seg(chunks_fts_seg, rowid, search_text) "
        "VALUES ('delete', old.id, old.search_text); "
        "END;"
        "CREATE TRIGGER IF NOT EXISTS chunks_search_au AFTER UPDATE OF search_text ON chunks BEGIN "
        "INSERT INTO chunks_fts_seg(chunks_fts_seg, rowid, search_text) "
        "VALUES ('delete', old.id, old.search_text); "
        "INSERT INTO chunks_fts_seg(rowid, search_text) VALUES (new.id, new.search_text); "
        "END;"
    )
    rows = connection.execute(
        "SELECT id, text, search_text FROM chunks WHERE text != ''"
    ).fetchall()
    for row in rows:
        expected = tokenization.segment(row["text"])
        if expected != (row["search_text"] or ""):
            connection.execute("UPDATE chunks SET search_text=? WHERE id=?",
                               (expected, row["id"]))
    indexed = connection.execute("SELECT COUNT(*) AS n FROM chunks_fts_seg").fetchone()["n"]
    total = connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    if indexed != total:
        connection.execute("INSERT INTO chunks_fts_seg(chunks_fts_seg) VALUES('rebuild')")


def initialize_storage(app_home: Optional[Path | str] = None,
                       db_path: Optional[Path | str] = None,
                       papers_dir: Optional[Path | str] = None,
                       vectors_dir: Optional[Path | str] = None) -> None:
    home = Path(app_home) if app_home is not None else APP_HOME
    database = Path(db_path) if db_path is not None else DB_PATH
    papers = Path(papers_dir) if papers_dir is not None else PAPERS_DIR
    vectors = Path(vectors_dir) if vectors_dir is not None else VECTORS_DIR
    for directory in (papers, database.parent, vectors, home / "cache",
                      home / "exports", home / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    with get_connection(database) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA)
        # Migrate databases created by earlier versions without making startup fragile.
        ensure_column(connection, "papers", "metadata_json", "metadata_json TEXT NOT NULL DEFAULT '{}'")
        ensure_column(connection, "papers", "vectorized", "vectorized INTEGER NOT NULL DEFAULT 0")
        ensure_column(connection, "papers", "needs_ocr", "needs_ocr INTEGER NOT NULL DEFAULT 0")
        ensure_column(connection, "papers", "embedding_model", "embedding_model TEXT NOT NULL DEFAULT ''")
        ensure_column(connection, "papers", "embedding_dim", "embedding_dim INTEGER")
        ensure_column(connection, "papers", "progress_stage", "progress_stage TEXT NOT NULL DEFAULT ''")
        ensure_column(connection, "papers", "progress_current", "progress_current INTEGER NOT NULL DEFAULT 0")
        ensure_column(connection, "papers", "progress_total", "progress_total INTEGER NOT NULL DEFAULT 0")
        ensure_column(connection, "chunks", "x0", "x0 REAL")
        ensure_column(connection, "chunks", "y0", "y0 REAL")
        ensure_column(connection, "chunks", "x1", "x1 REAL")
        ensure_column(connection, "chunks", "y1", "y1 REAL")
        ensure_column(connection, "chunks", "section", "section TEXT NOT NULL DEFAULT ''")
        ensure_column(connection, "chunks", "heading", "heading TEXT NOT NULL DEFAULT ''")
        ensure_column(connection, "chunks", "search_text", "search_text TEXT NOT NULL DEFAULT ''")
        _ensure_search_index(connection)
        connection.commit()
