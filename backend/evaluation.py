"""Offline/live evaluation harness for retrieval and citation grounding.

The harness is intentionally LLM-free by default: retrieval recall and quote
verification are deterministic and cheap enough to run in CI.  A golden set can
be pointed at the bundled fixture (``eval/corpus.json``) or at a real local
library via ``--db``.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Optional

from . import citations, retrieval, tokenization
from .storage import get_connection, initialize_storage, utc_now
from .structured import UNREPORTED

MAX_TOP_K = 50


def _load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cases(path: Path | str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("评测集格式错误：需要 cases 列表")
    return [case for case in cases if isinstance(case, dict) and case.get("question")]


def build_fixture(corpus_path: Path | str, app_home: Path | str) -> tuple[Path, Path]:
    """Create a self-contained SQLite/FTS fixture from a JSON corpus."""
    corpus = _load_json(corpus_path)
    papers = corpus.get("papers", []) if isinstance(corpus, dict) else []
    home = Path(app_home)
    db_path = home / "database" / "heartseeker_med.sqlite3"
    vectors_dir = home / "vectors"
    initialize_storage(app_home=home, db_path=db_path,
                       papers_dir=home / "papers", vectors_dir=vectors_dir)
    with get_connection(db_path) as connection:
        for paper in papers:
            title = str(paper.get("title") or "未命名论文")
            digest = hashlib.sha256(f"{paper.get('id')}:{title}".encode("utf-8")).hexdigest()
            cursor = connection.execute(
                "INSERT INTO papers(title,file_name,stored_path,sha256,page_count,status,imported_at,"
                "vectorized,needs_ocr) VALUES(?,?,?,?,?,?,?,0,0)",
                (title, paper.get("file_name") or "fixture.pdf", "",
                 digest, len(paper.get("pages") or []), "ready", utc_now()),
            )
            paper_id = int(cursor.lastrowid)
            page_ids: dict[int, int] = {}
            for page in paper.get("pages") or []:
                page_number = int(page.get("page_number") or 0)
                page_ids[page_number] = int(connection.execute(
                    "INSERT INTO pages(paper_id,page_number,text) VALUES(?,?,?)",
                    (paper_id, page_number, str(page.get("text") or "")),
                ).lastrowid)
            for index, chunk in enumerate(paper.get("chunks") or []):
                page_number = int(chunk.get("page_number") or 0)
                page_id = page_ids.get(page_number)
                if page_id is None:
                    continue
                chunk_text = str(chunk.get("text") or "")
                connection.execute(
                    "INSERT INTO chunks(paper_id,page_id,chunk_index,text,search_text,section,heading) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (paper_id, page_id, index, chunk_text,
                     tokenization.segment(chunk_text),
                     str(chunk.get("section") or ""), str(chunk.get("heading") or "")),
                )
        connection.commit()
    return db_path, vectors_dir


def _resolve_paper_id(case: dict[str, Any], db_path: Path | str) -> Optional[int]:
    if case.get("paper_id") is not None:
        try:
            return int(case["paper_id"])
        except (TypeError, ValueError):
            return None
    title = str(case.get("paper_title") or "").strip()
    if not title:
        return None
    connection = retrieval.connect(db_path)
    try:
        row = connection.execute(
            "SELECT id FROM papers WHERE title LIKE ? ORDER BY id LIMIT 1", (f"%{title}%",)
        ).fetchone()
    finally:
        connection.close()
    return int(row["id"]) if row else None


def _rank(rows: list[dict[str, Any]], case: dict[str, Any],
          paper_id: Optional[int]) -> Optional[int]:
    expected = {int(page) for page in case.get("expected_pages") or []}
    if not expected:
        return None
    for rank, row in enumerate(rows, start=1):
        if int(row.get("page_number") or 0) not in expected:
            continue
        if paper_id is not None and int(row.get("paper_id") or 0) != paper_id:
            continue
        return rank
    return None


def _page_text(db_path: Path | str, paper_id: int, page: int) -> str:
    return retrieval.read_page(db_path, paper_id, page)["text"]


def _citation_metrics(cases: list[dict[str, Any]], db_path: Path | str,
                      resolved: dict[int, Optional[int]]) -> dict[str, Any]:
    checked = 0
    positive_pass = 0
    negative_reject = 0
    failures: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        quote = str(case.get("quote") or "").strip()
        if not quote:
            continue
        paper_id = resolved.get(index)
        if paper_id is None:
            failures.append({"case": case.get("id"), "metric": "citation", "reason": "论文未解析"})
            continue
        try:
            page = int(case.get("quote_page") or (case.get("expected_pages") or [0])[0])
            page_text = _page_text(db_path, paper_id, page)
        except (ValueError, LookupError, TypeError) as error:
            failures.append({"case": case.get("id"), "metric": "citation", "reason": str(error)})
            continue
        checked += 1
        evidence = [{"paper_id": paper_id, "page_number": page, "text": page_text}]
        citation = {"paper_id": paper_id, "page": page, "quote": quote}
        positive = citations.validate_citations(
            f"结论见原文 [Page {page}]", [citation], evidence,
            default_paper_id=paper_id, require_citation=True,
        )
        if positive.ok and positive.level == "quote":
            positive_pass += 1
        else:
            failures.append({
                "case": case.get("id"), "metric": "citation_positive",
                "reason": "; ".join(positive.warnings) or "引文未通过核验",
            })

        fabricated = "本句为评测虚构内容，原文中不应出现 ZZQX"
        if citations.normalize_quote(fabricated) in citations.normalize_quote(page_text):
            fabricated = fabricated + " sentinel"
        negative = citations.validate_citations(
            f"结论见原文 [Page {page}]",
            [{"paper_id": paper_id, "page": page, "quote": fabricated}],
            evidence, default_paper_id=paper_id, require_citation=True,
        )
        if not negative.ok:
            negative_reject += 1
        else:
            failures.append({
                "case": case.get("id"), "metric": "citation_negative",
                "reason": "虚构引文被错误接受",
            })
    return {
        "checked": checked,
        "positive_pass_rate": round(positive_pass / checked, 4) if checked else None,
        "negative_reject_rate": round(negative_reject / checked, 4) if checked else None,
        "failures": failures,
    }


def _structured_metrics(db_path: Path | str) -> Optional[dict[str, Any]]:
    try:
        connection = get_connection(db_path)
        try:
            rows = connection.execute("SELECT fields_json FROM paper_analysis").fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return None
    total = 0
    verified = 0
    for row in rows:
        try:
            fields = json.loads(row["fields_json"])
        except (TypeError, ValueError):
            continue
        if not isinstance(fields, dict):
            continue
        for field in fields.values():
            if not isinstance(field, dict):
                continue
            value = str(field.get("value") or "")
            if not value or value == UNREPORTED:
                continue
            total += 1
            if field.get("evidence"):
                verified += 1
    return {
        "cached_papers": len(rows),
        "fields_with_evidence": verified,
        "non_empty_fields": total,
        "evidence_rate": round(verified / total, 4) if total else None,
    }


def evaluate(dataset_path: Path | str, db_path: Path | str, vectors_dir: Path | str,
             top_k: int = 8, enable_vectors: bool = False,
             api_key: Optional[str] = None, mode: str = "live") -> dict[str, Any]:
    cases = load_cases(dataset_path)
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    resolved = {index: _resolve_paper_id(case, db_path) for index, case in enumerate(cases)}

    hits = 0
    keyword_hits = 0
    reciprocal_rank = 0.0
    failures: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        paper_id = resolved.get(index)
        scope = str(case.get("scope") or "paper").lower()
        search_paper_id = paper_id if scope != "library" else None
        if paper_id is None:
            failures.append({"case": case.get("id"), "metric": "retrieval", "reason": "论文未解析"})
            details.append({"case": case.get("id"), "rank": None, "keyword_rank": None})
            continue
        rows = retrieval.hybrid_search(
            case["question"], db_path, vectors_dir, paper_id=search_paper_id,
            top_k=top_k, enable_vectors=enable_vectors, api_key=api_key,
        )
        rank = _rank(rows, case, paper_id)
        keyword_rank = _rank(
            retrieval.keyword_search(case["question"], db_path, search_paper_id, limit=top_k),
            case, paper_id,
        )
        if rank is not None:
            hits += 1
            reciprocal_rank += 1.0 / rank
        else:
            failures.append({
                "case": case.get("id"), "metric": "retrieval",
                "reason": f"前 {top_k} 条未命中预期页码 {case.get('expected_pages')}",
                "top_pages": [row.get("page_number") for row in rows],
            })
        if keyword_rank is not None:
            keyword_hits += 1
        details.append({
            "case": case.get("id"),
            "question": case["question"],
            "scope": scope,
            "expected_pages": case.get("expected_pages"),
            "rank": rank,
            "keyword_rank": keyword_rank,
            "top_pages": [row.get("page_number") for row in rows],
            "top_paper_ids": [row.get("paper_id") for row in rows],
        })

    total = len(cases)
    citation_metrics = _citation_metrics(cases, db_path, resolved)
    failures.extend(citation_metrics.pop("failures"))
    return {
        "mode": mode,
        "dataset": str(dataset_path),
        "top_k": top_k,
        "vector_search": bool(enable_vectors),
        "cases": total,
        "retrieval": {
            "recall": round(hits / total, 4) if total else None,
            "mrr": round(reciprocal_rank / total, 4) if total else None,
            "keyword_recall": round(keyword_hits / total, 4) if total else None,
        },
        "citations": citation_metrics,
        "structured": _structured_metrics(db_path),
        "failures": failures,
        "details": details,
    }


def evaluate_offline(corpus_path: Path | str, dataset_path: Path | str,
                     top_k: int = 8) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="heartseeker-med-eval-") as temp:
        db_path, vectors_dir = build_fixture(corpus_path, temp)
        return evaluate(dataset_path, db_path, vectors_dir, top_k=top_k,
                        enable_vectors=False, mode="offline")


def evaluate_live(dataset_path: Path | str, db_path: Path | str,
                  vectors_dir: Path | str, top_k: int = 8,
                  enable_vectors: bool = True,
                  api_key: Optional[str] = None) -> dict[str, Any]:
    return evaluate(dataset_path, db_path, vectors_dir, top_k=top_k,
                    enable_vectors=enable_vectors, api_key=api_key, mode="live")
