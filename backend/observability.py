"""In-process structured telemetry for the local dashboard.

觅心 runs as a single-user local process, so a bounded in-memory ring buffer
is the right trade-off: no log files to rotate and no external service.  The
UI polls incrementally by ``seq``; token usage is also persisted to SQLite so
statistics survive restarts.
"""
from __future__ import annotations

import collections
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

MAX_EVENTS = 1000
LOGGER_NAME = "heartseeker_med"

_events: collections.deque[dict[str, Any]] = collections.deque(maxlen=MAX_EVENTS)
_lock = threading.Lock()
_sequence = 0
_handler: Optional[logging.Handler] = None


class _StructuredHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            structured = getattr(record, "structured", None)
            fields = structured if isinstance(structured, dict) else {}
            publish({
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                **fields,
            })
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def publish(event: dict[str, Any]) -> dict[str, Any]:
    global _sequence
    with _lock:
        _sequence += 1
        item = {"seq": _sequence, "time": datetime.now(timezone.utc).isoformat(), **event}
        _events.append(item)
        return item


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    logging.getLogger(LOGGER_NAME).log(
        level, event, extra={"structured": {"event": event, **fields}})


def setup_logging() -> None:
    """Attach the ring-buffer handler once; leave console logging untouched."""
    global _handler
    logger = logging.getLogger(LOGGER_NAME)
    if _handler is None:
        _handler = _StructuredHandler()
        _handler.setLevel(logging.INFO)
        logger.addHandler(_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def recent(after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
    """Return events newer than ``after``, keeping at most ``limit`` recent ones."""
    try:
        after = int(after)
    except (TypeError, ValueError):
        after = 0
    limit = max(1, min(int(limit), MAX_EVENTS))
    with _lock:
        events = [event for event in _events if event["seq"] > after]
    return events[-limit:]


def record_usage(usage: dict[str, Any], db_path: Any = None) -> None:
    """Persist one model call and publish it to the live log.

    Called from :mod:`backend.providers`; failures here must never break a
    model call, so everything is best-effort.  ``db_path`` is bound by the API
    layer so test runs and fixtures never write into the live database.
    """
    prompt = _non_negative(usage.get("prompt_tokens"))
    completion = _non_negative(usage.get("completion_tokens"))
    total = _non_negative(usage.get("total_tokens")) or prompt + completion
    fields = {
        "provider": str(usage.get("provider") or ""),
        "model": str(usage.get("model") or ""),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "estimated": bool(usage.get("estimated")),
        "kind": str(usage.get("kind") or "chat"),
    }
    try:
        from .storage import get_connection, utc_now
        with get_connection(db_path) as connection:
            connection.execute(
                "INSERT INTO llm_usage(provider,model,prompt_tokens,completion_tokens,"
                "total_tokens,estimated,kind,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (fields["provider"], fields["model"], prompt, completion, total,
                 1 if fields["estimated"] else 0, fields["kind"], utc_now()),
            )
    except Exception:  # pragma: no cover - telemetry is best-effort
        logging.getLogger(LOGGER_NAME).debug("failed to persist token usage", exc_info=True)
    log_event("llm.usage", **fields)


def _non_negative(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
