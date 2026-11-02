"""Pluggable reranking for hybrid retrieval.

Three engines ship today:

* ``lexical``     – deterministic token-coverage reranker, no model required
* ``cross-encoder`` – optional local sentence-transformers cross encoder
* ``qwen``        – remote Qwen text-rerank service (DashScope-compatible API)

Third-party engines can be registered with :func:`register_reranker`; the
retrieval pipeline only depends on ``rerank(query, rows, top_k)``.  A missing
model or an engine error always degrades to the lexical reranker rather than
failing a search.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from . import tokenization

DEFAULT_QWEN_RERANK_URL = (
    "https://cn-beijing.maas.qianwenaiapi.com/api/v1/services/rerank/"
    "text-rerank/text-rerank"
)
DEFAULT_QWEN_RERANK_MODEL = "qwen3.7-text-rerank"
QWEN_RERANK_KEY = "qwen-rerank"


class RerankError(RuntimeError):
    """Raised when a configured reranker cannot run."""


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        ...


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in
                    ("heading", "section", "text", "paper_title")).strip()


class LexicalReranker:
    """Token-coverage reranker; fast, local and deterministic."""

    name = "lexical"

    def __init__(self, min_coverage: float = 0.0) -> None:
        self.min_coverage = min_coverage

    def score(self, query: str, row: dict[str, Any]) -> float:
        query_terms = tokenization.terms(query)
        if not query_terms:
            return 0.0
        text_terms = tokenization.terms(_row_text(row))
        coverage = len(query_terms & text_terms) / len(query_terms)
        heading_terms = tokenization.terms(
            f"{row.get('heading', '')} {row.get('section', '')}")
        heading_hit = 1.0 if query_terms & heading_terms else 0.0
        return 0.7 * coverage + 0.3 * heading_hit

    def rerank(self, query: str, rows: list[dict[str, Any]],
               top_k: int) -> list[dict[str, Any]]:
        if not rows:
            return []
        base_max = max(float(row.get("score") or 0.0) for row in rows) or 1.0
        scored: list[tuple[int, dict[str, Any]]] = []
        for order, row in enumerate(rows):
            lexical = self.score(query, row)
            base = float(row.get("score") or 0.0) / base_max
            updated = dict(row)
            updated["rerank_score"] = round(0.65 * base + 0.35 * lexical, 6)
            scored.append((order, updated))
        scored.sort(key=lambda item: (-item[1]["rerank_score"], item[0]))
        return [row for _, row in scored[:top_k]]


class CrossEncoderReranker:
    """Optional local cross-encoder (sentence-transformers) reranker."""

    name = "cross-encoder"

    def __init__(self, model: str) -> None:
        if not model.strip():
            raise RerankError("未配置 cross-encoder 模型名")
        self.model_name = model.strip()
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as error:
            raise RerankError("未安装 sentence-transformers") from error
        try:
            self._model = CrossEncoder(self.model_name)
        except Exception as error:
            raise RerankError(f"加载重排模型失败：{error}") from error
        return self._model

    def available(self) -> bool:
        try:
            self._load()
        except RerankError:
            return False
        return True

    def rerank(self, query: str, rows: list[dict[str, Any]],
               top_k: int) -> list[dict[str, Any]]:
        if not rows:
            return []
        model = self._load()
        pairs = [(query, _row_text(row)) for row in rows]
        try:
            scores = model.predict(pairs)
        except Exception as error:
            raise RerankError(f"重排推理失败：{error}") from error
        scored: list[tuple[int, dict[str, Any]]] = []
        for order, (row, score) in enumerate(zip(rows, scores)):
            updated = dict(row)
            updated["rerank_score"] = round(float(score), 6)
            scored.append((order, updated))
        scored.sort(key=lambda item: (-item[1]["rerank_score"], item[0]))
        return [row for _, row in scored[:top_k]]


class QwenReranker:
    """Remote Qwen text-rerank service (DashScope-compatible HTTP API).

    The endpoint and model are configurable from the settings page.  An
    explicit ``api_key`` wins; otherwise the dedicated rerank key, the Qwen
    embedding key, or ``DASHSCOPE_API_KEY`` is used in that order.
    """

    name = "qwen"

    def __init__(self, model: str = "", base_url: str = "",
                 api_key: str = "", timeout: float = 20.0) -> None:
        self.model_name = (model or "").strip() or DEFAULT_QWEN_RERANK_MODEL
        self.base_url = (
            (base_url or "").strip()
            or os.environ.get("HEARTSEEKER_MED_RERANK_URL", "").strip()
            or DEFAULT_QWEN_RERANK_URL
        )
        self.api_key = (api_key or "").strip()
        self.timeout = timeout

    def _resolve_key(self) -> str:
        if self.api_key:
            return self.api_key
        from .credentials import safe_key
        return (safe_key(QWEN_RERANK_KEY) or safe_key("qwen")
                or os.environ.get("DASHSCOPE_API_KEY", "").strip())

    @staticmethod
    def _parse_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
        output = payload.get("output")
        if isinstance(output, dict) and isinstance(output.get("results"), list):
            return output["results"]
        for key in ("results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        raise RerankError("重排服务返回格式无法识别")

    def rerank(self, query: str, rows: list[dict[str, Any]],
               top_k: int) -> list[dict[str, Any]]:
        if not rows:
            return []
        key = self._resolve_key()
        if not key:
            raise RerankError("未配置重排 API Key")
        documents = [_row_text(row) for row in rows]
        body = json.dumps({
            "model": self.model_name,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": len(documents), "return_documents": False},
        }, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise RerankError("重排 API Key 无效或没有访问权限") from error
            raise RerankError(f"重排服务返回 HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RerankError("重排服务暂不可用，请检查网络或 Base URL") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RerankError("重排服务返回了无法解析的响应") from error
        if not isinstance(payload, dict):
            raise RerankError("重排服务返回格式无法识别")

        scored: list[tuple[int, dict[str, Any]]] = []
        seen: set[int] = set()
        for item in self._parse_results(payload):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if index < 0 or index >= len(rows) or index in seen:
                continue
            raw_score = item.get("relevance_score", item.get("score", 0.0))
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            updated = dict(rows[index])
            updated["rerank_score"] = round(score, 6)
            seen.add(index)
            scored.append((index, updated))
        if not scored:
            raise RerankError("重排服务没有返回可用结果")
        # Keep rows the service omitted, after scored rows, in their original order.
        scored.extend((index, dict(row)) for index, row in enumerate(rows)
                      if index not in seen)
        scored.sort(key=lambda item: (-float(item[1].get("rerank_score", 0.0)), item[0]))
        return [row for _, row in scored[:top_k]]

    def available(self) -> bool:
        return bool(self._resolve_key())


_FACTORIES: dict[str, Callable[[str], Optional[Reranker]]] = {
    "lexical": lambda model: LexicalReranker(),
    "none": lambda model: None,
    "cross-encoder": lambda model: CrossEncoderReranker(model),
}


def register_reranker(name: str, factory: Callable[[str], Optional[Reranker]]) -> None:
    """Register a third-party reranker; factory receives the model name."""
    _FACTORIES[name.strip().lower()] = factory


def get_reranker(engine: Optional[str] = None, model: str = "",
                 base_url: str = "", api_key: str = "") -> Optional[Reranker]:
    name = (engine or "auto").strip().lower()
    if name == "none":
        return None
    if name in {"qwen", "qwen-rerank", "dashscope"}:
        return QwenReranker(model, base_url=base_url, api_key=api_key)
    if name == "auto":
        if model.strip():
            if model.strip().lower().startswith("qwen") or "rerank" in model.lower():
                return QwenReranker(model, base_url=base_url, api_key=api_key)
            candidate = CrossEncoderReranker(model)
            if candidate.available():
                return candidate
        return LexicalReranker()
    factory = _FACTORIES.get(name)
    if factory is None:
        raise RerankError(f"未知重排引擎：{name}")
    return factory(model)


def rerank(query: str, rows: list[dict[str, Any]], top_k: int,
           engine: Optional[str] = None, model: str = "",
           base_url: str = "", api_key: str = "") -> list[dict[str, Any]]:
    """Rerank with graceful fallback to the lexical engine."""
    if not rows:
        return []
    try:
        reranker = get_reranker(engine, model, base_url=base_url, api_key=api_key)
    except RerankError:
        reranker = LexicalReranker()
    if reranker is None:
        return rows[:top_k]
    try:
        return reranker.rerank(query, rows, top_k)
    except RerankError:
        return LexicalReranker().rerank(query, rows, top_k)
