"""Pluggable reranking for hybrid retrieval.

Two engines ship today:

* ``lexical``     – deterministic token-coverage reranker, no model required
* ``cross-encoder`` – optional local sentence-transformers cross encoder

Third-party engines can be registered with :func:`register_reranker`; the
retrieval pipeline only depends on ``rerank(query, rows, top_k)``.  A missing
model or an engine error always degrades to the lexical reranker rather than
failing a search.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

from . import tokenization


class RerankError(RuntimeError):
    """Raised when a configured reranker cannot run."""


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        ...


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get(key) or "") for key in
                    ("heading", "section", "text", "paper_title"))


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


_FACTORIES: dict[str, Callable[[str], Optional[Reranker]]] = {
    "lexical": lambda model: LexicalReranker(),
    "none": lambda model: None,
    "cross-encoder": lambda model: CrossEncoderReranker(model),
}


def register_reranker(name: str, factory: Callable[[str], Optional[Reranker]]) -> None:
    """Register a third-party reranker; factory receives the model name."""
    _FACTORIES[name.strip().lower()] = factory


def get_reranker(engine: Optional[str] = None, model: str = "") -> Optional[Reranker]:
    name = (engine or "auto").strip().lower()
    if name == "none":
        return None
    if name == "auto":
        if model.strip():
            candidate = CrossEncoderReranker(model)
            if candidate.available():
                return candidate
        return LexicalReranker()
    factory = _FACTORIES.get(name)
    if factory is None:
        raise RerankError(f"未知重排引擎：{name}")
    return factory(model)


def rerank(query: str, rows: list[dict[str, Any]], top_k: int,
           engine: Optional[str] = None, model: str = "") -> list[dict[str, Any]]:
    """Rerank with graceful fallback to the lexical engine."""
    if not rows:
        return []
    try:
        reranker = get_reranker(engine, model)
    except RerankError:
        reranker = LexicalReranker()
    if reranker is None:
        return rows[:top_k]
    try:
        return reranker.rerank(query, rows, top_k)
    except RerankError:
        return LexicalReranker().rerank(query, rows, top_k)
