"""Claim-level support checks for citations.

A verbatim quote proves the text exists, not that it supports the claim the
model attached to it.  This module adds deterministic checks that run fully
offline:

* numeric consistency – every number in the claim must occur in the quote
  (``20%`` may match ``0.20``; thousands separators are ignored)
* negation consistency – a negated claim must not be backed by an affirmative
  quote, and vice versa
* lexical coverage – same-language claims should share key terms with the quote

An optional NLI/cross-encoder judge can be registered for semantic entailment
(``register_nli``).  Without one, only contradictions (numbers/negation) are
treated as hard failures; low coverage is reported as a warning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from . import tokenization

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?%?")
CJK_RE = re.compile(r"[㐀-鿿]")
NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|neither|nor|failed\s+to|did\s+not|didn't|"
    r"does\s+not|doesn't|was\s+not|were\s+not|no\s+significant)\b"
    r"|未|无|不|没有|并非|未能|不能|未见",
    re.IGNORECASE,
)
STOPWORDS = {
    "the", "a", "an", "of", "was", "were", "is", "are", "and", "or", "to", "in",
    "on", "for", "with", "that", "this", "it", "as", "at", "by", "from", "be",
    "显著", "研究", "结果", "患者", "论文", "显示", "提示", "可以", "以及", "进行",
}


@dataclass
class ClaimSupport:
    supported: bool = True
    score: float = 1.0
    contradiction: bool = False
    reasons: list[str] = field(default_factory=list)
    coverage: Optional[float] = None
    same_language: bool = True
    judge: str = "deterministic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "score": round(self.score, 4),
            "contradiction": self.contradiction,
            "reasons": self.reasons,
            "coverage": None if self.coverage is None else round(self.coverage, 4),
            "judge": self.judge,
        }


@runtime_checkable
class NliJudge(Protocol):
    name: str

    def judge(self, claim: str, quote: str) -> float:
        """Return an entailment probability in [0, 1]."""


class CrossEncoderNliJudge:
    """Optional local NLI model, lazily loaded from sentence-transformers."""

    name = "cross-encoder-nli"

    def __init__(self, model: str) -> None:
        if not model.strip():
            raise ValueError("未配置 NLI 模型名")
        self.model_name = model.strip()
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder  # type: ignore
        self._model = CrossEncoder(self.model_name)
        return self._model

    def judge(self, claim: str, quote: str) -> float:
        model = self._load()
        scores = model.predict([(quote, claim)])
        value = float(scores[0])
        if value > 1.0:  # some checkpoints return logits
            value = 1.0 / (1.0 + pow(2.718281828, -value))
        return max(0.0, min(1.0, value))


_JUDGES: dict[str, Callable[[str], Optional[NliJudge]]] = {
    "none": lambda model: None,
    "cross-encoder-nli": lambda model: CrossEncoderNliJudge(model),
}


def register_nli(name: str, factory: Callable[[str], Optional[NliJudge]]) -> None:
    _JUDGES[name.strip().lower()] = factory


def get_nli(engine: Optional[str] = None, model: str = "") -> Optional[NliJudge]:
    name = (engine or "none").strip().lower()
    if name in {"", "none"}:
        return None
    factory = _JUDGES.get(name)
    if factory is None:
        raise ValueError(f"未知 NLI 引擎：{name}")
    return factory(model)


def _number_value(raw: str) -> Optional[str]:
    raw = raw.replace(",", "")
    try:
        if raw.endswith("%"):
            return f"{float(raw[:-1]):.6g}"
        return f"{float(raw):.6g}"
    except ValueError:
        return None


def _number_alternatives(raw: str) -> set[str]:
    """Return equivalent spellings, e.g. ``20%`` -> {20, 0.2}."""
    values: set[str] = set()
    raw = raw.replace(",", "")
    if raw.endswith("%"):
        try:
            percent = float(raw[:-1])
        except ValueError:
            return values
        values.add(f"{percent:.6g}")
        values.add(f"{percent / 100:.6g}")
    else:
        value = _number_value(raw)
        if value:
            values.add(value)
    return values


def _number_values(text: str) -> set[str]:
    values: set[str] = set()
    for raw in NUMBER_RE.findall(text or ""):
        values.update(_number_alternatives(raw))
    return values


def _numbers_supported(claim: str, quote: str) -> list[str]:
    groups: list[set[str]] = []
    for raw in NUMBER_RE.findall(claim or ""):
        alternatives = _number_alternatives(raw)
        if alternatives:
            groups.append(alternatives)
    if not groups:
        return []
    quote_values = _number_values(quote)
    missing = ["/".join(sorted(group)) for group in groups if not (group & quote_values)]
    if not missing:
        return []
    return [f"论断中的数值未出现在引文中：{', '.join(missing[:5])}"]


def _has_negation(text: str) -> bool:
    return bool(NEGATION_RE.search(text or ""))


def _coverage(claim: str, quote: str) -> float:
    claim_terms = {term for term in tokenization.terms(claim) if term not in STOPWORDS}
    if not claim_terms:
        return 1.0
    quote_terms = tokenization.terms(quote)
    return len(claim_terms & quote_terms) / len(claim_terms)


def verify_claim(claim: str, quote: str, judge: Optional[NliJudge] = None,
                 coverage_threshold: float = 0.35) -> ClaimSupport:
    claim = (claim or "").strip()
    quote = (quote or "").strip()
    if not claim:
        return ClaimSupport(supported=True, score=1.0, reasons=["没有论断文本"])
    if not quote:
        return ClaimSupport(supported=False, score=0.0, contradiction=True,
                            reasons=["没有可核对的引文"])

    same_language = bool(CJK_RE.search(claim)) == bool(CJK_RE.search(quote))
    reasons: list[str] = []
    contradiction = False
    score = 0.6

    numeric_reasons = _numbers_supported(claim, quote)
    if numeric_reasons:
        reasons.extend(numeric_reasons)
        contradiction = True
        score = 0.1

    if _has_negation(claim) != _has_negation(quote):
        reasons.append("论断与引文的否定/肯定方向不一致")
        contradiction = True
        score = min(score, 0.15)

    coverage = _coverage(claim, quote) if same_language else None
    if coverage is not None:
        if coverage >= 0.6:
            score = max(score, 0.85)
        elif coverage >= coverage_threshold:
            score = max(score, 0.65)
        else:
            reasons.append(f"论断与引文的关键词重合度偏低（{coverage:.2f}）")
            score = min(score, 0.45)

    supported = not contradiction
    if judge is not None:
        try:
            entailment = judge.judge(claim, quote)
        except Exception as error:  # local model failures must not block validation
            reasons.append(f"NLI 校验不可用：{error}")
            entailment = None
        if entailment is not None:
            score = round(0.5 * score + 0.5 * entailment, 4)
            if entailment < 0.5:
                supported = False
                reasons.append(f"NLI 判定蕴含概率偏低（{entailment:.2f}）")

    return ClaimSupport(
        supported=supported,
        score=max(0.0, min(1.0, score)),
        contradiction=contradiction,
        reasons=reasons,
        coverage=coverage,
        same_language=same_language,
        judge=judge.name if judge is not None else "deterministic",
    )
