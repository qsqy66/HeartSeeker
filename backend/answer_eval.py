"""Answer-level evaluation for the medical RAG agent.

Retrieval evaluation (``backend.evaluation``) answers "was the right page
retrieved?".  This module scores the final answer:

* citation precision – share of returned citations that verified
* citation recall – share of expected pages that were cited
* claim support rate – share of quoted claims supported by their quote
* unsupported claim rate – share of checked claims flagged as unsupported
* fact coverage – share of expected facts present in the answer
* refusal accuracy – did the agent refuse exactly the unanswerable questions?

The scorer is deterministic and works on saved agent output, so the same
harness can compare cloud models, local models, prompt versions and retrieval
settings without spending tokens during scoring.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

REFUSAL_PATTERNS = [
    r"证据不足", r"没有(?:足够的?|相关的?)?证据", r"现有证据不足",
    r"无法(?:回答|确定|判断|核实)", r"不能(?:回答|确定|判断)",
    r"insufficient\s+evidence", r"not\s+enough\s+evidence",
    r"unable\s+to\s+answer", r"cannot\s+(?:determine|answer|verify)",
    r"no\s+(?:sufficient\s+)?evidence",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").lower()
    return SPACE_RE.sub("", value)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer or ""))


def load_answer_cases(path: Path | str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("答案评测集格式错误：需要 cases 列表")
    return [case for case in cases if isinstance(case, dict) and case.get("question")]


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    return round(sum(usable) / len(usable), 4) if usable else None


def _fact_present(fact: Any, answer: str) -> bool:
    alternatives = fact if isinstance(fact, list) else [fact]
    normalized_answer = _norm(answer)
    for alternative in alternatives:
        if isinstance(alternative, str) and _norm(alternative) and _norm(alternative) in normalized_answer:
            return True
    return False


def _citation_matches_case(citation: dict[str, Any], case: dict[str, Any]) -> bool:
    expected_paper = case.get("paper_id")
    if expected_paper is not None and citation.get("paper_id") is not None:
        try:
            if int(citation["paper_id"]) != int(expected_paper):
                return False
        except (TypeError, ValueError):
            return False
    return True


def score_answer(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Score one agent result against one answer-evaluation case."""
    answer = str(result.get("content") or result.get("answer") or "")
    citations = [item for item in (result.get("citations") or []) if isinstance(item, dict)]
    issues: list[dict[str, str]] = []

    if result.get("error"):
        issues.append({"metric": "run", "reason": str(result["error"])})

    verified = [item for item in citations if item.get("verified")]
    citation_precision = len(verified) / len(citations) if citations else None
    if citations and citation_precision is not None and citation_precision < 1.0:
        issues.append({"metric": "citation_precision",
                       "reason": f"{len(citations) - len(verified)} 条引用未通过核验"})

    expected_pages = set()
    for page in case.get("expected_pages") or []:
        try:
            expected_pages.add(int(page))
        except (TypeError, ValueError):
            continue
    cited_pages = set()
    for citation in verified:
        if not _citation_matches_case(citation, case):
            continue
        try:
            cited_pages.add(int(citation.get("page")))
        except (TypeError, ValueError):
            continue
    citation_recall = (len(expected_pages & cited_pages) / len(expected_pages)
                       if expected_pages else None)
    if expected_pages and citation_recall is not None and citation_recall < 1.0:
        issues.append({"metric": "citation_recall",
                       "reason": "未引用预期页码 " + ",".join(
                           str(page) for page in sorted(expected_pages - cited_pages))})

    claim_checked = [item for item in citations
                     if item.get("claim_supported") is not None]
    supported = [item for item in claim_checked if item.get("claim_supported") is True]
    unsupported = [item for item in claim_checked if item.get("claim_supported") is False]
    claim_support_rate = len(supported) / len(claim_checked) if claim_checked else None
    unsupported_claim_rate = len(unsupported) / len(claim_checked) if claim_checked else None
    if unsupported:
        issues.append({"metric": "claim_support",
                       "reason": f"{len(unsupported)} 条论断未得到引文支持"})

    expected_facts = [fact for fact in case.get("expected_facts") or [] if fact]
    missing_facts: list[Any] = []
    for fact in expected_facts:
        if not _fact_present(fact, answer):
            missing_facts.append(fact)
    fact_coverage = ((len(expected_facts) - len(missing_facts)) / len(expected_facts)
                     if expected_facts else None)
    if missing_facts:
        issues.append({"metric": "fact_coverage",
                       "reason": "缺少要点：" + json.dumps(missing_facts, ensure_ascii=False)})

    expected_quotes = [quote for quote in case.get("expected_quotes") or [] if quote]
    normalized_answer = _norm(answer)
    matched_quotes = 0
    quote_texts = [_norm(item.get("quote") or "") for item in citations]
    for quote in expected_quotes:
        normalized_quote = _norm(quote)
        if normalized_quote and (normalized_quote in normalized_answer
                                 or any(normalized_quote in text for text in quote_texts)):
            matched_quotes += 1
    quote_recall = matched_quotes / len(expected_quotes) if expected_quotes else None

    should_refuse = case.get("should_refuse")
    predicted_refusal = is_refusal(answer)
    refusal_correct: Optional[bool] = None
    if isinstance(should_refuse, bool):
        refusal_correct = predicted_refusal == should_refuse
        if not refusal_correct:
            issues.append({
                "metric": "refusal",
                "reason": "应拒答但给出了回答" if should_refuse else "本可回答却拒答",
            })

    if result.get("verified") is False and case.get("require_citations", True):
        issues.append({"metric": "verified", "reason": "回答未被标记为可核验"})

    return {
        "case": case.get("id"),
        "question": case["question"],
        "answer_preview": answer[:240],
        "verified": bool(result.get("verified")),
        "citation_level": result.get("citation_level"),
        "metrics": {
            "citation_precision": _round(citation_precision),
            "citation_recall": _round(citation_recall),
            "claim_support_rate": _round(claim_support_rate),
            "unsupported_claim_rate": _round(unsupported_claim_rate),
            "fact_coverage": _round(fact_coverage),
            "quote_recall": _round(quote_recall),
            "refusal_correct": refusal_correct,
        },
        "counts": {
            "citations": len(citations),
            "verified_citations": len(verified),
            "claims_checked": len(claim_checked),
            "unsupported_claims": len(unsupported),
            "missing_facts": missing_facts,
        },
        "refusal_expected": should_refuse,
        "refusal_predicted": predicted_refusal,
        "issues": issues,
    }


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 4)


def _coerce_answers(answers: Any, cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if answers is None:
        return {}
    if isinstance(answers, dict):
        payload = answers.get("answers") if isinstance(answers.get("answers"), list) else answers
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items()
                    if isinstance(value, dict)}
        answers = payload
    if not isinstance(answers, list):
        raise ValueError("answers 必须是列表或按 case_id 索引的对象")
    mapped: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(answers):
        if not isinstance(item, dict):
            continue
        key = item.get("case_id") or item.get("case") or item.get("id")
        if key is None and index < len(cases):
            key = cases[index].get("id")
        if key is not None:
            mapped[str(key)] = item.get("result") if isinstance(item.get("result"), dict) else item
    return mapped


def evaluate_answers(dataset_path: Path | str, answers: Any = None,
                     run_case: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
                     limit: Optional[int] = None) -> dict[str, Any]:
    cases = load_answer_cases(dataset_path)
    if limit is not None:
        cases = cases[:max(1, int(limit))]
    mapped = _coerce_answers(answers, cases)
    details: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        key = str(case.get("id") or case["question"])
        result = mapped.get(key)
        if result is None and run_case is not None:
            try:
                result = run_case(case)
            except Exception as error:  # keep evaluating the remaining cases
                result = {"error": f"{type(error).__name__}: {error}"}
        if result is None:
            result = {"error": "缺少该问题的回答结果"}
        scored = score_answer(case, result)
        details.append(scored)
        failures.extend({"case": scored["case"], **issue} for issue in scored["issues"])

    metrics = {
        "citation_precision": _mean(item["metrics"]["citation_precision"] for item in details),
        "citation_recall": _mean(item["metrics"]["citation_recall"] for item in details),
        "claim_support_rate": _mean(item["metrics"]["claim_support_rate"] for item in details),
        "unsupported_claim_rate": _mean(
            item["metrics"]["unsupported_claim_rate"] for item in details),
        "fact_coverage": _mean(item["metrics"]["fact_coverage"] for item in details),
        "quote_recall": _mean(item["metrics"]["quote_recall"] for item in details),
        "verified_rate": _mean(1.0 if item["verified"] else 0.0 for item in details),
        "refusal_accuracy": _mean(
            1.0 if item["metrics"]["refusal_correct"] else 0.0
            for item in details
            if item["metrics"]["refusal_correct"] is not None),
    }
    return {
        "dataset": str(dataset_path),
        "cases": len(details),
        "metrics": metrics,
        "failures": failures,
        "details": details,
    }


def run_agent_cases(cases: list[dict[str, Any]], db_path: Path | str,
                    vectors_dir: Path | str, provider: str, key: str, model: str,
                    protocol: Optional[str] = None, base_url: Optional[str] = None,
                    max_steps: int = 5, enable_vectors: bool = False) -> list[dict[str, Any]]:
    """Run the real agent for each case and return results ready for scoring.

    Imports are local so this module stays usable for offline scoring without a
    configured model/provider stack.
    """
    from . import providers
    from .agent import AgentConfig, AgentRunner
    from .evaluation import _resolve_paper_id

    endpoint = providers.resolve_config(provider, protocol, base_url)
    answers: list[dict[str, Any]] = []
    for case in cases:
        paper_id = _resolve_paper_id(case, db_path)
        config = AgentConfig(
            provider=endpoint.id, key=key, model=model,
            mode="paper" if paper_id is not None else "library",
            paper_id=paper_id, max_steps=max_steps,
            protocol=endpoint.protocol, base_url=endpoint.base_url,
        )
        started = time.time()
        try:
            result = AgentRunner(config, db_path, vectors_dir,
                                 enable_vectors=enable_vectors).run(case["question"])
        except Exception as error:
            result = {"error": f"{type(error).__name__}: {error}"}
        result["case_id"] = case.get("id")
        result["provider"] = endpoint.id
        result["model"] = model
        result["elapsed_seconds"] = round(time.time() - started, 2)
        answers.append(result)
    return answers


def check_thresholds(report: dict[str, Any], thresholds: dict[str, float]) -> bool:
    """Return True when every supplied metric meets its minimum threshold."""
    for name, minimum in thresholds.items():
        value = (report.get("metrics") or {}).get(name)
        if value is None or value < minimum:
            return False
    return True
