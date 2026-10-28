#!/usr/bin/env python3
"""Score final agent answers against a golden set.

Two modes:

* offline scoring of saved results::

      python scripts/evaluate_answers.py --answers eval/answers.example.json

* live run against a local library (needs a configured chat model)::

      python scripts/evaluate_answers.py --db ~/PaperReader/database/paper_reader.sqlite3 \\
          --provider ollama --model qwen2.5:7b

Use ``--offline`` with live mode to build a temporary library from
``eval/corpus.json`` first.  This still calls the model, but needs no embedding
key and no personal data.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import answer_eval, evaluation, providers  # noqa: E402
from backend.credentials import safe_key  # noqa: E402
from backend.storage import VECTORS_DIR  # noqa: E402


def _load_answers(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        name: value for name, value in {
            "citation_precision": args.min_citation_precision,
            "claim_support_rate": args.min_claim_support,
            "fact_coverage": args.min_fact_coverage,
            "refusal_accuracy": args.min_refusal_accuracy,
        }.items() if value > 0
    }


def _run_live(args: argparse.Namespace, db_path: Path, vectors_dir: Path,
              cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    endpoint = providers.resolve_config(args.provider, args.protocol, args.base_url)
    key = safe_key(endpoint.id) or ""
    if endpoint.requires_key and not key:
        raise SystemExit(f"未配置 {endpoint.label} 的 API Key，无法运行在线评测")
    model = args.model or endpoint.default_model
    if not model:
        raise SystemExit("请用 --model 指定模型名称")
    print(f"运行 {len(cases)} 个问题 · {endpoint.id} · {model}", file=sys.stderr)
    return answer_eval.run_agent_cases(
        cases, db_path, vectors_dir, endpoint.id, key, model,
        protocol=endpoint.protocol, base_url=endpoint.base_url,
        max_steps=args.max_steps, enable_vectors=args.vectors_enabled,
    )


def _print_report(report: dict[str, Any]) -> int:
    metrics = report["metrics"]
    print(f"答案评测 · {report['cases']} 个问题 · {report['dataset']}")
    for name, label in [
        ("citation_precision", "引用精确率"),
        ("citation_recall", "引用召回率"),
        ("claim_support_rate", "论断支持率"),
        ("unsupported_claim_rate", "无支持论断率"),
        ("fact_coverage", "要点覆盖率"),
        ("quote_recall", "原文引用召回率"),
        ("verified_rate", "可核验回答率"),
        ("refusal_accuracy", "拒答判断准确率"),
    ]:
        value = metrics.get(name)
        shown = "—" if value is None else f"{value:.2%}"
        print(f"  {label}: {shown}")
    if report["failures"]:
        print("\n未通过项:")
        for failure in report["failures"]:
            print(f"  - [{failure.get('metric')}] {failure.get('case')}: {failure.get('reason')}")
    else:
        print("\n全部检查通过。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper Reader answer-level evaluation")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "eval" / "answer_cases.json")
    parser.add_argument("--answers", type=Path, default=None,
                        help="saved agent results (list, or {case_id: result}); skips model calls")
    parser.add_argument("--offline", action="store_true",
                        help="build a temporary fixture from --corpus before live evaluation")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval" / "corpus.json")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default="")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--no-vectors", action="store_true",
                        help="disable vector retrieval during live evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-citation-precision", type=float, default=0.0)
    parser.add_argument("--min-claim-support", type=float, default=0.0)
    parser.add_argument("--min-fact-coverage", type=float, default=0.0)
    parser.add_argument("--min-refusal-accuracy", type=float, default=0.0)
    parser.add_argument("--save-answers", type=Path, default=None,
                        help="write live results for later offline scoring")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the report JSON to a file")
    args = parser.parse_args()

    cases = answer_eval.load_answer_cases(args.dataset)
    if args.limit:
        cases = cases[:max(1, args.limit)]

    answers: Optional[Any] = None
    temp: Optional[tempfile.TemporaryDirectory] = None
    if args.answers is not None:
        answers = _load_answers(args.answers)
    else:
        if args.db is None and not args.offline:
            parser.error("活体评测需要 --db 或 --offline（或提供 --answers）")
        if args.offline:
            temp = tempfile.TemporaryDirectory(prefix="paper-reader-answer-eval-")
            db_path, vectors_dir = evaluation.build_fixture(args.corpus, temp.name)
        else:
            db_path = args.db
            vectors_dir = args.vectors or VECTORS_DIR
        args.vectors_enabled = not args.no_vectors and bool(safe_key("qwen"))
        answers = _run_live(args, db_path, vectors_dir, cases)
        if args.save_answers:
            args.save_answers.write_text(
                json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        if temp is not None:
            temp.cleanup()

    report = answer_eval.evaluate_answers(args.dataset, answers=answers)
    thresholds = _thresholds(args)
    ok = answer_eval.check_thresholds(report, thresholds) if thresholds else not report["failures"]

    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
        if thresholds:
            status = "通过" if ok else "未达到阈值"
            print(f"\n{status}: {thresholds}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
