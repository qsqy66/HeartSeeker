#!/usr/bin/env python3
"""Run deterministic retrieval/citation evaluation.

Offline (default) builds a temporary DB from eval/corpus.json, so it needs no
API key and no network.  Live mode evaluates questions against the local
library and can opt into vector search with --vectors.

Examples:
    python scripts/evaluate.py
    python scripts/evaluate.py --json
    python scripts/evaluate.py --db ~/HeartSeeker/database/heartseeker.sqlite3 \
        --vectors ~/HeartSeeker/vectors --dataset eval/live_cases.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import evaluation  # noqa: E402
from backend.credentials import safe_key  # noqa: E402
from backend.storage import VECTORS_DIR  # noqa: E402


def _failed(report: dict, threshold_recall: float, threshold_citation: float) -> bool:
    retrieval = report["retrieval"]
    citation = report["citations"]
    failed = retrieval["recall"] is not None and retrieval["recall"] < threshold_recall
    for name in ("positive_pass_rate", "negative_reject_rate"):
        value = citation.get(name)
        if value is not None and value < threshold_citation:
            failed = True
    return failed


def _print_report(report: dict, threshold_recall: float, threshold_citation: float) -> int:
    retrieval = report["retrieval"]
    citation = report["citations"]
    print(f"模式: {report['mode']} · {report['cases']} 个问题 · top-{report['top_k']}"
          f" · 向量检索: {'开' if report['vector_search'] else '关'}")
    print(f"检索召回率: {retrieval['recall']}    MRR: {retrieval['mrr']}    "
          f"纯关键词召回率: {retrieval['keyword_recall']}")
    print(f"引文核验通过率: {citation['positive_pass_rate']}    "
          f"虚构引文拒绝率: {citation['negative_reject_rate']}    "
          f"检查数: {citation['checked']}")
    structured = report.get("structured")
    if structured:
        print(f"结构化字段证据率: {structured['evidence_rate']}    "
              f"缓存论文: {structured['cached_papers']}")
    if report["failures"]:
        print("\n未通过项:")
        for failure in report["failures"]:
            print(f"  - [{failure.get('metric')}] {failure.get('case')}: {failure.get('reason')}")
    else:
        print("\n全部检查通过。")

    failed = _failed(report, threshold_recall, threshold_citation)
    if failed:
        print(f"\n✗ 未达到阈值：recall≥{threshold_recall}, citation≥{threshold_citation}",
              file=sys.stderr)
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="觅心（HeartSeeker_Med） retrieval/citation evaluation")
    parser.add_argument("--offline", action="store_true",
                        help="force the bundled fixture (default when --db is absent)")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "eval" / "offline_cases.json",
                        help="JSON golden set")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval" / "corpus.json",
                        help="offline fixture corpus")
    parser.add_argument("--db", type=Path, default=None,
                        help="evaluate a real local database instead of the fixture")
    parser.add_argument("--vectors", type=Path, default=None,
                        help="LanceDB directory for live vector search")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--no-vectors", action="store_true",
                        help="live mode: disable vector search (default when no Qwen key)")
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--min-citation", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    if args.db is not None and not args.offline:
        vectors = args.vectors or VECTORS_DIR
        key = safe_key("qwen")
        enable_vectors = not args.no_vectors and bool(key)
        report = evaluation.evaluate_live(
            args.dataset, args.db, vectors, top_k=args.top_k,
            enable_vectors=enable_vectors, api_key=key,
        )
    else:
        report = evaluation.evaluate_offline(
            args.corpus, args.dataset, top_k=args.top_k,
        )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1 if _failed(report, args.min_recall, args.min_citation) else 0
    return _print_report(report, args.min_recall, args.min_citation)


if __name__ == "__main__":
    raise SystemExit(main())
