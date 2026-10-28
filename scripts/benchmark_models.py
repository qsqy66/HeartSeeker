#!/usr/bin/env python3
"""Benchmark chat models on the answer-level golden set.

This is the local-model comparison entry point.  Point it at Ollama or any
OpenAI-compatible local server; it runs the full agent (tools, retrieval,
citation verification) for every question and reports answer quality plus
latency.  No embedding key is required when vector search is disabled.

Examples:
    python scripts/benchmark_models.py --offline \\
        --models qwen2.5:7b,llama3.1:8b

    python scripts/benchmark_models.py --db ~/PaperReader/database/paper_reader.sqlite3 \\
        --provider ollama --base-url http://127.0.0.1:11434/v1 \\
        --models qwen2.5:14b --output benchmarks/qwen.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import answer_eval, evaluation, providers  # noqa: E402
from backend.credentials import safe_key  # noqa: E402
from backend.storage import VECTORS_DIR  # noqa: E402

METRIC_LABELS = [
    ("citation_precision", "引用精确"),
    ("citation_recall", "引用召回"),
    ("claim_support_rate", "论断支持"),
    ("unsupported_claim_rate", "无支持论断"),
    ("fact_coverage", "要点覆盖"),
    ("refusal_accuracy", "拒答准确"),
    ("verified_rate", "可核验"),
]


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.2%}"


def _score(answers: list[dict[str, Any]], dataset: Path) -> dict[str, Any]:
    return answer_eval.evaluate_answers(dataset, answers=answers)


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["模型", *[label for _, label in METRIC_LABELS], "平均耗时", "失败"]
    print("\t".join(headers))
    for row in rows:
        metrics = row["report"]["metrics"]
        values = [str(metrics.get(name)) if metrics.get(name) is None
                  else _fmt(metrics.get(name)) for name, _ in METRIC_LABELS]
        elapsed = [item.get("elapsed_seconds", 0) for item in row["answers"]
                   if isinstance(item.get("elapsed_seconds"), (int, float))]
        print("\t".join([
            row["model"], *values,
            f"{statistics.mean(elapsed):.1f}s" if elapsed else "—",
            str(len(row["report"]["failures"])),
        ]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark chat models on the answer set")
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval" / "answer_cases.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "eval" / "corpus.json")
    parser.add_argument("--offline", action="store_true", default=False,
                        help="build a temporary fixture from --corpus (default if --db is absent)")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--vectors", type=Path, default=None)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--models", default="",
                        help="comma-separated model names; default is the provider default")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-vectors", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    endpoint = providers.resolve_config(args.provider, args.protocol, args.base_url)
    key = safe_key(endpoint.id) or ""
    if endpoint.requires_key and not key:
        raise SystemExit(f"未配置 {endpoint.label} 的 API Key")
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not models:
        if not endpoint.default_model:
            raise SystemExit("请用 --models 指定至少一个模型")
        models = [endpoint.default_model]

    cases = answer_eval.load_answer_cases(args.dataset)
    if args.limit:
        cases = cases[:max(1, args.limit)]

    temp = None
    if args.db is None or args.offline:
        temp = tempfile.TemporaryDirectory(prefix="paper-reader-benchmark-")
        db_path, vectors_dir = evaluation.build_fixture(args.corpus, temp.name)
    else:
        db_path = args.db
        vectors_dir = args.vectors or VECTORS_DIR
    enable_vectors = not args.no_vectors and bool(safe_key("qwen"))

    rows: list[dict[str, Any]] = []
    try:
        for model in models:
            print(f"\n== {model} ==", file=sys.stderr)
            started = time.time()
            answers = answer_eval.run_agent_cases(
                cases, db_path, vectors_dir, endpoint.id, key, model,
                protocol=endpoint.protocol, base_url=endpoint.base_url,
                max_steps=args.max_steps, enable_vectors=enable_vectors,
            )
            report = _score(answers, args.dataset)
            rows.append({"model": model, "elapsed_seconds": round(time.time() - started, 2),
                         "report": report, "answers": answers})
    finally:
        if temp is not None:
            temp.cleanup()

    payload = {"provider": endpoint.id, "dataset": str(args.dataset),
               "cases": len(cases), "results": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_table(rows)
        print("\n指标越高越好；“无支持论断率”越低越好。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
