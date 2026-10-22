from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import evaluation  # noqa: E402


class EvaluationTestCase(unittest.TestCase):
    def test_offline_golden_set_passes_retrieval_and_citations(self) -> None:
        report = evaluation.evaluate_offline(
            ROOT / "eval" / "corpus.json",
            ROOT / "eval" / "offline_cases.json",
            top_k=8,
        )
        self.assertEqual(report["cases"], 8)
        self.assertEqual(report["retrieval"]["recall"], 1.0)
        self.assertGreaterEqual(report["retrieval"]["mrr"], 0.8)
        self.assertEqual(report["citations"]["positive_pass_rate"], 1.0)
        self.assertEqual(report["citations"]["negative_reject_rate"], 1.0)
        self.assertEqual(report["failures"], [])

    def test_fixture_supports_library_scope_and_chinese(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            db_path, vectors = evaluation.build_fixture(ROOT / "eval" / "corpus.json", temp)
            report = evaluation.evaluate(
                ROOT / "eval" / "offline_cases.json", db_path, vectors,
                top_k=8, enable_vectors=False,
            )
        library_details = [item for item in report["details"]
                           if item.get("scope") == "library"]
        self.assertTrue(library_details)
        self.assertTrue(all(item["rank"] is not None for item in library_details))

    def test_title_resolution_for_live_datasets(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            db_path, vectors = evaluation.build_fixture(ROOT / "eval" / "corpus.json", temp)
            dataset = Path(temp) / "live.json"
            dataset.write_text(
                '{"cases": [{"id": "by-title", "paper_title": "运动康复", '
                '"question": "主要结局是什么？", "expected_pages": [1]}]}',
                encoding="utf-8",
            )
            report = evaluation.evaluate(dataset, db_path, vectors, top_k=5)
        self.assertIsNotNone(report["details"][0]["rank"])

    def test_citation_negative_control_rejects_fabrication(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            db_path, vectors = evaluation.build_fixture(ROOT / "eval" / "corpus.json", temp)
            report = evaluation.evaluate(
                ROOT / "eval" / "offline_cases.json", db_path, vectors,
                top_k=8, enable_vectors=False,
            )
        self.assertEqual(report["citations"]["checked"], 8)
        self.assertEqual(report["citations"]["negative_reject_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
