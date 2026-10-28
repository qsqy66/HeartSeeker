from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import answer_eval  # noqa: E402

DATASET = ROOT / "eval" / "answer_cases.json"


def _case(**overrides):
    case = {
        "id": "case-1",
        "question": "主要结局是什么？",
        "paper_id": 1,
        "expected_pages": [1],
        "expected_facts": ["心血管死亡", "心力衰竭住院"],
        "expected_quotes": ["主要结局是心血管死亡或心力衰竭住院的复合终点。"],
        "should_refuse": False,
    }
    case.update(overrides)
    return case


def _perfect_result():
    quote = "主要结局是心血管死亡或心力衰竭住院的复合终点。"
    return {
        "content": "主要结局是心血管死亡或心力衰竭住院的复合终点 [Page 1]。",
        "citations": [{
            "paper_id": 1, "page": 1, "quote": quote,
            "claim": "主要结局是心血管死亡或心力衰竭住院",
            "verified": True, "level": "quote", "claim_supported": True,
        }],
        "verified": True,
        "citation_level": "quote",
    }


class ScoreAnswerTestCase(unittest.TestCase):
    def test_perfect_answer_scores_one(self) -> None:
        scored = answer_eval.score_answer(_case(), _perfect_result())
        metrics = scored["metrics"]
        self.assertEqual(metrics["citation_precision"], 1.0)
        self.assertEqual(metrics["citation_recall"], 1.0)
        self.assertEqual(metrics["claim_support_rate"], 1.0)
        self.assertEqual(metrics["unsupported_claim_rate"], 0.0)
        self.assertEqual(metrics["fact_coverage"], 1.0)
        self.assertEqual(metrics["quote_recall"], 1.0)
        self.assertTrue(metrics["refusal_correct"])
        self.assertEqual(scored["issues"], [])

    def test_unverified_citation_and_missing_fact_are_reported(self) -> None:
        result = _perfect_result()
        result["citations"][0]["verified"] = False
        result["verified"] = False
        result["content"] = "主要结局是死亡 [Page 1]。"
        scored = answer_eval.score_answer(_case(), result)
        self.assertEqual(scored["metrics"]["citation_precision"], 0.0)
        self.assertEqual(scored["metrics"]["citation_recall"], 0.0)
        self.assertEqual(scored["metrics"]["fact_coverage"], 0.0)
        reasons = {issue["metric"] for issue in scored["issues"]}
        self.assertIn("citation_precision", reasons)
        self.assertIn("citation_recall", reasons)
        self.assertIn("fact_coverage", reasons)
        self.assertIn("verified", reasons)

    def test_unsupported_claim_is_flagged(self) -> None:
        result = _perfect_result()
        result["citations"][0]["claim_supported"] = False
        scored = answer_eval.score_answer(_case(), result)
        self.assertEqual(scored["metrics"]["claim_support_rate"], 0.0)
        self.assertEqual(scored["metrics"]["unsupported_claim_rate"], 1.0)
        self.assertIn("claim_support", {issue["metric"] for issue in scored["issues"]})

    def test_fact_alternatives(self) -> None:
        case = _case(expected_facts=[["心血管死亡", "cardiovascular death"], "心力衰竭住院"])
        result = _perfect_result()
        result["content"] = "主要结局包括 cardiovascular death 和心力衰竭住院。"
        scored = answer_eval.score_answer(case, result)
        self.assertEqual(scored["metrics"]["fact_coverage"], 1.0)

    def test_refusal_classification(self) -> None:
        case = _case(expected_pages=[], expected_facts=[], expected_quotes=[],
                     should_refuse=True, require_citations=False)
        refusal = {"content": "现有证据不足，无法回答该问题。", "citations": [],
                   "verified": False}
        scored = answer_eval.score_answer(case, refusal)
        self.assertTrue(scored["refusal_predicted"])
        self.assertTrue(scored["metrics"]["refusal_correct"])

        answer = {"content": "该药可降低儿童死亡率。", "citations": [], "verified": True}
        scored = answer_eval.score_answer(case, answer)
        self.assertFalse(scored["metrics"]["refusal_correct"])
        self.assertIn("refusal", {issue["metric"] for issue in scored["issues"]})

    def test_over_refusal_is_penalised(self) -> None:
        result = {"content": "现有证据不足，无法回答。", "citations": [], "verified": False}
        scored = answer_eval.score_answer(_case(), result)
        self.assertFalse(scored["metrics"]["refusal_correct"])
        self.assertTrue(any("本可回答却拒答" in issue["reason"] for issue in scored["issues"]))

    def test_error_result_is_surfaced(self) -> None:
        scored = answer_eval.score_answer(_case(), {"error": "provider down"})
        self.assertIn("run", {issue["metric"] for issue in scored["issues"]})


class EvaluateAnswersTestCase(unittest.TestCase):
    def test_dataset_loads_all_cases(self) -> None:
        cases = answer_eval.load_answer_cases(DATASET)
        self.assertGreaterEqual(len(cases), 8)
        self.assertTrue(any(case.get("should_refuse") for case in cases))

    def test_answers_join_by_case_id_and_aggregate(self) -> None:
        cases = answer_eval.load_answer_cases(DATASET)
        answers = []
        for case in cases:
            quote = (case.get("expected_quotes") or [""])[0]
            pages = case.get("expected_pages") or []
            refused = bool(case.get("should_refuse"))
            content = ("现有证据不足，无法回答。" if refused
                       else "回答：" + " ".join(str(fact) for fact in case.get("expected_facts") or [])
                       + (f" [Page {pages[0]}]" if pages else ""))
            answers.append({
                "case_id": case["id"],
                "content": content,
                "verified": not refused,
                "citations": ([{
                    "paper_id": case.get("paper_id"), "page": pages[0], "quote": quote,
                    "claim": "支持论断", "verified": True, "level": "quote",
                    "claim_supported": True,
                }] if pages else []),
            })
        report = answer_eval.evaluate_answers(DATASET, answers=answers)
        self.assertEqual(report["cases"], len(cases))
        self.assertEqual(report["metrics"]["citation_precision"], 1.0)
        self.assertEqual(report["metrics"]["citation_recall"], 1.0)
        self.assertEqual(report["metrics"]["fact_coverage"], 1.0)
        self.assertEqual(report["metrics"]["refusal_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["unsupported_claim_rate"], 0.0)
        self.assertFalse(report["failures"])
        self.assertTrue(answer_eval.check_thresholds(
            report, {"citation_precision": 0.9, "fact_coverage": 0.9, "refusal_accuracy": 1.0}))
        self.assertFalse(answer_eval.check_thresholds(report, {"fact_coverage": 1.01}))

    def test_run_case_callback_and_limits(self) -> None:
        seen: list[str] = []

        def run_case(case):
            seen.append(case["id"])
            return {"content": "现有证据不足。", "citations": [], "verified": False}

        report = answer_eval.evaluate_answers(DATASET, run_case=run_case, limit=2)
        self.assertEqual(len(seen), 2)
        self.assertEqual(report["cases"], 2)

    def test_run_case_failure_does_not_abort(self) -> None:
        def run_case(case):
            raise RuntimeError("模型不可用")

        report = answer_eval.evaluate_answers(DATASET, run_case=run_case, limit=1)
        self.assertEqual(report["cases"], 1)
        self.assertTrue(any(issue["metric"] == "run" for issue in report["failures"]))

    def test_missing_answer_is_reported(self) -> None:
        report = answer_eval.evaluate_answers(DATASET, answers=[], limit=1)
        self.assertTrue(any(issue["metric"] == "run" for issue in report["failures"]))


if __name__ == "__main__":
    unittest.main()
