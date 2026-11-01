from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import citations, verification  # noqa: E402


class ClaimVerificationTestCase(unittest.TestCase):
    def test_numeric_mismatch_is_a_contradiction(self) -> None:
        support = verification.verify_claim(
            "死亡风险降低 20%",
            "hazard ratio 0.92; 95% CI 0.78 to 1.09",
        )
        self.assertFalse(support.supported)
        self.assertTrue(support.contradiction)
        self.assertTrue(any("数值" in reason for reason in support.reasons))

    def test_percent_and_fraction_are_equivalent(self) -> None:
        support = verification.verify_claim("risk reduction 10%",
                                            "risk reduction was 0.10")
        self.assertTrue(support.supported)
        self.assertFalse(support.contradiction)

    def test_negation_mismatch_is_a_contradiction(self) -> None:
        support = verification.verify_claim("the drug did not reduce mortality",
                                            "the drug reduced mortality significantly")
        self.assertFalse(support.supported)
        self.assertTrue(support.contradiction)

    def test_nonsense_variant_is_not_treated_as_negation(self) -> None:
        support = verification.verify_claim(
            "无义变异位置与疾病临床严重程度相关",
            "Our findings suggest a novel association between the location of "
            "nonsense variants and the clinical severity of the disease.",
        )
        self.assertTrue(support.supported)
        self.assertFalse(support.contradiction)

    def test_cross_language_negation_is_not_judged_by_keyword(self) -> None:
        support = verification.verify_claim(
            "作者说明该相关性远不穷尽",
            "this is far from exhaustive",
        )
        self.assertFalse(support.contradiction)
        self.assertTrue(support.supported)

    def test_cross_language_claim_is_not_penalized_for_coverage(self) -> None:
        support = verification.verify_claim(
            "主要终点为 12 个月全因死亡",
            "The primary outcome was all-cause mortality at 12 months.",
        )
        self.assertTrue(support.supported)
        self.assertIsNone(support.coverage)
        self.assertFalse(support.contradiction)

    def test_optional_nli_judge_can_reject(self) -> None:
        class RejectingJudge:
            name = "rejecting"

            def judge(self, claim: str, quote: str) -> float:
                return 0.1

        support = verification.verify_claim(
            "all-cause mortality", "all-cause mortality at 12 months",
            judge=RejectingJudge(),
        )
        self.assertFalse(support.supported)
        self.assertEqual(support.judge, "rejecting")

    def test_nli_registry_supports_plugins(self) -> None:
        class ConstantJudge:
            name = "constant"

            def judge(self, claim: str, quote: str) -> float:
                return 0.9

        verification.register_nli("constant", lambda model: ConstantJudge())
        self.assertIsInstance(verification.get_nli("constant"), ConstantJudge)
        self.assertIsNone(verification.get_nli("none"))


class CitationClaimIntegrationTestCase(unittest.TestCase):
    def _evidence(self):
        return [{"paper_id": 1, "page_number": 1,
                 "text": "Hazard ratio, 0.92; 95% CI, 0.78 to 1.09. "
                         "The primary outcome was cardiovascular death."}]

    def test_contradictory_claim_invalidates_citation(self) -> None:
        result = citations.validate_citations(
            "治疗降低死亡风险 20% [Page 1]",
            [{"paper_id": 1, "page": 1,
              "quote": "Hazard ratio, 0.92; 95% CI, 0.78 to 1.09.",
              "claim": "治疗降低死亡风险 20%"}],
            self._evidence(), default_paper_id=1, require_citation=True,
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.citations[0].claim_supported)
        self.assertEqual(result.claim_failures, 1)
        self.assertTrue(result.unsupported_claims)
        self.assertTrue(any("论断与引文不一致" in item["reason"] for item in result.invalid))

    def test_supported_claim_is_marked(self) -> None:
        result = citations.validate_citations(
            "Primary outcome [Page 1]",
            [{"paper_id": 1, "page": 1,
              "quote": "The primary outcome was cardiovascular death.",
              "claim": "The primary outcome was cardiovascular death"}],
            self._evidence(), default_paper_id=1, require_citation=True,
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.citations[0].claim_supported)
        self.assertEqual(result.claim_failures, 0)
        self.assertIn("claim_supported", result.as_dicts()[0])

    def test_cross_language_claim_is_not_invalidated(self) -> None:
        result = citations.validate_citations(
            "主要终点是心血管死亡 [Page 1]",
            [{"paper_id": 1, "page": 1,
              "quote": "The primary outcome was cardiovascular death.",
              "claim": "主要终点是心血管死亡"}],
            self._evidence(), default_paper_id=1, require_citation=True,
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.citations[0].claim_supported)


if __name__ == "__main__":
    unittest.main()
