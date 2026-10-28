from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import agent, clinical, evidence, ontology  # noqa: E402


class ClinicalCalculatorTestCase(unittest.TestCase):
    def test_cha2ds2_vasc_scores_and_stratifies(self) -> None:
        result = clinical.calculate("cha2ds2_vasc", {
            "heart_failure": True, "hypertension": True, "age": 72,
            "diabetes": True, "sex": "male",
        })
        # HF 1 + HTN 1 + age 65-74 1 + DM 1
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["risk_level"], "high")
        self.assertIn("抗凝", result["interpretation"])

    def test_female_sex_alone_is_not_an_anticoagulation_indication(self) -> None:
        result = clinical.calculate("cha2ds2_vasc", {"age": 50, "sex": "female"})
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["risk_level"], "low")
        self.assertIn("女性性别本身", result["interpretation"])

    def test_has_bled_high_score(self) -> None:
        result = clinical.calculate("has_bled", {
            "hypertension": True, "renal_disease": True, "age": 70,
        })
        self.assertEqual(result["score"], 3)
        self.assertEqual(result["risk_level"], "high")
        self.assertIn("纠正可逆", result["interpretation"])

    def test_curb_65(self) -> None:
        result = clinical.calculate("curb_65", {
            "confusion": True, "urea_mmol_l": 8, "respiratory_rate": 30,
            "systolic_bp": 85, "diastolic_bp": 55, "age": 70,
        })
        self.assertEqual(result["score"], 5)
        self.assertEqual(result["risk_level"], "high")
        self.assertIn("重症", result["interpretation"])

    def test_wells_dvt_two_tier(self) -> None:
        result = clinical.calculate("wells_dvt", {
            "cancer": True, "tenderness": True, "swelling": True,
            "previous_dvt": True,
        })
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["details"]["two_tier"], "likely")
        self.assertEqual(result["risk_level"], "high")

    def test_egfr_ckd_epi_2021(self) -> None:
        result = clinical.calculate("egfr_ckd_epi_2021", {
            "age": 60, "sex": "male", "creatinine": 1.0,
            "creatinine_unit": "mg_dl",
        })
        self.assertGreater(result["score"], 80)
        self.assertLess(result["score"], 90)
        self.assertEqual(result["details"]["stage"], "G2")
        female = clinical.calculate("egfr_ckd_epi_2021", {
            "age": 60, "sex": "female", "creatinine": 1.0,
        })
        self.assertLess(female["score"], result["score"])

    def test_umol_l_creatinine_is_converted(self) -> None:
        mg_dl = clinical.calculate("egfr_ckd_epi_2021", {
            "age": 50, "sex": "male", "creatinine": 1.0})
        umol_l = clinical.calculate("egfr_ckd_epi_2021", {
            "age": 50, "sex": "male", "creatinine": 88.4,
            "creatinine_unit": "umol_l"})
        self.assertAlmostEqual(mg_dl["score"], umol_l["score"], places=1)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(clinical.CalculatorError):
            clinical.calculate("not_a_score", {})
        with self.assertRaises(clinical.CalculatorError):
            clinical.calculate("curb_65", {"age": 70})  # missing required numbers
        with self.assertRaises(clinical.CalculatorError):
            clinical.calculate("cha2ds2_vasc", {"age": 70, "sex": "other"})
        with self.assertRaises(clinical.CalculatorError):
            clinical.calculate("cha2ds2_vasc", {"age": 70, "unknown": True})

    def test_catalog_lists_inputs(self) -> None:
        catalog = {item["key"]: item for item in clinical.list_calculators()}
        self.assertIn("cha2ds2_vasc", catalog)
        self.assertIn("egfr_ckd_epi_2021", catalog)
        age = next(item for item in catalog["curb_65"]["inputs"] if item["key"] == "age")
        self.assertEqual(age["kind"], "number")
        self.assertEqual(age["unit"], "岁")


class OntologyTestCase(unittest.TestCase):
    def test_lookup_by_chinese_alias_and_icd(self) -> None:
        chinese = ontology.lookup("心衰")
        self.assertEqual(chinese[0]["id"], "HF")
        self.assertEqual(chinese[0]["match_score"], 1.0)
        coded = ontology.lookup("I50.9")
        self.assertEqual(coded[0]["id"], "HF")
        abbreviation = ontology.lookup("PE")
        self.assertEqual(abbreviation[0]["id"], "PE")

    def test_lookup_inside_a_question(self) -> None:
        matches = ontology.lookup("房颤患者围术期如何抗凝？")
        self.assertEqual(matches[0]["id"], "AF")
        self.assertTrue(0 < matches[0]["match_score"] < 1)

    def test_related_concepts_and_stats(self) -> None:
        loaded = ontology.get_ontology()
        related_ids = {item["id"] for item in loaded.related("AF")}
        self.assertIn("HF", related_ids)
        stats = loaded.stats()
        self.assertGreater(stats["concepts"], 20)
        self.assertIn("心血管", stats["categories"])

    def test_external_ontology_loader(self) -> None:
        payload = {
            "version": "test-1",
            "concepts": [
                {"id": "TEST", "term": "测试疾病", "aliases": ["test disease"],
                 "icd10": ["Z00.0"], "category": "测试", "related": []}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ontology.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            loaded = ontology.load_ontology(path)
        self.assertEqual(loaded.version, "test-1")
        self.assertEqual(loaded.lookup("test disease")[0]["id"], "TEST")
        self.assertEqual(loaded.lookup("Z00.0")[0]["id"], "TEST")

    def test_malformed_ontology_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"concepts": [{"id": "A"}]}), encoding="utf-8")
            with self.assertRaises(ontology.OntologyError):
                ontology.load_ontology(path)
            duplicate = Path(tmp) / "duplicate.json"
            duplicate.write_text(json.dumps({"concepts": [
                {"id": "A", "term": "甲"}, {"id": "A", "term": "乙"},
            ]}), encoding="utf-8")
            with self.assertRaises(ontology.OntologyError):
                ontology.load_ontology(duplicate)


class EvidenceAppraisalTestCase(unittest.TestCase):
    def test_randomized_trial_starts_high(self) -> None:
        result = evidence.appraise(
            "Randomized, double-blind, placebo-controlled trial. A total of 450 "
            "patients were randomized. Allocation concealment was ensured. "
            "Intention-to-treat analysis was performed. The hazard ratio was "
            "0.75 (95% CI 0.60 to 0.91); lost to follow-up was 5%."
        )
        self.assertEqual(result["design"]["type"], "rct")
        self.assertEqual(result["certainty"], "high")
        self.assertEqual(result["sample_size"], 450)
        self.assertEqual(result["stars"], 4)

    def test_small_unadjusted_cohort_is_very_low(self) -> None:
        result = evidence.appraise(
            "Retrospective cohort study of 80 patients with heart failure. "
            "Outcomes were compared without adjustment for confounders."
        )
        self.assertEqual(result["design"]["type"], "cohort")
        self.assertEqual(result["certainty"], "very_low")
        domains = {domain["domain"]: domain for domain in result["domains"]}
        self.assertEqual(domains["risk_of_bias"]["concern"], "some")
        self.assertEqual(domains["imprecision"]["concern"], "serious")

    def test_meta_analysis_heterogeneity_downgrades(self) -> None:
        result = evidence.appraise(
            "Systematic review and meta-analysis. PubMed and Embase were searched. "
            "Risk of bias was assessed with RoB 2. Heterogeneity was high "
            "(I² = 80%). The pooled risk ratio was 0.70 (95% CI 0.55 to 0.90)."
        )
        self.assertEqual(result["design"]["type"], "systematic_review")
        domains = {domain["domain"]: domain for domain in result["domains"]}
        self.assertEqual(domains["inconsistency"]["downgrade"], 2)
        self.assertEqual(result["certainty"], "low")

    def test_short_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evidence.appraise("randomized trial")


class MedicalAgentToolsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite3"
        self.vectors = Path(self.tmp.name) / "vectors"
        self.vectors.mkdir()
        connection = sqlite3.connect(self.db)
        connection.executescript(
            "CREATE TABLE papers (id INTEGER PRIMARY KEY, title TEXT NOT NULL,"
            "page_count INTEGER, status TEXT, imported_at TEXT NOT NULL DEFAULT '');"
            "CREATE TABLE pages (id INTEGER PRIMARY KEY, paper_id INTEGER, page_number INTEGER,"
            "text TEXT);"
            "CREATE TABLE chunks (id INTEGER PRIMARY KEY, paper_id INTEGER, page_id INTEGER,"
            "text TEXT, heading TEXT NOT NULL DEFAULT '', section TEXT NOT NULL DEFAULT '');"
        )
        connection.execute("INSERT INTO papers VALUES(1,'Trial',10,'ready','2026-01-01')")
        connection.execute("INSERT INTO pages VALUES(1,1,1,'x')")
        connection.execute(
            "INSERT INTO chunks VALUES(1,1,1,?,'Methods','Methods')",
            ("Randomized, double-blind trial. A total of 450 patients were randomized. "
             "Allocation concealment was ensured. Intention-to-treat analysis. "
             "Hazard ratio 0.75 (95% CI 0.60 to 0.91); lost to follow-up was 5%.",),
        )
        connection.commit()
        connection.close()
        self.runner = agent.AgentRunner(
            agent.AgentConfig(provider="deepseek", key="test", model="test",
                              mode="paper", paper_id=1),
            self.db, self.vectors, chat_fn=lambda messages: "", enable_vectors=False,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_calculator_tool_lists_and_computes(self) -> None:
        listing, _ = self.runner._execute("clinical_calculator", {})
        self.assertTrue(listing["ok"])
        self.assertTrue(listing["calculators"])
        observation, _ = self.runner._execute("clinical_calculator", {
            "calculator": "wells_dvt",
            "inputs": {"cancer": True, "previous_dvt": True},
        })
        self.assertEqual(observation["result"]["score"], 2)

    def test_calculator_tool_accepts_json_string_inputs(self) -> None:
        observation, _ = self.runner._execute("clinical_calculator", {
            "calculator": "curb_65",
            "inputs": json.dumps({"urea_mmol_l": 8, "respiratory_rate": 18,
                                  "systolic_bp": 120, "diastolic_bp": 80, "age": 70}),
        })
        self.assertEqual(observation["result"]["score"], 2)

    def test_ontology_tool(self) -> None:
        observation, _ = self.runner._execute("lookup_ontology", {"query": "心梗"})
        self.assertEqual(observation["matches"][0]["id"], "AMI")

    def test_appraise_tool_with_paper_id(self) -> None:
        observation, _ = self.runner._execute("appraise_evidence", {"paper_id": 1})
        self.assertEqual(observation["appraisal"]["certainty"], "high")

    def test_appraise_tool_with_inline_text(self) -> None:
        observation, _ = self.runner._execute("appraise_evidence", {
            "text": "Prospective cohort study of 1200 adults. Multivariable "
                    "adjustment was performed for age and sex. 95% CI 0.8 to 1.4.",
        })
        self.assertEqual(observation["appraisal"]["design"]["type"], "cohort")

    def test_tool_specs_are_registered(self) -> None:
        for name in ("clinical_calculator", "lookup_ontology", "appraise_evidence"):
            self.assertIn(name, agent.TOOL_SPECS)


if __name__ == "__main__":
    unittest.main()
