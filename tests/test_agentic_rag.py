from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import agent, citations, medical_terms, retrieval, safety, structured  # noqa: E402


SCHEMA = """
CREATE TABLE papers (
  id INTEGER PRIMARY KEY, title TEXT NOT NULL, file_name TEXT NOT NULL,
  stored_path TEXT NOT NULL, sha256 TEXT NOT NULL, page_count INTEGER NOT NULL,
  status TEXT NOT NULL, imported_at TEXT NOT NULL, vectorized INTEGER NOT NULL DEFAULT 0,
  needs_ocr INTEGER NOT NULL DEFAULT 0, embedding_model TEXT NOT NULL DEFAULT '',
  embedding_dim INTEGER
);
CREATE TABLE pages (
  id INTEGER PRIMARY KEY, paper_id INTEGER NOT NULL, page_number INTEGER NOT NULL, text TEXT NOT NULL
);
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY, paper_id INTEGER NOT NULL, page_id INTEGER NOT NULL,
  chunk_index INTEGER NOT NULL, text TEXT NOT NULL, x0 REAL, y0 REAL, x1 REAL, y1 REAL,
  section TEXT NOT NULL DEFAULT '', heading TEXT NOT NULL DEFAULT ''
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
"""


class AgenticRagTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite3"
        self.vectors = Path(self.tmp.name) / "vectors"
        self.vectors.mkdir()
        connection = sqlite3.connect(self.db)
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO papers(id,title,file_name,stored_path,sha256,page_count,status,imported_at,"
            "vectorized,needs_ocr) VALUES(1,'Heart Failure Trial','a.pdf','a.pdf','sha-1',1,'ready',"
            "'2026-01-01',0,0)"
        )
        connection.execute(
            "INSERT INTO pages(id,paper_id,page_number,text) VALUES(1,1,1,?)",
            ("The primary outcome was all-cause mortality at 12 months. "
             "The trial enrolled 1200 patients with heart failure. "
             "心力衰竭患者接受标准治疗后随访一年。",),
        )
        connection.executemany(
            "INSERT INTO chunks(id,paper_id,page_id,chunk_index,text,section,heading) "
            "VALUES(?,?,?,?,?,?,?)",
            [
                (1, 1, 1, 0, "The primary outcome was all-cause mortality at 12 months.", "Results", "Results"),
                (2, 1, 1, 1, "The trial enrolled 1200 patients with heart failure.", "Methods", "Methods"),
                (3, 1, 1, 2, "心力衰竭患者接受标准治疗后随访一年。", "结果", "结果"),
                (4, 1, 1, 3, "MI was defined as myocardial infarction on admission.", "Methods", "Methods"),
            ],
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_keyword_search_english_and_chinese(self) -> None:
        english = retrieval.keyword_search("primary outcome", self.db)
        self.assertTrue(any("primary outcome" in row["text"].lower() for row in english))
        chinese = retrieval.keyword_search("心力衰竭", self.db)
        self.assertTrue(any("心力衰竭" in row["text"] for row in chinese))

    def test_bilingual_expansion_finds_cross_language_chunks(self) -> None:
        self.assertIn("heart failure", medical_terms.expand_query("心衰"))
        self.assertIn("myocardial infarction", medical_terms.expand_query("MI"))
        # "or" in a normal English sentence must not expand to odds ratio.
        self.assertNotIn("odds ratio", medical_terms.expand_query("primary outcome"))
        chinese_hits = retrieval.keyword_search("心衰", self.db)
        self.assertTrue(any("heart failure" in row["text"].lower() for row in chinese_hits))
        abbreviation_hits = retrieval.keyword_search("MI", self.db)
        self.assertTrue(any("myocardial infarction" in row["text"].lower() for row in abbreviation_hits))

    def test_chinese_section_heading_detection(self) -> None:
        from backend import app as app_module
        self.assertTrue(app_module._is_heading({"text": "方法", "size": 10, "bold_ratio": 0.0}, 10.0))
        self.assertTrue(app_module._is_heading({"text": "结果", "size": 10, "bold_ratio": 0.0}, 10.0))
        self.assertFalse(app_module._is_heading({"text": "这是一段正文内容", "size": 10, "bold_ratio": 0.0}, 10.0))

    def test_rrf_fuses_both_sources(self) -> None:
        fake_vector = [{
            "id": 2, "paper_id": 1, "page_number": 1,
            "text": "The trial enrolled 1200 patients with heart failure.",
            "section": "Methods", "heading": "Methods", "_distance": 0.2,
        }]
        with mock.patch.object(retrieval, "vector_search", return_value=fake_vector):
            rows = retrieval.hybrid_search("primary outcome", self.db, self.vectors,
                                           paper_id=1, top_k=5, enable_vectors=True)
        self.assertTrue(rows)
        by_id = {row["id"]: row for row in rows}
        self.assertIn(1, by_id)
        self.assertIn(2, by_id)
        self.assertIn("vector", by_id[2]["sources"])
        self.assertIn("keyword", by_id[1]["sources"])

    def _runner(self, responses: list[str], mode: str = "paper") -> agent.AgentRunner:
        calls = iter(responses)

        def fake_chat(messages):
            try:
                return next(calls)
            except StopIteration as error:
                raise AssertionError("agent made more model calls than expected") from error

        config = agent.AgentConfig(provider="deepseek", key="test", model="test-model",
                                   mode=mode, paper_id=1 if mode == "paper" else None)
        return agent.AgentRunner(config, self.db, self.vectors, chat_fn=fake_chat,
                                 enable_vectors=False)

    def test_agent_searches_then_finishes_with_verified_quote(self) -> None:
        search = json.dumps({
            "thought": "先检索主要终点",
            "tool": "search_library",
            "args": {"query": "primary outcome", "top_k": 3},
        }, ensure_ascii=False)
        finish = json.dumps({
            "thought": "证据足够",
            "tool": "finish",
            "args": {
                "answer": "主要终点是 12 个月全因死亡 [Page 1]。",
                "citations": [{
                    "paper_id": 1, "page": 1,
                    "quote": "The primary outcome was all-cause mortality at 12 months.",
                    "claim": "主要终点为 12 个月全因死亡",
                }],
            },
        }, ensure_ascii=False)
        result = self._runner([search, finish]).run("主要终点是什么？")
        self.assertTrue(result["verified"])
        self.assertIn("12 个月", result["content"])
        self.assertIn("医学学习", result["content"])  # disclaimer
        self.assertEqual([step["tool"] for step in result["trace"]], ["search_library", "finish"])
        self.assertEqual(result["citations"][0]["page"], 1)

    def test_agent_repairs_invalid_citation_once(self) -> None:
        search = json.dumps({
            "thought": "先检索",
            "tool": "search_library",
            "args": {"query": "primary outcome", "top_k": 3},
        }, ensure_ascii=False)
        bad_finish = json.dumps({
            "thought": "回答",
            "tool": "finish",
            "args": {"answer": "主要终点是死亡 [Page 1].",
                     "citations": [{"paper_id": 1, "page": 1, "quote": "编造的引文"}]},
        }, ensure_ascii=False)
        good_finish = json.dumps({
            "thought": "修正引用",
            "tool": "finish",
            "args": {"answer": "主要终点是 12 个月全因死亡 [Page 1].",
                     "citations": [{"paper_id": 1, "page": 1,
                                    "quote": "The primary outcome was all-cause mortality at 12 months.",
                                    "claim": "主要终点"}]},
        }, ensure_ascii=False)
        result = self._runner([search, bad_finish, good_finish]).run("主要终点是什么？")
        self.assertTrue(result["verified"])

    def test_agent_retries_empty_finish_answer(self) -> None:
        search = json.dumps({
            "thought": "先检索",
            "tool": "search_library",
            "args": {"query": "primary outcome", "top_k": 3},
        }, ensure_ascii=False)
        empty_finish = json.dumps({
            "thought": "准备回答",
            "tool": "finish",
            "args": {},
        }, ensure_ascii=False)
        good_finish = json.dumps({
            "thought": "补全回答",
            "tool": "finish",
            "args": {
                "answer": "主要终点是 12 个月全因死亡 [Page 1]。",
                "citations": [{
                    "paper_id": 1, "page": 1,
                    "quote": "The primary outcome was all-cause mortality at 12 months.",
                    "claim": "主要终点为 12 个月全因死亡",
                }],
            },
        }, ensure_ascii=False)
        result = self._runner([search, empty_finish, good_finish]).run("主要终点是什么？")
        self.assertTrue(result["verified"])
        self.assertIn("12 个月", result["content"])

    def test_agent_falls_back_when_no_citation(self) -> None:
        search = json.dumps({
            "thought": "先检索",
            "tool": "search_library",
            "args": {"query": "primary outcome", "top_k": 3},
        }, ensure_ascii=False)
        finish = json.dumps({
            "thought": "直接回答",
            "tool": "finish",
            "args": {"answer": "主要终点是死亡。", "citations": []},
        }, ensure_ascii=False)
        result = self._runner([search, finish]).run("主要终点是什么？")
        self.assertIn("未能生成可核验", result["content"])
        self.assertTrue(result["citations"])

    def test_citation_quote_validation(self) -> None:
        evidence = [{"paper_id": 1, "page_number": 1,
                     "text": "The primary outcome was all-cause mortality at 12 months."}]
        good = citations.validate_citations(
            "结论 [Page 1]",
            [{"paper_id": 1, "page": 1,
              "quote": "all-cause mortality at 12 months"}],
            evidence, default_paper_id=1, require_citation=True,
        )
        self.assertTrue(good.ok)
        self.assertEqual(good.level, "quote")

        bad = citations.validate_citations(
            "结论 [Page 1]",
            [{"paper_id": 1, "page": 1, "quote": "not in the paper"}],
            evidence, default_paper_id=1, require_citation=True,
        )
        self.assertFalse(bad.ok)
        self.assertIn("引文未在证据原文中找到", bad.invalid[0]["reason"])

        missing = citations.validate_citations(
            "结论 [Page 9]", [], evidence, default_paper_id=1, require_citation=True,
        )
        self.assertFalse(missing.ok)
        self.assertEqual(missing.missing_pages, [9])

    def test_emergency_safety_preface(self) -> None:
        answer, assessment = safety.apply_safety("可能是心梗。", "胸痛伴呼吸困难怎么办？")
        self.assertTrue(assessment.emergency)
        self.assertIn("120", answer)
        self.assertIn(safety.DISCLAIMER, answer)

    def test_structured_llm_evidence_is_quote_validated_and_cached(self) -> None:
        raw = {"fields": {
            "one_line_summary": {
                "value": "该试验评估心衰患者的主要终点。",
                "evidence": [{"paper_id": 1, "page_number": 1,
                              "quote": "The primary outcome was all-cause mortality at 12 months."}],
            },
            "study_design": {
                "value": "随机对照试验",
                "evidence": [{"paper_id": 1, "page_number": 1, "quote": "这不是原文"}],
            },
        }}
        with mock.patch.object(structured.providers, "chat_json", return_value=raw) as chat_json:
            first = structured.extract_fields(1, self.db, "deepseek", "test-key", "test-model")
            second = structured.extract_fields(1, self.db, "deepseek", "test-key", "test-model")
        self.assertEqual(first["source"], "llm")
        self.assertTrue(first["fields"]["one_line_summary"]["verified"])
        self.assertFalse(first["fields"]["study_design"]["verified"])
        self.assertEqual(first["fields"]["study_design"]["evidence"], [])
        self.assertTrue(second["cached"])
        self.assertEqual(chat_json.call_count, 1)

    def test_structured_heuristic_without_key(self) -> None:
        result = structured.extract_fields(1, self.db, "deepseek", "", "test-model")
        self.assertEqual(result["source"], "heuristic")
        self.assertIn("研究设计", {field["label"] for field in result["fields"].values()})

    def test_safe_calculator(self) -> None:
        self.assertAlmostEqual(agent._safe_calculate("1 / abs(0.20 - 0.10)"), 10.0)
        with self.assertRaises(ValueError):
            agent._safe_calculate("__import__('os').system('echo bad')")


if __name__ == "__main__":
    unittest.main()
