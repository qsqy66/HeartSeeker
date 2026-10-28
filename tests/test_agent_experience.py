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

from fastapi.testclient import TestClient  # noqa: E402

from backend import agent, app as app_module, memory, token_budget  # noqa: E402

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
CREATE TABLE agent_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id INTEGER, mode TEXT NOT NULL,
  question TEXT NOT NULL, answer TEXT NOT NULL, provider TEXT, model TEXT,
  verified INTEGER NOT NULL DEFAULT 0, citation_level TEXT,
  trace_json TEXT NOT NULL DEFAULT '[]', citations_json TEXT NOT NULL DEFAULT '[]',
  warnings_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
);
CREATE TABLE agent_steps (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, step_index INTEGER NOT NULL,
  thought TEXT, tool TEXT, args_json TEXT, observation_json TEXT, duration_ms INTEGER,
  created_at TEXT NOT NULL
);
"""


class _TempDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.sqlite3"
        self.vectors = Path(self.tmp.name) / "vectors"
        self.vectors.mkdir()
        connection = sqlite3.connect(self.db)
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO papers(id,title,file_name,stored_path,sha256,page_count,status,"
            "imported_at,vectorized,needs_ocr) VALUES(1,'Heart Failure Trial','a.pdf',"
            "'a.pdf','sha-1',1,'ready','2026-01-01',0,0)"
        )
        connection.execute(
            "INSERT INTO pages(id,paper_id,page_number,text) VALUES(1,1,1,?)",
            ("The primary outcome was all-cause mortality at 12 months.",),
        )
        connection.execute(
            "INSERT INTO chunks(id,paper_id,page_id,chunk_index,text,section,heading) "
            "VALUES(1,1,1,0,?, 'Results','Results')",
            ("The primary outcome was all-cause mortality at 12 months.",),
        )
        connection.commit()
        connection.close()

    def tearDown(self) -> None:
        self.tmp.cleanup()


class AnswerStreamExtractorTestCase(unittest.TestCase):
    def test_non_finish_turns_emit_no_deltas(self) -> None:
        extractor = agent._AnswerStreamExtractor()
        turn = json.dumps({"thought": "检索", "tool": "search_library",
                           "args": {"query": "primary outcome"}}, ensure_ascii=False)
        self.assertEqual(extractor.feed(turn), [])

    def test_escapes_and_split_chunks(self) -> None:
        answer = '第一行\n第二行 "引用" 与 C:\\path'
        payload = json.dumps({"thought": "", "tool": "finish",
                              "args": {"answer": answer, "citations": []}},
                             ensure_ascii=False)
        extractor = agent._AnswerStreamExtractor()
        pieces: list[str] = []
        for index in range(0, len(payload), 3):
            pieces.extend(extractor.feed(payload[index:index + 3]))
        self.assertEqual("".join(pieces), answer)
        self.assertTrue(extractor.done)
        self.assertEqual(extractor.feed("trailing"), [])

    def test_unicode_escape_split_across_chunks(self) -> None:
        answer = "中文回答"
        payload = json.dumps({"tool": "finish", "args": {"answer": answer}},
                             ensure_ascii=True)
        self.assertIn("\\u", payload)
        extractor = agent._AnswerStreamExtractor()
        pieces: list[str] = []
        for char in payload:
            pieces.extend(extractor.feed(char))
        self.assertEqual("".join(pieces), answer)

    def test_incomplete_escape_waits_for_next_chunk(self) -> None:
        extractor = agent._AnswerStreamExtractor()
        first = '{"tool": "finish", "args": {"answer": "abc\\'
        self.assertEqual(extractor.feed(first), ["abc"])
        self.assertEqual(extractor.feed('n"}, "citations": []}'), ["\n"])


class StreamingAgentTestCase(_TempDatabase):
    def _runner(self, chunks_per_call: list[list[str]]) -> agent.AgentRunner:
        calls = iter(chunks_per_call)

        def fake_stream(messages):
            try:
                yield from next(calls)
            except StopIteration as error:
                raise AssertionError("more model calls than expected") from error

        config = agent.AgentConfig(provider="deepseek", key="test", model="test-model",
                                   mode="paper", paper_id=1)
        return agent.AgentRunner(config, self.db, self.vectors, stream_fn=fake_stream,
                                 enable_vectors=False)

    def test_run_iter_emits_answer_deltas_before_done(self) -> None:
        search = json.dumps({
            "thought": "先检索",
            "tool": "search_library",
            "args": {"query": "primary outcome", "top_k": 3},
        }, ensure_ascii=False)
        answer = "主要终点是 12 个月全因死亡 [Page 1]。"
        finish = json.dumps({
            "thought": "证据足够",
            "tool": "finish",
            "args": {"answer": answer, "citations": [{
                "paper_id": 1, "page": 1,
                "quote": "The primary outcome was all-cause mortality at 12 months.",
                "claim": "主要终点",
            }]},
        }, ensure_ascii=False)
        search_chunks = [search[index:index + 40] for index in range(0, len(search), 40)]
        finish_chunks = [finish[index:index + 11] for index in range(0, len(finish), 11)]
        events = list(self._runner([search_chunks, finish_chunks]).run_iter("主要终点是什么？"))

        deltas = [event["text"] for event in events if event["type"] == "delta"]
        self.assertEqual("".join(deltas), answer)
        delta_index = next(i for i, event in enumerate(events) if event["type"] == "delta")
        done_index = next(i for i, event in enumerate(events) if event["type"] == "done")
        self.assertLess(delta_index, done_index)
        self.assertTrue(events[done_index]["verified"])


class MemoryTestCase(_TempDatabase):
    def test_remember_recall_and_forget(self) -> None:
        record = memory.remember(self.db, "患者对青霉素过敏，需在用药前确认。",
                                 tags="过敏, 用药安全")
        self.assertEqual(record["tags"], ["过敏", "用药安全"])

        recent = memory.list_memories(self.db)
        self.assertEqual(recent[0]["text"], record["text"])

        hits = memory.recall(self.db, "青霉素过敏")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["id"], record["id"])
        self.assertGreater(hits[0]["score"], 0)

        self.assertTrue(memory.forget(self.db, record["id"]))
        self.assertIsNone(memory.get(self.db, record["id"]))
        self.assertFalse(memory.forget(self.db, record["id"]))

    def test_empty_and_oversized_notes_are_rejected(self) -> None:
        with self.assertRaises(memory.MemoryError):
            memory.remember(self.db, "   ")
        with self.assertRaises(memory.MemoryError):
            memory.remember(self.db, "x" * (memory.MAX_TEXT + 1))


class MemoryApiTestCase(_TempDatabase):
    def test_memory_crud_endpoints(self) -> None:
        client = TestClient(app_module.app)
        with mock.patch.object(app_module, "DB_PATH", self.db):
            created = client.post("/api/memory",
                                  json={"text": "患者肌酐偏高", "tags": ["肾功能"]})
            self.assertEqual(created.status_code, 200)
            memory_id = created.json()["id"]

            listed = client.get("/api/memory")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()[0]["id"], memory_id)

            invalid = client.post("/api/memory", json={"text": "   "})
            self.assertEqual(invalid.status_code, 422)

            deleted = client.delete(f"/api/memory/{memory_id}")
            self.assertEqual(deleted.status_code, 200)
            self.assertEqual(client.delete(f"/api/memory/{memory_id}").status_code, 404)


class TokenBudgetTestCase(unittest.TestCase):
    def test_estimate_is_cjk_aware(self) -> None:
        self.assertEqual(token_budget.estimate_tokens("心力衰竭"), 4)
        self.assertEqual(token_budget.estimate_tokens("abcdefgh"), 2)
        self.assertEqual(token_budget.estimate_tokens(""), 0)

    def test_fit_keeps_system_and_question_and_shrinks_context(self) -> None:
        messages = [
            {"role": "system", "content": "系统提示"},
            {"role": "user", "content": "原始问题：主要终点是什么？"},
        ]
        for index in range(30):
            messages.append({"role": "assistant",
                             "content": json.dumps({"tool": "search_library",
                                                    "args": {"query": "x"}}) * 20})
            messages.append({"role": "user", "content": f"Observation {index}: " + "证据" * 300})
        fitted = token_budget.fit_messages(messages, max_tokens=4000, keep_last=2)
        self.assertEqual(fitted[0], messages[0])
        self.assertEqual(fitted[1], messages[1])
        self.assertLess(len(fitted), len(messages))
        self.assertLess(token_budget.estimate_messages(fitted),
                        token_budget.estimate_messages(messages))
        self.assertLessEqual(token_budget.estimate_messages(fitted), 4000)

    def test_non_positive_budget_returns_copy(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        fitted = token_budget.fit_messages(messages, max_tokens=0)
        self.assertEqual(fitted, messages)
        self.assertIsNot(fitted[0], messages[0])


class AgentToolsTestCase(_TempDatabase):
    def _runner(self, responses: list[str]) -> agent.AgentRunner:
        calls = iter(responses)

        def fake_chat(messages):
            try:
                return next(calls)
            except StopIteration as error:
                raise AssertionError("more model calls than expected") from error

        config = agent.AgentConfig(provider="deepseek", key="test", model="test-model",
                                   mode="paper", paper_id=1)
        return agent.AgentRunner(config, self.db, self.vectors, chat_fn=fake_chat,
                                 enable_vectors=False)

    def test_memory_tools_and_evidence_table(self) -> None:
        responses = [
            json.dumps({"thought": "检索", "tool": "search_library",
                        "args": {"query": "primary outcome", "top_k": 3}},
                       ensure_ascii=False),
            json.dumps({"thought": "记住", "tool": "remember_note",
                        "args": {"text": "主要终点是 12 个月全因死亡", "tags": ["主要终点"]}},
                       ensure_ascii=False),
            json.dumps({"thought": "回忆", "tool": "recall_notes",
                        "args": {"query": "主要终点"}}, ensure_ascii=False),
            json.dumps({"thought": "整理证据", "tool": "evidence_table", "args": {}},
                       ensure_ascii=False),
            json.dumps({"thought": "完成", "tool": "finish", "args": {
                "answer": "主要终点是 12 个月全因死亡 [Page 1]。",
                "citations": [{"paper_id": 1, "page": 1,
                               "quote": "The primary outcome was all-cause mortality at 12 months.",
                               "claim": "主要终点"}]}}, ensure_ascii=False),
        ]
        result = self._runner(responses).run("主要终点是什么？")
        tools = [step["tool"] for step in result["trace"]]
        self.assertEqual(tools, ["search_library", "remember_note", "recall_notes",
                                 "evidence_table", "finish"])

        steps = {step["tool"]: step for step in result["trace"]}
        self.assertIn("memory_id", steps["remember_note"]["observation"])
        self.assertGreaterEqual(steps["recall_notes"]["observation"]["memories"], 1)
        self.assertGreaterEqual(steps["evidence_table"]["observation"]["rows"], 1)
        self.assertEqual(len(memory.list_memories(self.db)), 1)
        self.assertTrue(result["verified"])

    def test_forget_note_tool(self) -> None:
        record = memory.remember(self.db, "临时笔记")
        responses = [
            json.dumps({"thought": "检索", "tool": "search_library",
                        "args": {"query": "primary outcome", "top_k": 3}},
                       ensure_ascii=False),
            json.dumps({"thought": "删除", "tool": "forget_note",
                        "args": {"id": record["id"]}}, ensure_ascii=False),
            json.dumps({"thought": "完成", "tool": "finish",
                        "args": {"answer": "已删除笔记 [Page 1]。",
                                 "citations": [{"paper_id": 1, "page": 1,
                                                "quote": "The primary outcome was all-cause mortality at 12 months.",
                                                "claim": "删除确认"}]}},
                       ensure_ascii=False),
        ]
        result = self._runner(responses).run("忘记笔记")
        forget_step = next(step for step in result["trace"] if step["tool"] == "forget_note")
        self.assertEqual(forget_step["observation"]["forgotten"], record["id"])
        self.assertIsNone(memory.get(self.db, record["id"]))


class ExportApiTestCase(_TempDatabase):
    def setUp(self) -> None:
        super().setUp()
        connection = sqlite3.connect(self.db)
        connection.row_factory = sqlite3.Row
        connection.execute(
            "INSERT INTO agent_runs(id,paper_id,mode,question,answer,provider,model,verified,"
            "citation_level,trace_json,citations_json,warnings_json,created_at) "
            "VALUES(1,1,'paper','主要终点？','主要终点是死亡。','deepseek','m',1,'claim',?,?,?,?)",
            (
                json.dumps([{"index": 1, "tool": "search_library"}]),
                json.dumps([{"paper_id": 1, "page": 1, "claim": "含|竖线",
                             "quote": "quote line\nnext", "verified": True,
                             "claim_supported": True, "support_score": 0.9}]),
                json.dumps(["需要注意"]),
                "2026-01-01T00:00:00",
            ),
        )
        connection.execute(
            "INSERT INTO agent_steps(run_id,step_index,thought,tool,args_json,"
            "observation_json,duration_ms,created_at) VALUES(1,1,'检索','search_library',?,?,5,?)",
            (json.dumps({"query": "primary"}), json.dumps({"ok": True}),
             "2026-01-01T00:00:00"),
        )
        connection.commit()
        connection.close()
        self.client = TestClient(app_module.app)

    def _get(self, path: str):
        def connect():
            connection = sqlite3.connect(self.db)
            connection.row_factory = sqlite3.Row
            return connection

        with mock.patch.object(app_module, "get_connection", connect):
            return self.client.get(path)

    def test_markdown_export(self) -> None:
        response = self._get("/api/agent/runs/1/export?format=md")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/markdown", response.headers["content-type"])
        self.assertIn("attachment; filename=\"paper-reader-run-1.md\"",
                      response.headers["content-disposition"])
        self.assertIn("## 回答", response.text)
        self.assertIn("含\\|竖线", response.text)
        self.assertIn("## 执行轨迹", response.text)

    def test_csv_export_has_bom_and_citation_rows(self) -> None:
        response = self._get("/api/agent/runs/1/export?format=csv")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.text.startswith("\ufeff"))
        self.assertIn("claim_supported", response.text)
        self.assertIn("含|竖线", response.text)

    def test_json_export_parses_steps(self) -> None:
        response = self._get("/api/agent/runs/1/export?format=json")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["run"]["question"], "主要终点？")
        self.assertEqual(payload["steps"][0]["args_json"], {"query": "primary"})
        self.assertEqual(payload["run"]["citations_json"][0]["page"], 1)

    def test_invalid_format_and_missing_run(self) -> None:
        self.assertEqual(self._get("/api/agent/runs/1/export?format=xml").status_code, 400)
        self.assertEqual(self._get("/api/agent/runs/99/export?format=md").status_code, 404)


if __name__ == "__main__":
    unittest.main()
