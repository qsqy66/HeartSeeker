from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module, observability, storage  # noqa: E402


class _TempDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.db = self.home / "test.sqlite3"
        storage.initialize_storage(
            app_home=self.home,
            db_path=self.db,
            papers_dir=self.home / "papers",
            vectors_dir=self.home / "vectors",
        )
        self.get_connection = mock.patch.object(
            app_module, "get_connection",
            lambda: storage.get_connection(self.db),
        )
        self.get_connection.start()
        self.client = TestClient(app_module.app)

    def tearDown(self) -> None:
        self.get_connection.stop()
        self.tmp.cleanup()


class UsageStatsApiTestCase(_TempDatabase):
    def _insert_usage(self, prompt: int, completion: int,
                      estimated: int = 0, provider: str = "qwen",
                      model: str = "qwen-plus") -> None:
        with storage.get_connection(self.db) as connection:
            connection.execute(
                "INSERT INTO llm_usage(provider,model,prompt_tokens,completion_tokens,"
                "total_tokens,estimated,kind,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (provider, model, prompt, completion, prompt + completion,
                 estimated, "stream", storage.utc_now()),
            )

    def _insert_run(self, mode: str, verified: int, tools: list[str]) -> None:
        with storage.get_connection(self.db) as connection:
            cursor = connection.execute(
                "INSERT INTO agent_runs(paper_id,mode,question,answer,provider,model,"
                "verified,created_at) VALUES(NULL,?,'q','a','qwen','m',?,?)",
                (mode, verified, storage.utc_now()),
            )
            for index, tool in enumerate(tools, start=1):
                connection.execute(
                    "INSERT INTO agent_steps(run_id,step_index,tool,created_at)"
                    " VALUES(?,?,?,?)",
                    (cursor.lastrowid, index, tool, storage.utc_now()),
                )

    def test_stats_aggregate_tokens_runs_and_tools(self) -> None:
        self._insert_usage(100, 50)
        self._insert_usage(20, 10, estimated=1)
        self._insert_run("library", 1, ["search_library", "read_page", "finish"])
        self._insert_run("general", 0, ["finish"])

        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        stats = response.json()

        self.assertEqual(stats["tokens"]["calls"], 2)
        self.assertEqual(stats["tokens"]["prompt"], 120)
        self.assertEqual(stats["tokens"]["completion"], 60)
        self.assertEqual(stats["tokens"]["total"], 180)
        self.assertEqual(stats["tokens"]["estimated_calls"], 1)
        self.assertEqual(stats["tokens"]["today_calls"], 2)
        self.assertEqual(stats["runs"]["total"], 2)
        self.assertEqual(stats["runs"]["verified"], 1)
        self.assertEqual(stats["runs"]["avg_steps"], 2.0)
        self.assertEqual({item["mode"] for item in stats["runs"]["modes"]},
                         {"library", "general"})
        self.assertEqual(stats["tools"]["total"], 4)
        self.assertEqual(
            {item["tool"]: item["count"] for item in stats["tools"]["by_tool"]},
            {"search_library": 1, "read_page": 1, "finish": 2},
        )
        self.assertEqual(stats["models"][0]["total_tokens"], 180)

    def test_empty_database_returns_zeroed_stats(self) -> None:
        stats = self.client.get("/api/stats").json()
        self.assertEqual(stats["tokens"]["total"], 0)
        self.assertEqual(stats["runs"]["avg_steps"], 0)
        self.assertEqual(stats["tools"], {"total": 0, "by_tool": []})
        self.assertEqual(stats["papers"]["total"], 0)


class UsageRecorderIsolationTestCase(unittest.TestCase):
    def test_importing_the_app_does_not_touch_live_usage(self) -> None:
        # Recorder registration belongs to the server startup hook; plain
        # imports (tests, scripts, evaluation) must not write to the live DB.
        from backend import providers
        self.assertIsNone(providers._usage_recorder)


class StructuredLogApiTestCase(_TempDatabase):
    def test_events_are_structured_and_incremental(self) -> None:
        observability.setup_logging()
        observability.log_event("test.event", answer=42, tag="unit")

        events = self.client.get("/api/logs?after=0&limit=1000").json()
        matched = [event for event in events if event.get("event") == "test.event"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["answer"], 42)
        self.assertEqual(matched[0]["tag"], "unit")
        self.assertEqual(matched[0]["level"], "INFO")

        cursor = matched[0]["seq"]
        observability.log_event("test.next", ok=True)
        newer = self.client.get(f"/api/logs?after={cursor}").json()
        self.assertNotIn("test.event", [event.get("event") for event in newer])
        self.assertIn("test.next", [event.get("event") for event in newer])

    def test_logging_setup_is_idempotent(self) -> None:
        observability.setup_logging()
        observability.setup_logging()
        observability.log_event("test.once")
        events = self.client.get("/api/logs?after=0&limit=1000").json()
        self.assertEqual(
            len([event for event in events if event.get("event") == "test.once"]), 1)

    def test_record_usage_persists_and_publishes(self) -> None:
        with mock.patch.object(storage, "DB_PATH", self.db):
            observability.record_usage({
                "provider": "qwen", "model": "qwen-plus",
                "prompt_tokens": 12, "completion_tokens": 8,
                "total_tokens": 20, "estimated": False, "kind": "chat",
            })
        with storage.get_connection(self.db) as connection:
            row = connection.execute(
                "SELECT provider, model, total_tokens, estimated FROM llm_usage"
            ).fetchone()
        self.assertEqual(dict(row), {"provider": "qwen", "model": "qwen-plus",
                                     "total_tokens": 20, "estimated": 0})
        events = self.client.get("/api/logs?after=0&limit=1000").json()
        self.assertIn("llm.usage", [event.get("event") for event in events])


if __name__ == "__main__":
    unittest.main()
