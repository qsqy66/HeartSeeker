from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module, storage  # noqa: E402


class ChatSessionTestCase(unittest.TestCase):
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

    def _create_session(self) -> int:
        response = self.client.post("/api/chat/sessions")
        self.assertEqual(response.status_code, 200)
        return int(response.json()["id"])

    def test_new_sessions_keep_separate_recent_memory(self) -> None:
        first = self._create_session()
        second = self._create_session()

        app_module._store_agent_message(
            "general", None, "user", "第一个会话的问题", "test", first)
        app_module._store_agent_message(
            "general", None, "assistant", "第一个会话的回答", "test", first)

        self.assertEqual(
            app_module._agent_history("general", None, False, "x", first),
            [{"role": "user", "content": "第一个会话的问题"},
             {"role": "assistant", "content": "第一个会话的回答"}],
        )
        self.assertEqual(app_module._agent_history("general", None, False, "x", second), [])

        first_history = self.client.get(f"/api/chat?session_id={first}").json()
        second_history = self.client.get(f"/api/chat?session_id={second}").json()
        self.assertEqual(len(first_history), 2)
        self.assertEqual(second_history, [])

    def test_session_title_comes_from_first_user_message(self) -> None:
        session_id = self._create_session()
        app_module._store_agent_message(
            "general", None, "user", "急性心衰的治疗原则？", "test", session_id)
        app_module._store_agent_message(
            "general", None, "user", "那慢性心衰呢？", "test", session_id)

        sessions = self.client.get("/api/chat/sessions").json()
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["id"], session_id)
        self.assertEqual(sessions[0]["title"], "急性心衰的治疗原则？")
        self.assertEqual(sessions[0]["message_count"], 2)

    def test_latest_session_is_the_default_and_unknown_ids_are_404(self) -> None:
        first = self._create_session()
        app_module._store_agent_message(
            "general", None, "user", "旧会话", "test", first)
        second = self._create_session()
        app_module._store_agent_message(
            "general", None, "user", "新会话", "test", second)

        self.assertEqual(app_module._resolve_chat_session(), second)
        self.assertEqual(self.client.get("/api/chat").json()[0]["content"], "新会话")
        self.assertEqual(
            self.client.get("/api/chat?session_id=999999").status_code, 404)
        with self.assertRaises(HTTPException):
            app_module._resolve_chat_session(999999)


class LegacyGeneralHistoryMigrationTestCase(unittest.TestCase):
    def test_preexisting_messages_join_a_legacy_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db = home / "test.sqlite3"
            connection = sqlite3.connect(db)
            connection.execute(
                "CREATE TABLE general_messages (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " role TEXT NOT NULL, content TEXT NOT NULL, provider TEXT,"
                " created_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO general_messages(role,content,provider,created_at) "
                "VALUES('user','旧问题','deepseek','2026-01-01T00:00:00')"
            )
            connection.execute(
                "INSERT INTO general_messages(role,content,provider,created_at) "
                "VALUES('assistant','旧回答','deepseek','2026-01-01T00:01:00')"
            )
            connection.commit()
            connection.close()

            storage.initialize_storage(
                app_home=home, db_path=db,
                papers_dir=home / "papers", vectors_dir=home / "vectors",
            )

            connection = sqlite3.connect(db)
            connection.row_factory = sqlite3.Row
            sessions = connection.execute(
                "SELECT * FROM general_sessions").fetchall()
            messages = connection.execute(
                "SELECT session_id, content FROM general_messages ORDER BY id").fetchall()
            connection.close()

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["title"], "历史对话")
            self.assertEqual({row["session_id"] for row in messages}, {sessions[0]["id"]})
            self.assertEqual(len(messages), 2)


if __name__ == "__main__":
    unittest.main()
