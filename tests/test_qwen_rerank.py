from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import rerank, retrieval  # noqa: E402


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


ROWS = [
    {"id": 1, "text": "量子计算是计算科学的一个前沿领域"},
    {"id": 2, "text": "文本排序模型广泛用于搜索引擎和推荐系统中"},
    {"id": 3, "text": "预训练语言模型的发展给文本排序模型带来了新的进展"},
]


class QwenRerankerTestCase(unittest.TestCase):
    def test_payload_and_score_ordering(self) -> None:
        payload = {"output": {"results": [
            {"index": 0, "relevance_score": 0.05},
            {"index": 1, "relevance_score": 0.42},
            {"index": 2, "relevance_score": 0.93},
        ]}}
        window = rerank.QwenReranker(
            "qwen3.7-text-rerank",
            base_url="https://example.test/rerank",
            api_key="sk-test",
        )
        with mock.patch.object(rerank.urllib.request, "urlopen",
                               return_value=FakeResponse(payload)) as urlopen:
            ranked = window.rerank("什么是文本排序模型", ROWS, 2)

        self.assertEqual([row["id"] for row in ranked], [3, 2])
        self.assertEqual(ranked[0]["rerank_score"], 0.93)

        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, "https://example.test/rerank")
        self.assertEqual(request.get_header("Authorization"), "Bearer sk-test")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3.7-text-rerank")
        self.assertEqual(body["input"]["query"], "什么是文本排序模型")
        self.assertEqual(body["input"]["documents"], [row["text"] for row in ROWS])
        self.assertEqual(body["parameters"]["top_n"], 3)
        self.assertFalse(body["parameters"]["return_documents"])

    def test_flat_results_and_omitted_rows(self) -> None:
        payload = {"results": [{"index": 1, "score": 0.8}]}
        window = rerank.QwenReranker(api_key="sk-test")
        with mock.patch.object(rerank.urllib.request, "urlopen",
                               return_value=FakeResponse(payload)):
            ranked = window.rerank("query", ROWS, 3)
        self.assertEqual([row["id"] for row in ranked], [2, 1, 3])

    def test_http_error_falls_back_to_lexical(self) -> None:
        error = urllib.error.HTTPError("https://example.test", 401, "unauthorized", {}, None)
        with mock.patch.object(rerank.urllib.request, "urlopen", side_effect=error):
            ranked = rerank.rerank("text rerank", ROWS, 3, engine="qwen",
                                   model="qwen3.7-text-rerank",
                                   base_url="https://example.test", api_key="bad")
        self.assertEqual(len(ranked), 3)
        self.assertTrue(all("rerank_score" in row for row in ranked))

    def test_missing_key_raises_and_degrades(self) -> None:
        window = rerank.QwenReranker(api_key="")
        with mock.patch("backend.credentials.safe_key", return_value=None), \
                mock.patch.dict("os.environ", {"DASHSCOPE_API_KEY": ""}):
            with self.assertRaises(rerank.RerankError):
                window.rerank("query", ROWS, 2)

    def test_factory_recognises_qwen_and_defaults(self) -> None:
        window = rerank.get_reranker("qwen", "", base_url="", api_key="k")
        self.assertIsInstance(window, rerank.QwenReranker)
        self.assertEqual(window.model_name, rerank.DEFAULT_QWEN_RERANK_MODEL)
        self.assertEqual(window.base_url, rerank.DEFAULT_QWEN_RERANK_URL)
        self.assertIsInstance(
            rerank.get_reranker("auto", "qwen3.7-text-rerank", api_key="k"),
            rerank.QwenReranker,
        )


class RerankSettingsTestCase(unittest.TestCase):
    def test_retrieval_options_read_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.sqlite3"
            connection = sqlite3.connect(db)
            connection.execute(
                "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                " updated_at TEXT NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO settings(key,value,updated_at) VALUES(?,?,?)",
                [("rerank_engine", "qwen", "now"),
                 ("rerank_model", "qwen3.7-text-rerank", "now"),
                 ("rerank_base_url", "https://example.test/rerank", "now")],
            )
            connection.commit()
            connection.close()
            options = retrieval._retrieval_options(db)
        self.assertEqual(options["rerank_engine"], "qwen")
        self.assertEqual(options["rerank_base_url"], "https://example.test/rerank")


if __name__ == "__main__":
    unittest.main()
