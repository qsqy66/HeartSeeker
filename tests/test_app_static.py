from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module  # noqa: E402


class BuiltFrontendTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app_module.app)

    def test_health_reports_built_frontend(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["frontend"], {"built", "source"})

    def test_root_serves_built_index_when_available(self) -> None:
        if not app_module._frontend_built():
            self.skipTest("dist/ 尚未构建")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="root"', response.text)
        self.assertIn("text/html", response.headers["content-type"])

    def test_spa_fallback_serves_index_for_client_routes(self) -> None:
        if not app_module._frontend_built():
            self.skipTest("dist/ 尚未构建")
        response = self.client.get("/papers/42")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="root"', response.text)

    def test_unknown_api_is_404_not_spa(self) -> None:
        response = self.client.get("/api/definitely-not-a-route")
        self.assertEqual(response.status_code, 404)

    def test_dist_file_blocks_path_traversal(self) -> None:
        self.assertIsNone(app_module._dist_file("../README.md"))
        self.assertIsNone(app_module._dist_file("/etc/passwd"))


if __name__ == "__main__":
    unittest.main()
