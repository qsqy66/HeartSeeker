from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from starlette.datastructures import UploadFile  # noqa: E402

from backend import app as app_module, storage  # noqa: E402


def _pdf_bytes(text: str = "Batch import test") -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def _upload(name: str, data: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(data))


class BatchImportTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        home = Path(self.tmp.name)
        self.db = home / "database.sqlite3"
        self.papers = home / "papers"
        self.vectors = home / "vectors"
        storage.initialize_storage(
            app_home=home, db_path=self.db,
            papers_dir=self.papers, vectors_dir=self.vectors,
        )
        self.enqueued: list[tuple[int, Path]] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _enqueue(self, paper_id: int, path: Path) -> None:
        self.enqueued.append((paper_id, path))

    def _count(self) -> int:
        with storage.get_connection(self.db) as connection:
            return int(connection.execute("SELECT COUNT(*) AS n FROM papers").fetchone()["n"])

    async def test_import_one_persists_file_and_enqueues(self) -> None:
        result = await app_module._import_one(
            _upload("paper.pdf", _pdf_bytes()), self.db, self.papers, self._enqueue)
        self.assertEqual(result["status"], "processing")
        self.assertEqual(result["page_count"], 1)
        self.assertEqual(self._count(), 1)
        self.assertEqual(len(self.enqueued), 1)
        paper_id, stored_path = self.enqueued[0]
        self.assertEqual(paper_id, result["id"])
        self.assertTrue(stored_path.exists())
        self.assertEqual(stored_path.parent, self.papers)

    async def test_duplicate_pdf_is_rejected_without_extra_row(self) -> None:
        data = _pdf_bytes("Duplicate")
        await app_module._import_one(
            _upload("first.pdf", data), self.db, self.papers, self._enqueue)
        with self.assertRaises(HTTPException) as raised:
            await app_module._import_one(
                _upload("second.pdf", data), self.db, self.papers, self._enqueue)
        self.assertEqual(raised.exception.status_code, 409)
        self.assertEqual(self._count(), 1)
        self.assertEqual(len(self.enqueued), 1)

    async def test_non_pdf_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await app_module._import_one(
                _upload("notes.txt", b"not a pdf"), self.db, self.papers, self._enqueue)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(self._count(), 0)

    async def test_batch_continues_after_invalid_and_duplicate_files(self) -> None:
        data = _pdf_bytes("Batch")
        uploads = [
            _upload("a.pdf", data),
            _upload("a-copy.pdf", data),
            _upload("broken.pdf", b"not a pdf"),
            _upload("b.pdf", _pdf_bytes("Second paper")),
        ]
        with mock.patch.object(app_module, "DB_PATH", self.db), \
                mock.patch.object(app_module, "PAPERS_DIR", self.papers), \
                mock.patch.object(app_module, "_enqueue_import", self._enqueue):
            result = await app_module.import_papers(uploads)
        self.assertEqual(result["imported"], 2)
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual([item["status"] for item in result["results"]],
                         ["queued", "duplicate", "failed", "queued"])
        self.assertEqual(self._count(), 2)
        self.assertEqual(len(self.enqueued), 2)

    async def test_batch_rejects_too_many_files(self) -> None:
        uploads = [_upload(f"{index}.pdf", b"") for index in range(app_module.MAX_BATCH_FILES + 1)]
        with self.assertRaises(HTTPException) as raised:
            await app_module.import_papers(uploads)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(self._count(), 0)


if __name__ == "__main__":
    unittest.main()


class BatchImportRouteTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        home = Path(self.tmp.name)
        self.db = home / "database.sqlite3"
        self.papers = home / "papers"
        self.vectors = home / "vectors"
        storage.initialize_storage(
            app_home=home, db_path=self.db,
            papers_dir=self.papers, vectors_dir=self.vectors,
        )
        self.enqueued: list[tuple[int, Path]] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _enqueue(self, paper_id: int, path: Path) -> None:
        self.enqueued.append((paper_id, path))

    def test_multipart_batch_route_returns_per_file_status(self) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(app_module.app)
        with mock.patch.object(app_module, "DB_PATH", self.db), \
                mock.patch.object(app_module, "PAPERS_DIR", self.papers), \
                mock.patch.object(app_module, "_enqueue_import", self._enqueue):
            response = client.post("/api/papers/import/batch", files=[
                ("files", ("first.pdf", _pdf_bytes("First"), "application/pdf")),
                ("files", ("second.pdf", _pdf_bytes("Second"), "application/pdf")),
                ("files", ("notes.txt", b"not a pdf", "text/plain")),
            ])
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["imported"], 2)
        self.assertEqual(payload["failed"], 1)
        self.assertEqual([row["status"] for row in payload["results"]],
                         ["queued", "queued", "failed"])
        self.assertEqual(len(self.enqueued), 2)

    def test_single_import_route_still_accepts_one_file(self) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(app_module.app)
        with mock.patch.object(app_module, "DB_PATH", self.db), \
                mock.patch.object(app_module, "PAPERS_DIR", self.papers), \
                mock.patch.object(app_module, "_enqueue_import", self._enqueue):
            response = client.post(
                "/api/papers/import",
                files={"file": ("single.pdf", _pdf_bytes("Single"), "application/pdf")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "processing")
        self.assertEqual(len(self.enqueued), 1)
