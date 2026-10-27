from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from backend import ingestion, ocr, storage  # noqa: E402


class FakeOcrEngine:
    name = "fake"

    def available(self) -> bool:
        return True

    def ocr_image(self, image_path: Path, lang: str) -> str:
        assert Path(image_path).exists()
        assert lang == "chi_sim+eng"
        return "摘要\n扫描版论文的心力衰竭主要结局。\n\nMethods\nWe enrolled 120 patients."


class UnavailableEngine(FakeOcrEngine):
    name = "unavailable"

    def available(self) -> bool:
        return False


class OcrTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.db = self.home / "database" / "paper_reader.sqlite3"
        self.vectors = self.home / "vectors"
        storage.initialize_storage(app_home=self.home, db_path=self.db, vectors_dir=self.vectors)
        with storage.get_connection(self.db) as connection:
            cursor = connection.execute(
                "INSERT INTO papers(title,file_name,stored_path,sha256,page_count,status,imported_at) "
                "VALUES('Scanned','a.pdf','a.pdf','sha-ocr',1,'processing',?)",
                (storage.utc_now(),),
            )
            self.paper_id = int(cursor.lastrowid)
            connection.commit()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _blank_pdf(self):
        document = fitz.open()
        document.new_page()
        return document

    def test_scanned_page_uses_local_ocr_engine(self) -> None:
        progress: list[tuple[str, int, int]] = []
        document = self._blank_pdf()
        try:
            result = ingestion.process_document(
                self.paper_id, document, self.db, self.vectors,
                progress=lambda stage, current, total: progress.append((stage, current, total)),
                ocr_engine=FakeOcrEngine(),
            )
        finally:
            document.close()
        self.assertEqual(result["ocr_pages"], 1)
        self.assertEqual(result["ocr_engine"], "fake")
        self.assertFalse(result["needs_ocr"])
        self.assertGreater(result["chunks"], 0)
        self.assertIn("ocr", [stage for stage, _, _ in progress])
        with storage.get_connection(self.db) as connection:
            page = connection.execute(
                "SELECT text FROM pages WHERE paper_id=?", (self.paper_id,)
            ).fetchone()
            chunks = connection.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE paper_id=?", (self.paper_id,)
            ).fetchone()["n"]
        self.assertIn("心力衰竭", page["text"])
        self.assertEqual(chunks, result["chunks"])

    def test_missing_engine_keeps_page_marked_needs_ocr(self) -> None:
        document = self._blank_pdf()
        try:
            result = ingestion.process_document(
                self.paper_id, document, self.db, self.vectors,
                ocr_engine=UnavailableEngine(),
            )
        finally:
            document.close()
        self.assertTrue(result["needs_ocr"])
        self.assertEqual(result["ocr_pages"], 0)

    def test_engine_registry(self) -> None:
        self.assertEqual(ocr.get_engine("none").name, "none")
        self.assertEqual(ocr.get_engine("tesseract").name, "tesseract")
        with self.assertRaises(ocr.OcrError):
            ocr.get_engine("made-up")

    def test_ocr_chunks_detect_headings(self) -> None:
        chunks = ingestion._ocr_chunks("摘要\n第一段。\n\nResults\nPrimary outcome was death.", 600, 800)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section"], "摘要")
        self.assertTrue(any(chunk["section"] == "Results" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
