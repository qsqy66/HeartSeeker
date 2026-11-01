from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from backend import ingestion, storage  # noqa: E402


class IngestionTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.db = self.home / "database" / "heartseeker_med.sqlite3"
        self.vectors = self.home / "vectors"
        storage.initialize_storage(app_home=self.home, db_path=self.db, vectors_dir=self.vectors)
        with storage.get_connection(self.db) as connection:
            cursor = connection.execute(
                "INSERT INTO papers(title,file_name,stored_path,sha256,page_count,status,imported_at) "
                "VALUES('Test Paper','a.pdf','a.pdf','sha-ingest',1,'processing',?)",
                (storage.utc_now(),),
            )
            self.paper_id = int(cursor.lastrowid)
            connection.commit()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _pdf(self, text: str = "Abstract\nHeart failure therapy improves outcomes."):
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=11)
        return document

    def test_process_document_extracts_chunks_and_fts(self) -> None:
        progress: list[tuple[str, int, int]] = []
        document = self._pdf()
        try:
            result = ingestion.process_document(
                self.paper_id, document, self.db, self.vectors,
                progress=lambda stage, current, total: progress.append((stage, current, total)),
            )
        finally:
            document.close()
        self.assertEqual(result["pages"], 1)
        self.assertGreaterEqual(result["chunks"], 1)
        self.assertFalse(result["vectorized"])
        self.assertIn("ready", [stage for stage, _, _ in progress])
        self.assertIn("parsing", [stage for stage, _, _ in progress])

        with storage.get_connection(self.db) as connection:
            paper = connection.execute(
                "SELECT status, vectorized, progress_stage FROM papers WHERE id=?",
                (self.paper_id,),
            ).fetchone()
            chunks = connection.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE paper_id=?", (self.paper_id,)
            ).fetchone()["n"]
            fts = connection.execute(
                "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'heart'"
            ).fetchone()["n"]
        self.assertEqual(paper["status"], "ready")
        self.assertEqual(paper["progress_stage"], "ready")
        self.assertEqual(chunks, result["chunks"])
        self.assertGreaterEqual(fts, 1)

    def test_reimport_replaces_chunks_without_duplicating_fts(self) -> None:
        for _ in range(2):
            document = self._pdf()
            try:
                first = ingestion.process_document(self.paper_id, document, self.db, self.vectors)
            finally:
                document.close()
        with storage.get_connection(self.db) as connection:
            chunks = connection.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE paper_id=?", (self.paper_id,)
            ).fetchone()["n"]
            fts = connection.execute(
                "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH 'heart'"
            ).fetchone()["n"]
        self.assertEqual(chunks, first["chunks"])
        self.assertEqual(fts, first["chunks"])

    def test_blank_pdf_is_marked_needs_ocr(self) -> None:
        document = fitz.open()
        document.new_page()
        try:
            result = ingestion.process_document(self.paper_id, document, self.db, self.vectors)
        finally:
            document.close()
        self.assertTrue(result["needs_ocr"])

    def test_reindex_uses_existing_chunks(self) -> None:
        document = self._pdf()
        try:
            parsed = ingestion.process_document(self.paper_id, document, self.db, self.vectors)
        finally:
            document.close()

        def fake_embed(texts, key, model, on_progress=None):
            if on_progress:
                on_progress(len(texts), len(texts))
            return [[1.0, 0.0] for _ in texts]

        with mock.patch.object(ingestion, "embed_texts", side_effect=fake_embed), \
                mock.patch.object(ingestion, "upsert_chunks") as upsert, \
                mock.patch.object(ingestion, "_safe_delete_vectors"):
            result = ingestion.reindex_document(
                self.paper_id, self.db, self.vectors, "test-key", "text-embedding-v3"
            )
        self.assertTrue(result["vectorized"])
        self.assertEqual(result["embedding_dim"], 2)
        self.assertEqual(upsert.call_count, 1)
        self.assertEqual(len(upsert.call_args.args[1]), parsed["chunks"])

        with storage.get_connection(self.db) as connection:
            paper = connection.execute(
                "SELECT vectorized, embedding_model, embedding_dim FROM papers WHERE id=?",
                (self.paper_id,),
            ).fetchone()
        self.assertEqual(paper["vectorized"], 1)
        self.assertEqual(paper["embedding_model"], "text-embedding-v3")
        self.assertEqual(paper["embedding_dim"], 2)

    def test_storage_migrates_legacy_papers_table(self) -> None:
        legacy = self.home / "legacy.sqlite3"
        connection = sqlite3.connect(legacy)
        connection.executescript(
            "CREATE TABLE papers (id INTEGER PRIMARY KEY, title TEXT NOT NULL, file_name TEXT NOT NULL, "
            "stored_path TEXT NOT NULL, sha256 TEXT NOT NULL, page_count INTEGER NOT NULL, "
            "status TEXT NOT NULL, imported_at TEXT NOT NULL);"
        )
        connection.commit()
        connection.close()
        storage.initialize_storage(
            app_home=self.home, db_path=legacy, vectors_dir=self.home / "legacy-vectors"
        )
        with storage.get_connection(legacy) as migrated:
            columns = {row["name"] for row in migrated.execute("PRAGMA table_info(papers)")}
            mode = migrated.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertTrue({"progress_stage", "embedding_model", "metadata_json"} <= columns)
        self.assertEqual(str(mode).lower(), "wal")


if __name__ == "__main__":
    unittest.main()
