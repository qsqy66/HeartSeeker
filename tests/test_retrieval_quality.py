from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import ingestion, rerank, retrieval, storage, tokenization  # noqa: E402


class TokenizationTestCase(unittest.TestCase):
    def test_fallback_segments_english_and_chinese(self) -> None:
        with mock.patch.object(tokenization, "_jieba", None):
            tokens = tokenization.tokenize("心力衰竭 Heart Failure 120 mg")
            segmented = tokenization.segment("心力衰竭")
        self.assertIn("heart", tokens)
        self.assertIn("failure", tokens)
        self.assertIn("120", tokens)
        self.assertIn("心力衰竭", segmented)
        self.assertIn("心力", segmented)

    def test_split_text_overlaps_adjacent_chunks(self) -> None:
        text = "这是第一句。" * 400
        chunks = ingestion._split_text(text, 0, 0, 600, 800, "Results", "Results",
                                       max_len=1200, overlap=200)
        self.assertGreater(len(chunks), 1)
        self.assertIn(chunks[1]["text"][:80], chunks[0]["text"])
        self.assertTrue(all(chunk["section"] == "Results" for chunk in chunks))


class RetrievalQualityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.db = self.home / "database" / "paper_reader.sqlite3"
        self.vectors = self.home / "vectors"
        storage.initialize_storage(app_home=self.home, db_path=self.db, vectors_dir=self.vectors)
        with storage.get_connection(self.db) as connection:
            paper_id = connection.execute(
                "INSERT INTO papers(title,file_name,stored_path,sha256,page_count,status,imported_at) "
                "VALUES('HF Trial','a.pdf','a.pdf','sha-rq',1,'ready',?)",
                (storage.utc_now(),),
            ).lastrowid
            self.paper_id = int(paper_id)
            page_id = connection.execute(
                "INSERT INTO pages(paper_id,page_number,text) VALUES(?,?,?)",
                (self.paper_id, 1, "first middle last"),
            ).lastrowid
            self.page_id = int(page_id)
            texts = ["first sentence about heart failure.",
                     "middle sentence with primary outcome.",
                     "last sentence about mortality.",
                     "心力衰竭患者的主要结局是心血管死亡。"]
            for index, text in enumerate(texts):
                connection.execute(
                    "INSERT INTO chunks(paper_id,page_id,chunk_index,text,search_text,section,heading) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (self.paper_id, self.page_id, index, text,
                     tokenization.segment(text), "Results", "Results"),
                )
            connection.commit()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_segmented_fts_ranks_chinese_without_like(self) -> None:
        with storage.get_connection(self.db) as connection:
            tokens = tokenization.tokenize("主要结局")
            rows = retrieval._seg_fts_rows(connection, tokens, self.paper_id, 10)
        self.assertTrue(rows)
        self.assertTrue(any("心力衰竭" in row["text"] for row in rows))

    def test_hybrid_search_attaches_neighbour_context(self) -> None:
        rows = retrieval.hybrid_search(
            "middle", self.db, self.vectors, paper_id=self.paper_id, top_k=5,
            enable_vectors=False, rerank_enabled=False, expand_neighbors=True,
        )
        hit = next(row for row in rows if "middle" in row["text"])
        self.assertIn("first sentence", hit["context_text"])
        self.assertIn("last sentence", hit["context_text"])
        self.assertEqual(len(hit["context_chunk_ids"]), 3)

    def test_lexical_reranker_prefers_term_coverage(self) -> None:
        rows = [
            {"id": 1, "score": 0.02, "text": "unrelated diabetes content", "heading": ""},
            {"id": 2, "score": 0.02, "text": "heart failure primary outcome mortality",
             "heading": "Results"},
        ]
        ranked = rerank.LexicalReranker().rerank("heart failure primary outcome", rows, 2)
        self.assertEqual(ranked[0]["id"], 2)
        self.assertGreater(ranked[0]["rerank_score"], ranked[1]["rerank_score"])

    def test_rrf_deduplicates_identical_text_across_sources(self) -> None:
        text = "The primary outcome was cardiovascular death in this trial."
        vector_rows = [{"id": 10, "paper_id": 1, "text": text, "page_number": 1}]
        keyword_rows = [{"id": 11, "paper_id": 1, "text": text, "page_number": 1}]
        fused = retrieval.rrf_fuse([("vector", vector_rows), ("keyword", keyword_rows)])
        self.assertEqual(len(fused), 1)
        self.assertEqual(set(fused[0]["sources"]), {"vector", "keyword"})

    def test_relevance_coverage_metric(self) -> None:
        row = {"text": "heart failure primary outcome", "heading": "Results"}
        self.assertEqual(retrieval._coverage("heart failure", row), 1.0)
        self.assertLess(retrieval._coverage("kidney dialysis", row), 1.0)

    def test_reranker_plugin_registry(self) -> None:
        class ConstantReranker:
            name = "constant"

            def rerank(self, query, rows, top_k):
                return rows[:top_k]

        rerank.register_reranker("constant", lambda model: ConstantReranker())
        self.assertIsInstance(rerank.get_reranker("constant"), ConstantReranker)

    def test_chinese_query_uses_segmented_index_end_to_end(self) -> None:
        chinese_rows = retrieval.keyword_search("主要结局", self.db, paper_id=self.paper_id)
        self.assertTrue(any("心力衰竭" in row["text"] for row in chinese_rows))
        # Synonym expansion + segmentation should still cross the language gap.
        english_rows = retrieval.keyword_search("心衰", self.db, paper_id=self.paper_id)
        self.assertTrue(any("heart failure" in row["text"] for row in english_rows))


if __name__ == "__main__":
    unittest.main()
