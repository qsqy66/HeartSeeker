from __future__ import annotations

import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from backend import app as app_module, ingestion, ocr, storage  # noqa: E402


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


class FakeAliyunClient:
    def __init__(self, body=None, error: Exception | None = None) -> None:
        self.body = body
        self.error = error
        self.request = None
        self.runtime = None

    def recognize_all_text_with_options(self, request, runtime):
        self.request = request
        self.runtime = runtime
        if self.error is not None:
            raise self.error
        return SimpleNamespace(body=self.body)


def aliyun_body(content: str = "", blocks: list[str] | None = None):
    sub_images = [SimpleNamespace(block_info=SimpleNamespace(
        block_details=[SimpleNamespace(block_content=value) for value in blocks]
    ))] if blocks is not None else []
    return SimpleNamespace(code="", message="", data=SimpleNamespace(
        content=content, sub_images=sub_images))


class AliyunOcrEngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.image = Path(self.tmp.name) / "page.png"
        self.image.write_bytes(b"fake-png")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _engine(self, client, **overrides) -> ocr.AliyunOcrEngine:
        options = {"access_key_id": "LTAI-test", "access_key_secret": "secret",
                   "client": client}
        options.update(overrides)
        return ocr.AliyunOcrEngine(**options)

    def test_sends_body_and_returns_content(self) -> None:
        client = FakeAliyunClient(aliyun_body(content="第一段\n第二段"))
        engine = self._engine(client, ocr_type="MultiLang")
        text = engine.ocr_image(self.image, "chi_sim+eng")
        self.assertEqual(text, "第一段\n第二段")
        self.assertEqual(client.request.type, "MultiLang")
        self.assertEqual(client.request.body.read(), b"fake-png")
        self.assertIsNone(client.request.url)
        self.assertEqual(client.runtime.read_timeout, 60000)

    def test_falls_back_to_block_details(self) -> None:
        client = FakeAliyunClient(aliyun_body(content="", blocks=["标题", "正文内容"]))
        engine = self._engine(client)
        self.assertEqual(engine.ocr_image(self.image, ""), "标题\n正文内容")

    def test_rejects_oversized_image_without_calling_service(self) -> None:
        client = FakeAliyunClient(aliyun_body(content="unused"))
        engine = self._engine(client)
        with unittest.mock.patch.object(ocr, "MAX_IMAGE_BYTES", 4):
            with self.assertRaises(ocr.OcrError):
                engine.ocr_image(self.image, "")
        self.assertIsNone(client.request)

    def test_missing_credentials_marks_unavailable(self) -> None:
        engine = ocr.AliyunOcrEngine(access_key_id="", access_key_secret="",
                                     client=FakeAliyunClient(aliyun_body()))
        self.assertFalse(engine.available())
        with self.assertRaises(ocr.OcrError):
            engine.ocr_image(self.image, "")

    def test_sdk_error_is_wrapped(self) -> None:
        error = RuntimeError("InvalidAccessKeyId.NotFound")
        engine = self._engine(FakeAliyunClient(error=error))
        with self.assertRaises(ocr.OcrError) as raised:
            engine.ocr_image(self.image, "")
        self.assertIn("InvalidAccessKeyId", str(raised.exception))
        self.assertNotIn("secret", str(raised.exception))

    def test_service_error_code_is_reported(self) -> None:
        body = SimpleNamespace(code="InvalidImage", message="image format error", data=None)
        engine = self._engine(FakeAliyunClient(body))
        with self.assertRaises(ocr.OcrError) as raised:
            engine.ocr_image(self.image, "")
        self.assertIn("InvalidImage", str(raised.exception))

    def test_endpoint_normalisation_and_engine_registry(self) -> None:
        engine = ocr.AliyunOcrEngine(
            access_key_id="id", access_key_secret="secret",
            endpoint="https://ocr-api.cn-shanghai.aliyuncs.com/",
            ocr_type="made-up", client=FakeAliyunClient(aliyun_body()),
        )
        self.assertEqual(engine.endpoint, "ocr-api.cn-shanghai.aliyuncs.com")
        self.assertEqual(engine.ocr_type, ocr.DEFAULT_ALIYUN_OCR_TYPE)
        registry_engine = ocr.get_engine("aliyun")
        self.assertIsInstance(registry_engine, ocr.AliyunOcrEngine)
        self.assertEqual(registry_engine.name, "aliyun")


class AliyunOcrApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app_module.app)
        self.payload = {
            "access_key_id": "LTAI-test",
            "access_key_secret": "secret",
            "endpoint": "ocr-api.cn-hangzhou.aliyuncs.com",
            "ocr_type": "Advanced",
        }

    def test_connection_returns_recognised_text(self) -> None:
        with unittest.mock.patch.object(ocr.AliyunOcrEngine, "available", return_value=True), \
                unittest.mock.patch.object(ocr.AliyunOcrEngine, "ocr_image",
                                          return_value="HeartSeeker_Med OCR test 12345"):
            response = self.client.post("/api/ocr/test", json=self.payload)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["engine"], "aliyun")
        self.assertIn("12345", body["text"])
        self.assertGreaterEqual(body["elapsed_ms"], 0)

    def test_connection_reports_engine_error(self) -> None:
        with unittest.mock.patch.object(ocr.AliyunOcrEngine, "available", return_value=True), \
                unittest.mock.patch.object(ocr.AliyunOcrEngine, "ocr_image",
                                          side_effect=ocr.OcrError("AccessKey 无效")):
            response = self.client.post("/api/ocr/test", json=self.payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("AccessKey", response.json()["detail"])

    def test_connection_requires_credentials(self) -> None:
        payload = {**self.payload, "access_key_id": "", "access_key_secret": ""}
        with unittest.mock.patch.object(app_module, "safe_key", return_value=None), \
                unittest.mock.patch.object(app_module, "_setting", return_value=""):
            response = self.client.post("/api/ocr/test", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("AccessKey", response.json()["detail"])


class SettingsCredentialTestCase(unittest.TestCase):
    def test_non_chat_credential_slots_are_accepted(self) -> None:
        with unittest.mock.patch.object(
                app_module, "_credential_key_store", return_value="saved") as store:
            self.assertEqual(app_module._key_store(ocr.ALIYUN_OCR_KEY, "secret"), "saved")
            app_module._key_store("qwen-rerank", "secret")
        self.assertEqual(store.call_count, 2)

    def test_unknown_provider_is_still_rejected(self) -> None:
        with self.assertRaises(app_module.HTTPException) as caught:
            app_module._key_store("not-a-provider", "secret")
        self.assertEqual(caught.exception.status_code, 400)


class OcrTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.db = self.home / "database" / "heartseeker_med.sqlite3"
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
