"""Pluggable local OCR backends for scanned PDF pages.

OCR is intentionally an interface-first module: the ingestion pipeline calls
:func:`ocr_page`, and the concrete engine (Tesseract CLI, RapidOCR, Alibaba
Cloud OCR, or a user-supplied engine) is resolved from settings/environment.
A missing engine or credential degrades to ``needs_ocr`` instead of breaking
an import.

Resolution order for ``engine="auto"``:

1. ``HEARTSEEKER_MED_OCR_ENGINE``
2. RapidOCR (``rapidocr_onnxruntime`` or ``rapidocr``), if importable
3. Tesseract CLI (``tesseract`` on PATH or ``TESSERACT_BIN``)

Cloud OCR (``engine="aliyun"``) is opt-in and always requires an explicit
AccessKey; it uses the official ``alibabacloud_ocr_api20210707`` SDK.

A third-party engine only needs ``name``, ``available()`` and
``ocr_image(path, lang) -> str`` to plug in.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

DEFAULT_LANG = os.environ.get("HEARTSEEKER_MED_OCR_LANG", "chi_sim+eng")
DEFAULT_DPI = int(os.environ.get("HEARTSEEKER_MED_OCR_DPI", "220"))
MIN_PAGE_CHARS = 20
DEFAULT_ALIYUN_ENDPOINT = os.environ.get(
    "HEARTSEEKER_MED_ALIYUN_OCR_ENDPOINT", "ocr-api.cn-hangzhou.aliyuncs.com")
DEFAULT_ALIYUN_OCR_TYPE = os.environ.get("HEARTSEEKER_MED_ALIYUN_OCR_TYPE", "Advanced")
ALIYUN_OCR_KEY = "aliyun-ocr"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALIYUN_OCR_TYPES = {"General", "Advanced", "MultiLang"}


class OcrError(RuntimeError):
    """Raised when a configured OCR engine fails on a page."""


@runtime_checkable
class OcrEngine(Protocol):
    """Minimal contract for a local image-to-text engine."""

    name: str

    def available(self) -> bool:
        ...

    def ocr_image(self, image_path: Path, lang: str) -> str:
        ...


class TesseractEngine:
    name = "tesseract"

    def __init__(self, binary: Optional[str] = None) -> None:
        self.binary = binary or os.environ.get("TESSERACT_BIN") or shutil.which("tesseract")

    def available(self) -> bool:
        return bool(self.binary)

    def ocr_image(self, image_path: Path, lang: str) -> str:
        if not self.binary:
            raise OcrError("未找到 tesseract 可执行文件")
        command = [str(self.binary), str(image_path), "stdout", "-l", lang or DEFAULT_LANG,
                   "--psm", "3"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as error:
            raise OcrError(f"Tesseract 调用失败：{error}") from error
        if result.returncode != 0:
            message = (result.stderr or "").strip()[:300]
            raise OcrError(f"Tesseract 返回错误：{message or result.returncode}")
        return result.stdout or ""


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self) -> None:
        self._engine = None

    def _load(self):
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError:
            try:
                from rapidocr import RapidOCR  # type: ignore
            except ImportError as error:
                raise OcrError("未安装 RapidOCR（pip install rapidocr-onnxruntime）") from error
        self._engine = RapidOCR()
        return self._engine

    def available(self) -> bool:
        try:
            self._load()
        except OcrError:
            return False
        return True

    def ocr_image(self, image_path: Path, lang: str) -> str:
        engine = self._load()
        try:
            result, _ = engine(str(image_path))
        except Exception as error:  # engine-specific exceptions
            raise OcrError(f"RapidOCR 识别失败：{error}") from error
        lines = []
        for item in result or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(str(item[1]))
        return "\n".join(lines)


class NullEngine:
    name = "none"

    def available(self) -> bool:
        return False

    def ocr_image(self, image_path: Path, lang: str) -> str:
        raise OcrError("未配置本地 OCR 引擎")


class AliyunOcrEngine:
    """Alibaba Cloud OCR unified recognition (RecognizeAllText).

    Uses the official ``alibabacloud_ocr_api20210707`` SDK and the RPC
    endpoint ``ocr-api.<region>.aliyuncs.com``.  Only the raw image bytes and
    the OCR type are sent; credentials stay in the local keyring/settings.
    """

    name = "aliyun"
    max_image_bytes = MAX_IMAGE_BYTES

    def __init__(self, access_key_id: Optional[str] = None,
                 access_key_secret: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 ocr_type: Optional[str] = None,
                 timeout: float = 60.0,
                 client: Any = None) -> None:
        self.access_key_id = (access_key_id or os.environ.get(
            "ALIBABA_CLOUD_ACCESS_KEY_ID", "")).strip()
        self.access_key_secret = (access_key_secret or os.environ.get(
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET", "")).strip()
        endpoint_value = (endpoint or DEFAULT_ALIYUN_ENDPOINT).strip()
        if "://" in endpoint_value:
            endpoint_value = endpoint_value.split("://", 1)[1]
        self.endpoint = endpoint_value.rstrip("/")
        selected_type = (ocr_type or DEFAULT_ALIYUN_OCR_TYPE).strip()
        self.ocr_type = (selected_type if selected_type in ALIYUN_OCR_TYPES
                         else DEFAULT_ALIYUN_OCR_TYPE)
        self.timeout = timeout
        self._sdk_client = client

    def available(self) -> bool:
        if not (self.access_key_id and self.access_key_secret):
            return False
        if self._sdk_client is not None:
            return True
        try:
            import alibabacloud_ocr_api20210707  # noqa: F401
            import alibabacloud_tea_util  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self):
        if self._sdk_client is not None:
            return self._sdk_client
        try:
            from alibabacloud_ocr_api20210707.client import Client
            from alibabacloud_tea_openapi import models as open_api_models
        except ImportError as error:
            raise OcrError(
                "未安装阿里云 OCR SDK：pip install alibabacloud_ocr_api20210707"
            ) from error
        config = open_api_models.Config(
            access_key_id=self.access_key_id,
            access_key_secret=self.access_key_secret,
            endpoint=self.endpoint,
            connect_timeout=int(self.timeout * 1000),
            read_timeout=int(self.timeout * 1000),
        )
        self._sdk_client = Client(config)
        return self._sdk_client

    @staticmethod
    def _extract_text(body: Any) -> str:
        data = getattr(body, "data", None)
        if data is None:
            return ""
        content = str(getattr(data, "content", "") or "").strip()
        if content:
            return content
        lines: list[str] = []
        for sub_image in getattr(data, "sub_images", None) or []:
            block_info = getattr(sub_image, "block_info", None)
            for block in getattr(block_info, "block_details", None) or []:
                value = str(getattr(block, "block_content", "") or "").strip()
                if value:
                    lines.append(value)
        return "\n".join(lines)

    def ocr_image(self, image_path: Path, lang: str) -> str:
        if not self.available():
            raise OcrError("未配置阿里云 OCR AccessKey ID/Secret")
        data = Path(image_path).read_bytes()
        if len(data) > MAX_IMAGE_BYTES:
            raise OcrError(f"图片超过 10MB，无法调用阿里云 OCR（{len(data)} 字节）")
        try:
            from alibabacloud_ocr_api20210707 import models as ocr_models
            from alibabacloud_tea_util import models as util_models
        except ImportError as error:
            raise OcrError(
                "未安装阿里云 OCR SDK：pip install alibabacloud_ocr_api20210707"
            ) from error
        request = ocr_models.RecognizeAllTextRequest(
            type=self.ocr_type,
            body=io.BytesIO(data),
        )
        runtime = util_models.RuntimeOptions(
            connect_timeout=int(self.timeout * 1000),
            read_timeout=int(self.timeout * 1000),
        )
        try:
            response = self._get_client().recognize_all_text_with_options(request, runtime)
        except Exception as error:  # TeaException and transport errors
            code = str(getattr(error, "code", "") or "")
            message = str(getattr(error, "message", "") or error)
            detail = f"{code}: {message}" if code else message
            raise OcrError(f"阿里云 OCR 调用失败：{detail[:300]}") from error
        body = getattr(response, "body", None)
        if body is None:
            raise OcrError("阿里云 OCR 未返回结果")
        code = str(getattr(body, "code", "") or "")
        if code:
            message = str(getattr(body, "message", "") or "")
            raise OcrError(f"阿里云 OCR 返回错误：{code} {message}"[:300])
        return self._extract_text(body)


def get_engine(name: Optional[str] = None) -> OcrEngine:
    """Resolve an OCR engine by name, falling back to ``NullEngine``.

    ``name`` may be ``auto``/``none``/``tesseract``/``rapidocr``/``aliyun``.
    """
    selected = (name or os.environ.get("HEARTSEEKER_MED_OCR_ENGINE") or "auto").strip().lower()
    if selected == "none":
        return NullEngine()
    if selected == "tesseract":
        return TesseractEngine()
    if selected == "rapidocr":
        return RapidOcrEngine()
    if selected == "aliyun":
        return AliyunOcrEngine()
    if selected not in ("", "auto"):
        raise OcrError(f"未知 OCR 引擎：{selected}")
    for engine in (RapidOcrEngine(), TesseractEngine()):
        if engine.available():
            return engine
    return NullEngine()


def render_page_png(page, dpi: int = DEFAULT_DPI) -> bytes:
    pixmap = page.get_pixmap(dpi=max(72, min(int(dpi), 600)), alpha=False)
    return pixmap.tobytes("png")


def ocr_page(page, engine: OcrEngine, lang: str = DEFAULT_LANG,
             dpi: int = DEFAULT_DPI) -> str:
    """Render one PDF page and run the local OCR engine over it."""
    if not engine.available():
        raise OcrError("没有可用的 OCR 引擎")
    image = render_page_png(page, dpi=dpi)
    limit = getattr(engine, "max_image_bytes", None)
    while limit and len(image) > limit and dpi > 100:
        dpi = max(100, int(dpi * 0.8))
        image = render_page_png(page, dpi=dpi)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(image)
        path = Path(handle.name)
    try:
        return (engine.ocr_image(path, lang) or "").strip()
    finally:
        path.unlink(missing_ok=True)
