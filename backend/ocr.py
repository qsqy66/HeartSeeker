"""Pluggable local OCR backends for scanned PDF pages.

OCR is intentionally an interface-first module: the ingestion pipeline calls
:func:`ocr_page`, and the concrete engine (Tesseract CLI, RapidOCR, or a
user-supplied engine) is resolved from settings/environment.  No cloud OCR is
used, and a missing engine degrades to ``needs_ocr`` instead of breaking an
import.

Resolution order for ``engine="auto"``:

1. ``PAPER_READER_OCR_ENGINE`` environment variable
2. RapidOCR (``rapidocr_onnxruntime`` or ``rapidocr``), if importable
3. Tesseract CLI (``tesseract`` on PATH or ``TESSERACT_BIN``)

A third-party engine only needs ``name``, ``available()`` and
``ocr_image(path, lang) -> str`` to plug in.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

DEFAULT_LANG = os.environ.get("PAPER_READER_OCR_LANG", "chi_sim+eng")
DEFAULT_DPI = int(os.environ.get("PAPER_READER_OCR_DPI", "220"))
MIN_PAGE_CHARS = 20


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


def get_engine(name: Optional[str] = None) -> OcrEngine:
    """Resolve an OCR engine by name, falling back to ``NullEngine``.

    ``name`` may be ``auto``/``none``/``tesseract``/``rapidocr`` or a
    colon-separated registry key for a custom engine installed by the user.
    """
    selected = (name or os.environ.get("PAPER_READER_OCR_ENGINE") or "auto").strip().lower()
    if selected == "none":
        return NullEngine()
    if selected == "tesseract":
        return TesseractEngine()
    if selected == "rapidocr":
        return RapidOcrEngine()
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
        raise OcrError("没有可用的本地 OCR 引擎")
    image = render_page_png(page, dpi=dpi)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(image)
        path = Path(handle.name)
    try:
        return (engine.ocr_image(path, lang) or "").strip()
    finally:
        path.unlink(missing_ok=True)
