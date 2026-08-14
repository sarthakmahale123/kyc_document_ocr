"""OCR engine abstraction with PaddleOCR and Tesseract backends.

Provides a unified interface for OCR recognition regardless of engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from ocr_pipeline.config import OCRConfig

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A recognized text block with position and confidence."""

    text: str
    confidence: float
    bbox: list[tuple[int, int]] = field(default_factory=list)


class OCRBackend(Protocol):
    """Protocol for OCR engine backends."""

    def recognize(self, image: np.ndarray) -> list[TextBlock]:
        """Recognize text in an image region.

        Args:
            image: RGB numpy array of the zone to recognize.

        Returns:
            List of recognized text blocks with confidence scores.
        """
        ...


class PaddleOCRBackend:
    """PaddleOCR backend using PP-OCRv4 mobile models."""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        """Lazy-load PaddleOCR engine."""
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR

                self._engine = PaddleOCR(
                    use_textline_orientation=True,
                    use_doc_unwarping=False,  # Disabled: our preprocessing handles deskew
                    use_doc_orientation_classify=True,
                    lang="en",
                )
                logger.info("PaddleOCR engine initialized (v3.7+)")
            except (ImportError, Exception) as e:
                logger.error("PaddleOCR initialization failed: %s", e)
                raise
        return self._engine

    def recognize(self, image: np.ndarray) -> list[TextBlock]:
        """Recognize text using PaddleOCR v3.7+ API."""
        engine = self._get_engine()

        # PaddleOCR expects BGR
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            results = engine.predict(bgr_image)
        except Exception as e:
            logger.warning("PaddleOCR recognition failed: %s", e)
            return []

        blocks = []
        if results:
            for item in results:
                texts = item.get("rec_texts", [])
                scores = item.get("rec_scores", [])
                polys = item.get("rec_polys", [])

                for i, text in enumerate(texts):
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    bbox = []
                    if i < len(polys) and polys[i] is not None:
                        try:
                            bbox = [(int(p[0]), int(p[1])) for p in polys[i]]
                        except (TypeError, IndexError):
                            pass
                    blocks.append(TextBlock(text=text, confidence=conf, bbox=bbox))

        return blocks


class TesseractBackend:
    """Tesseract OCR backend."""

    def recognize(self, image: np.ndarray) -> list[TextBlock]:
        """Recognize text using Tesseract."""
        try:
            import pytesseract
        except ImportError:
            logger.error("pytesseract not installed. Install with: pip install pytesseract")
            raise

        # Convert to grayscale for better Tesseract performance
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Apply threshold for cleaner input
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            # Get detailed output with confidence per word
            data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
        except Exception as e:
            logger.warning("Tesseract recognition failed: %s", e)
            return []

        blocks = []
        n_boxes = len(data["text"])

        # Group words into lines
        current_line: list[str] = []
        current_confs: list[float] = []
        current_line_num = -1

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            line_num = data["line_num"][i]

            if line_num != current_line_num and current_line:
                # Emit previous line
                line_text = " ".join(current_line)
                avg_conf = sum(current_confs) / len(current_confs) / 100.0 if current_confs else 0
                if line_text.strip():
                    blocks.append(TextBlock(text=line_text, confidence=avg_conf))
                current_line = []
                current_confs = []

            current_line_num = line_num
            if text and conf > 0:
                current_line.append(text)
                current_confs.append(float(conf))

        # Emit last line
        if current_line:
            line_text = " ".join(current_line)
            avg_conf = sum(current_confs) / len(current_confs) / 100.0 if current_confs else 0
            if line_text.strip():
                blocks.append(TextBlock(text=line_text, confidence=avg_conf))

        return blocks


# Singleton instances (lazy-loaded)
_paddle_backend: PaddleOCRBackend | None = None
_tesseract_backend: TesseractBackend | None = None


def get_ocr_backend(engine: str = "auto") -> OCRBackend:
    """Get the appropriate OCR backend.

    Args:
        engine: "paddle", "tesseract", or "auto".
                "auto" tries PaddleOCR first, falls back to Tesseract.

    Returns:
        An OCR backend instance.
    """
    global _paddle_backend, _tesseract_backend

    if engine == "paddle":
        if _paddle_backend is None:
            _paddle_backend = PaddleOCRBackend()
        return _paddle_backend

    if engine == "tesseract":
        if _tesseract_backend is None:
            _tesseract_backend = TesseractBackend()
        return _tesseract_backend

    # Auto mode: try Paddle first
    try:
        if _paddle_backend is None:
            _paddle_backend = PaddleOCRBackend()
        # Test that PaddleOCR can be loaded
        _paddle_backend._get_engine()
        return _paddle_backend
    except (ImportError, Exception) as e:
        logger.info("PaddleOCR unavailable (%s), using Tesseract", e)
        if _tesseract_backend is None:
            _tesseract_backend = TesseractBackend()
        return _tesseract_backend


def recognize_zone(
    zone_image: np.ndarray,
    config: OCRConfig | None = None,
    _backend: OCRBackend | None = None,
) -> list[TextBlock]:
    """Recognize text in a single zone image.

    Args:
        zone_image: RGB numpy array of a cropped zone.
        config: Pipeline configuration.
        _backend: Pre-resolved backend (avoids repeated lookups in parallel calls).

    Returns:
        List of recognized text blocks.
    """
    if config is None:
        config = OCRConfig()

    backend = _backend or get_ocr_backend(config.engine)
    return backend.recognize(zone_image)


def warm_up_backend(config: OCRConfig | None = None) -> OCRBackend:
    """Ensure the OCR backend is initialized and ready for use.

    Call this before parallel OCR to avoid concurrent initialization.

    Returns:
        The initialized backend instance.
    """
    if config is None:
        config = OCRConfig()

    backend = get_ocr_backend(config.engine)
    # For PaddleOCR, ensure the engine is loaded
    if isinstance(backend, PaddleOCRBackend):
        backend._get_engine()
    return backend
