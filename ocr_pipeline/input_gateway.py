"""Input gateway: file validation, image decode, and size enforcement.

Validates uploaded images before they enter the OCR pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ocr_pipeline.config import OCRConfig

logger = logging.getLogger(__name__)

# Supported MIME types and their magic bytes
SUPPORTED_MIMES = {"image/jpeg", "image/png"}

# Magic byte signatures
MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
}

MIN_RESOLUTION = (640, 480)


class InputValidationError(Exception):
    """Raised when input image fails validation."""

    def __init__(self, message: str, error_code: str = "invalid_input"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


def _detect_mime_type(data: bytes) -> str | None:
    """Detect MIME type from magic bytes."""
    for signature, mime_type in MAGIC_SIGNATURES.items():
        if data[: len(signature)] == signature:
            return mime_type
    return None


def validate_and_decode(
    source: str | Path | bytes,
    config: OCRConfig | None = None,
) -> np.ndarray:
    """Validate an image file and decode it to a numpy array (RGB).

    Args:
        source: File path (str/Path) or raw bytes of the image.
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        numpy array of the image in RGB color space.

    Raises:
        InputValidationError: If the image fails validation.
    """
    if config is None:
        config = OCRConfig()

    # Read bytes from source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise InputValidationError(f"File not found: {path}", "file_not_found")
        if not path.is_file():
            raise InputValidationError(f"Not a regular file: {path}", "invalid_input")
        data = path.read_bytes()
    elif isinstance(source, bytes):
        data = source
    else:
        raise InputValidationError(
            f"Unsupported source type: {type(source).__name__}", "invalid_input"
        )

    # Size check
    if len(data) > config.max_image_size_bytes:
        raise InputValidationError(
            f"Image size {len(data) / 1024 / 1024:.1f} MB exceeds "
            f"limit of {config.max_image_size_mb} MB",
            "file_too_large",
        )

    if len(data) < 100:
        raise InputValidationError("File too small to be a valid image", "invalid_input")

    # Magic bytes validation
    mime_type = _detect_mime_type(data)
    if mime_type is None or mime_type not in SUPPORTED_MIMES:
        raise InputValidationError(
            f"Unsupported image format. Supported: JPEG, PNG. Detected: {mime_type}",
            "unsupported_format",
        )

    # Decode image
    try:
        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise InputValidationError("Failed to decode image data", "decode_error")
    except InputValidationError:
        raise
    except Exception as e:
        raise InputValidationError(f"Image decode error: {e}", "decode_error") from e

    # Apply EXIF orientation via PIL for correct rotation
    try:
        from io import BytesIO

        pil_img = Image.open(BytesIO(data))
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        image = np.array(pil_img)
        # Convert RGB to BGR for OpenCV pipeline consistency, then back to RGB at end
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception:
        # If EXIF handling fails, continue with cv2-decoded image
        logger.debug("EXIF orientation handling failed, using cv2-decoded image")

    # Resolution check
    h, w = image.shape[:2]
    if w < MIN_RESOLUTION[0] or h < MIN_RESOLUTION[1]:
        logger.warning(
            "Image resolution %dx%d is below minimum %dx%d",
            w,
            h,
            MIN_RESOLUTION[0],
            MIN_RESOLUTION[1],
        )
        # Don't reject — just warn. Low-res images degrade gracefully.

    # Convert to RGB for pipeline (OpenCV loads as BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    logger.debug("Image validated: %dx%d, %s, %.1f KB", w, h, mime_type, len(data) / 1024)
    return image_rgb
