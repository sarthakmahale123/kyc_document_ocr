"""PAN/Aadhaar OCR Auto-Fill Pipeline.

Public API:
    extract_card(image_path, config=None) -> dict
    extract_batch(image_paths, config=None) -> list[dict]
    OCRConfig - configuration dataclass
"""

from ocr_pipeline.config import OCRConfig
from ocr_pipeline.pipeline import extract_batch, extract_card

__all__ = ["extract_card", "extract_batch", "OCRConfig"]
__version__ = "0.1.0"
