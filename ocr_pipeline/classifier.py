"""Document classifier: PAN vs Aadhaar vs Unknown.

Uses a combination of heuristics (template matching, text cues) for
classification. An ONNX model can be plugged in when trained.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np

from ocr_pipeline.config import OCRConfig

logger = logging.getLogger(__name__)


class DocumentType(StrEnum):
    """Supported document types."""

    PAN = "PAN"
    AADHAAR = "AADHAAR"
    UNKNOWN = "UNKNOWN"


class ClassificationResult:
    """Result of document classification."""

    def __init__(self, doc_type: DocumentType, confidence: float):
        self.doc_type = doc_type
        self.confidence = confidence


def _color_histogram_features(image: np.ndarray) -> dict:
    """Extract color histogram features for classification.

    PAN cards are predominantly blue/tan. Aadhaar cards have distinct
    orange-white-green header bands.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Blue detection (PAN cards have blue background)
    # HSV blue range: H=100-130, S>50, V>50
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    blue_ratio = np.sum(blue_mask > 0) / blue_mask.size

    # Orange/saffron detection (Aadhaar header)
    orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
    orange_ratio = np.sum(orange_mask > 0) / orange_mask.size

    # Green detection (Aadhaar header)
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    green_ratio = np.sum(green_mask > 0) / green_mask.size

    return {
        "blue_ratio": blue_ratio,
        "orange_ratio": orange_ratio,
        "green_ratio": green_ratio,
    }


def _aspect_ratio_check(image: np.ndarray) -> float:
    """Check aspect ratio. PAN ~1.585 (credit card), Aadhaar ~1.585 too.

    Returns the aspect ratio (width / height).
    """
    h, w = image.shape[:2]
    return w / h if h > 0 else 0


def _text_cue_detection(image: np.ndarray) -> dict:
    """Detect structural text cues using simple template/pattern matching.

    Look for distinguishing visual elements:
    - PAN: "INCOME TAX DEPARTMENT", "GOVT OF INDIA", signature panel at bottom
    - Aadhaar: "UNIQUE IDENTIFICATION", UIDAI logo position, 12-digit number format
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Analyze top region for header content
    top_region = gray[: int(h * 0.25), :]

    # PAN cards typically have a lighter/tan top with emblem
    top_mean = float(np.mean(top_region))

    # Analyze bottom region — PAN has the number, Aadhaar has number lower
    bottom_region = gray[int(h * 0.7) :, :]
    bottom_mean = float(np.mean(bottom_region))

    # Photo region detection (left side for both cards)
    left_region = gray[int(h * 0.2) : int(h * 0.7), : int(w * 0.3)]

    # Detect if there's a rectangular face/photo region
    edges = cv2.Canny(left_region, 50, 150)
    photo_edge_density = float(np.sum(edges > 0)) / edges.size

    return {
        "top_mean_intensity": top_mean,
        "bottom_mean_intensity": bottom_mean,
        "photo_edge_density": photo_edge_density,
    }


def classify_document(image: np.ndarray, config: OCRConfig | None = None) -> ClassificationResult:
    """Classify an image as PAN, Aadhaar, or Unknown.

    Uses a heuristic approach combining:
    1. Color histogram analysis (blue for PAN, tricolor for Aadhaar)
    2. Aspect ratio verification
    3. Structural layout cues

    Args:
        image: Preprocessed RGB image (numpy array).
        config: Pipeline configuration.

    Returns:
        ClassificationResult with document type and confidence.
    """
    if config is None:
        config = OCRConfig()

    # Try ONNX model first if available
    model_path = config.models_dir / "classifier.onnx"
    if model_path.exists():
        result = _classify_with_model(image, model_path)
        if result is not None:
            return result

    # Fallback to heuristic classification
    return _classify_heuristic(image)


def _classify_with_model(image: np.ndarray, model_path: Path) -> ClassificationResult | None:
    """Classify using ONNX model if available."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name

        # Preprocess for model: resize to 224x224, normalize
        resized = cv2.resize(image, (224, 224))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)  # Add batch dim

        outputs = session.run(None, {input_name: blob})
        probs = outputs[0][0]

        # Classes: [PAN, AADHAAR, UNKNOWN]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        doc_types = [DocumentType.PAN, DocumentType.AADHAAR, DocumentType.UNKNOWN]

        return ClassificationResult(doc_types[class_idx], confidence)
    except Exception as e:
        logger.warning("ONNX model classification failed: %s. Falling back to heuristic.", e)
        return None


def _classify_heuristic(image: np.ndarray) -> ClassificationResult:
    """Heuristic classification based on color and structural features."""
    color_features = _color_histogram_features(image)
    text_cues = _text_cue_detection(image)

    pan_score = 0.0
    aadhaar_score = 0.0

    # Color scoring
    if color_features["blue_ratio"] > 0.05:
        pan_score += 0.4
    if color_features["blue_ratio"] > 0.15:
        pan_score += 0.2

    if color_features["orange_ratio"] > 0.02 and color_features["green_ratio"] > 0.02:
        aadhaar_score += 0.4
    if color_features["orange_ratio"] > 0.01:
        aadhaar_score += 0.1
    if color_features["green_ratio"] > 0.01:
        aadhaar_score += 0.1

    # Layout scoring — PAN tends to have darker/more uniform bottom (number region)
    if text_cues["top_mean_intensity"] > 150:
        pan_score += 0.1

    # Photo region presence (both cards have it, but location differs slightly)
    if text_cues["photo_edge_density"] > 0.05:
        pan_score += 0.1
        aadhaar_score += 0.1

    # Aspect ratio — both are credit-card shaped
    aspect = _aspect_ratio_check(image)
    if 1.3 < aspect < 1.8:
        pan_score += 0.1
        aadhaar_score += 0.1

    # Determine result
    if pan_score > aadhaar_score and pan_score > 0.3:
        confidence = min(pan_score / (pan_score + aadhaar_score + 0.1), 0.95)
        return ClassificationResult(DocumentType.PAN, confidence)
    elif aadhaar_score > pan_score and aadhaar_score > 0.3:
        confidence = min(aadhaar_score / (pan_score + aadhaar_score + 0.1), 0.95)
        return ClassificationResult(DocumentType.AADHAAR, confidence)
    else:
        # Neither scored high enough — try to make a best guess or reject
        if pan_score > aadhaar_score:
            return ClassificationResult(DocumentType.PAN, max(pan_score, 0.3))
        elif aadhaar_score > pan_score:
            return ClassificationResult(DocumentType.AADHAAR, max(aadhaar_score, 0.3))
        else:
            return ClassificationResult(DocumentType.UNKNOWN, 0.5)
