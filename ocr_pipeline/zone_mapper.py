"""Card boundary detection and zone mapping.

Detects the card boundary via contour detection, applies perspective
correction, then crops known zones based on card-type-specific templates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from ocr_pipeline.classifier import DocumentType

logger = logging.getLogger(__name__)


@dataclass
class ZoneTemplate:
    """A named region of interest defined in relative coordinates (0-1)."""

    name: str
    top: float  # Top edge as fraction of card height
    left: float  # Left edge as fraction of card width
    width: float  # Width as fraction of card width
    height: float  # Height as fraction of card height


# PAN Card zone templates (relative coordinates)
# Based on empirical measurement: after boundary detection crops to the card,
# text often starts near x=0 (photo may or may not be present on left side)
PAN_ZONES = [
    ZoneTemplate(name="name_block", top=0.20, left=0.0, width=0.70, height=0.14),
    ZoneTemplate(name="father_name_block", top=0.33, left=0.0, width=0.70, height=0.14),
    ZoneTemplate(name="dob_zone", top=0.46, left=0.0, width=0.45, height=0.14),
    ZoneTemplate(name="pan_number_zone", top=0.65, left=0.0, width=0.60, height=0.14),
]

# Aadhaar Card zone templates (relative coordinates)
AADHAAR_ZONES = [
    ZoneTemplate(name="name_block", top=0.22, left=0.0, width=0.95, height=0.18),
    ZoneTemplate(name="guardian_block", top=0.38, left=0.0, width=0.95, height=0.12),
    ZoneTemplate(name="dob_gender_zone", top=0.48, left=0.0, width=0.70, height=0.15),
    ZoneTemplate(name="aadhaar_number_zone", top=0.68, left=0.0, width=0.95, height=0.18),
]


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest difference

    return rect


def detect_card_boundary(image: np.ndarray) -> np.ndarray:
    """Detect card boundary and apply perspective correction.

    Args:
        image: Preprocessed RGB image.

    Returns:
        Perspective-corrected card image (RGB), or the original image
        with margin crop if no card boundary is detected.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logger.debug("No contours found; using full image with margin crop")
        return _margin_crop(image)

    # Filter contours by area (card should be > 20% of image area)
    min_area = h * w * 0.2
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        logger.debug("No large contours found; using full image with margin crop")
        return _margin_crop(image)

    # Find the largest contour and approximate to polygon
    largest = max(valid_contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        # Found a quadrilateral — apply perspective transform
        pts = approx.reshape(4, 2).astype(np.float32)
        ordered = _order_points(pts)
        return _perspective_transform(image, ordered)
    else:
        # Not a clean quad — use bounding rect
        x, y, rw, rh = cv2.boundingRect(largest)
        cropped = image[y : y + rh, x : x + rw]
        if cropped.size == 0:
            return _margin_crop(image)
        return cropped


def _perspective_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply perspective transform to straighten the card."""
    tl, tr, br, bl = pts

    # Compute output dimensions
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # Target points
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def _margin_crop(image: np.ndarray, margin_pct: float = 0.03) -> np.ndarray:
    """Crop a small margin from edges to remove potential borders."""
    h, w = image.shape[:2]
    mx = int(w * margin_pct)
    my = int(h * margin_pct)
    return image[my : h - my, mx : w - mx]


def extract_zones(
    card_image: np.ndarray,
    doc_type: DocumentType,
    expand_retry: bool = True,
) -> dict[str, np.ndarray]:
    """Crop card image into named zones based on document type template.

    Args:
        card_image: Perspective-corrected card image (RGB).
        doc_type: Detected document type (PAN or AADHAAR).
        expand_retry: If True, expand zones by 10% on empty crop.

    Returns:
        Dictionary mapping zone names to cropped image arrays.
    """
    templates = PAN_ZONES if doc_type == DocumentType.PAN else AADHAAR_ZONES
    h, w = card_image.shape[:2]
    zones = {}

    for zone in templates:
        crop = _crop_zone(card_image, zone, h, w)
        if crop is None or crop.size == 0:
            if expand_retry:
                # Expand zone by 10% in each direction
                expanded = ZoneTemplate(
                    name=zone.name,
                    top=max(0, zone.top - 0.05),
                    left=max(0, zone.left - 0.05),
                    width=min(1.0 - max(0, zone.left - 0.05), zone.width + 0.10),
                    height=min(1.0 - max(0, zone.top - 0.05), zone.height + 0.10),
                )
                crop = _crop_zone(card_image, expanded, h, w)
                if crop is not None and crop.size > 0:
                    logger.debug("Zone '%s' expanded for retry", zone.name)
            if crop is None or crop.size == 0:
                logger.warning("Zone '%s' resulted in empty crop", zone.name)
                continue
        zones[zone.name] = crop

    return zones


def _crop_zone(image: np.ndarray, zone: ZoneTemplate, h: int, w: int) -> np.ndarray | None:
    """Crop a single zone from the card image."""
    y1 = int(zone.top * h)
    x1 = int(zone.left * w)
    y2 = int((zone.top + zone.height) * h)
    x2 = int((zone.left + zone.width) * w)

    # Clamp to image bounds
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))

    if y2 <= y1 or x2 <= x1:
        return None

    return image[y1:y2, x1:x2]
