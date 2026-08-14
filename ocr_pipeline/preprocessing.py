"""Image preprocessing: deskew, contrast enhancement, noise reduction.

Each step is a standalone function that can be individually disabled.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def deskew(image: np.ndarray, max_angle: float = 45.0) -> np.ndarray:
    """Detect and correct skew in the image using Hough line detection.

    Args:
        image: Input image (RGB numpy array).
        max_angle: Maximum skew angle to correct (degrees).

    Returns:
        Deskewed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return image

    # Compute angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Only consider near-horizontal lines (within ±max_angle of 0 or 180)
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return image

    # Use median angle to be robust against outliers
    median_angle = float(np.median(angles))

    # Only correct if skew is meaningful (> 0.5 degrees)
    if abs(median_angle) < 0.5:
        return image

    logger.debug("Deskew: detected angle %.2f°", median_angle)

    # Rotate to correct
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # Calculate new bounding box to avoid cropping
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    new_w = int(h * sin_val + w * cos_val)
    new_h = int(h * cos_val + w * sin_val)
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE contrast enhancement on the L-channel of LAB color space.

    Args:
        image: Input image (RGB numpy array).
        clip_limit: CLAHE clip limit.
        tile_size: CLAHE tile grid size.

    Returns:
        Contrast-enhanced image in RGB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return result


def denoise(
    image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75
) -> np.ndarray:
    """Apply bilateral filter for noise reduction while preserving edges.

    Args:
        image: Input image (RGB numpy array).
        d: Diameter of pixel neighborhood.
        sigma_color: Filter sigma in color space.
        sigma_space: Filter sigma in coordinate space.

    Returns:
        Denoised image.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def preprocess(image: np.ndarray, skip_deskew: bool = False) -> np.ndarray:
    """Run the full preprocessing pipeline.

    Args:
        image: Input image (RGB numpy array).
        skip_deskew: If True, skip deskew step.

    Returns:
        Preprocessed image ready for classification and OCR.
    """
    result = image.copy()

    # Step 1: Deskew
    if not skip_deskew:
        result = deskew(result)

    # Step 2: Contrast enhancement
    result = enhance_contrast(result)

    # Step 3: Noise reduction
    result = denoise(result)

    return result
