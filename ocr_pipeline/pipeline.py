"""Pipeline orchestrator: ties all stages together.

Public API:
    extract_card(source, config=None) -> dict
    extract_batch(sources, config=None) -> list[dict]

Strategy: Run OCR once on the full card image, then assign detected text blocks
to logical zones based on their Y-position (vertical percentage). This is faster
and more accurate than cropping zones and running OCR per-zone, because PaddleOCR's
text detection model works best with full-card context.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

from ocr_pipeline.classifier import DocumentType, classify_document
from ocr_pipeline.config import OCRConfig
from ocr_pipeline.input_gateway import InputValidationError, validate_and_decode
from ocr_pipeline.ocr_engine import TextBlock, warm_up_backend
from ocr_pipeline.postprocessing import (
    normalize_aadhaar_number,
    normalize_date,
    normalize_gender,
    normalize_name,
    normalize_pan_number,
)
from ocr_pipeline.preprocessing import preprocess
from ocr_pipeline.response import (
    ExtractionResult,
    FieldResult,
    assemble_response,
    build_field_result,
)
from ocr_pipeline.script_filter import is_latin_text
from ocr_pipeline.validators import validate_aadhaar, validate_dob, validate_gender, validate_pan

logger = logging.getLogger(__name__)

# PAN number regex for detection in OCR output
_PAN_PATTERN = re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
# Date patterns
_DATE_PATTERN = re.compile(r"\d{1,2}[/\-\.]+\d{1,2}[/\-\.]+\d{4}")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
# Aadhaar number: 4 digits space 4 digits space 4 digits
_AADHAAR_PATTERN = re.compile(r"\d{4}\s?\d{4}\s?\d{4}")


def extract_card(
    source: str | Path | bytes,
    config: OCRConfig | None = None,
) -> dict:
    """Extract fields from a single PAN or Aadhaar card image.

    Args:
        source: Image file path (str/Path) or raw bytes.
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        Dictionary matching the structured JSON output schema.
    """
    if config is None:
        config = OCRConfig()

    start_time = time.perf_counter()

    try:
        result = _run_pipeline(source, config)
    except InputValidationError as e:
        result = ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            error=f"{e.error_code}: {e.message}",
        )
    except Exception as e:
        logger.exception("Unexpected pipeline error")
        result = ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            error=f"pipeline_error: {e!s}",
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    result.processing_time_ms = elapsed_ms

    return result.to_dict()


def extract_batch(
    sources: list[str | Path | bytes],
    config: OCRConfig | None = None,
) -> list[dict]:
    """Extract fields from multiple card images sequentially."""
    if config is None:
        config = OCRConfig()

    return [extract_card(source, config) for source in sources]


def _run_pipeline(source: str | Path | bytes, config: OCRConfig) -> ExtractionResult:
    """Execute the full extraction pipeline.

    Stages:
    1. Input validation + decode
    2. Preprocessing
    3. Classification
    4. Boundary detection
    5. Full-card OCR (single pass)
    6. Positional field assignment + post-processing
    7. Validation + confidence scoring
    8. Response assembly
    """
    # Stage 1: Input validation + decode
    image = validate_and_decode(source, config)

    # Stage 2: Preprocessing
    preprocessed = preprocess(image)

    # Stage 3: Classification (heuristic — will be corrected by OCR content below)
    classification = classify_document(preprocessed, config)
    doc_type = classification.doc_type
    logger.info("Classified as %s (confidence: %.2f)", doc_type.value, classification.confidence)

    # Stage 4: Use the preprocessed image directly
    # PaddleOCR v3.7 has built-in document orientation and detection that handles
    # card finding, whitespace, and orientation. Boundary detection was causing
    # regressions (clipping text edges on tight-cropped card photos).
    card_image = preprocessed

    # Resize only very large images (> 1600px) to keep OCR responsive
    # Don't resize aggressively — it degrades text quality on smaller card text
    h, w = card_image.shape[:2]
    max_side = 1600
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        import cv2

        card_image = cv2.resize(card_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug("Resized card image from %dx%d to %dx%d", w, h, new_w, new_h)

    # Stage 5: Full-card OCR (single pass — faster and more accurate)
    backend = warm_up_backend(config)
    all_blocks = backend.recognize(card_image)

    if not all_blocks:
        return ExtractionResult(
            document_type=doc_type,
            error="ocr_failed: No text detected on card",
        )

    # Stage 5b: Override classification based on OCR content
    # If we find a PAN number or "INCOME TAX" text, it's definitely a PAN card
    # regardless of what the heuristic classifier said.
    all_text_upper = " ".join(b.text.upper() for b in all_blocks)
    has_pan_number = bool(_PAN_PATTERN.search(all_text_upper))
    has_income_tax = "INCOME TAX" in all_text_upper
    has_aadhaar_number = bool(_AADHAAR_PATTERN.search(all_text_upper))
    has_uidai = "UNIQUE IDENTIFICATION" in all_text_upper or "UIDAI" in all_text_upper
    has_govt_india_aadhaar = "GOVERNMENT OF INDIA" in all_text_upper and not has_income_tax

    # Also check for partial Aadhaar (4-digit standalone) as an indicator
    has_partial_aadhaar = any(
        re.match(r"^\d{4}$", b.text.strip()) and b.confidence > 0.8
        for b in all_blocks
    )

    if has_pan_number or has_income_tax:
        doc_type = DocumentType.PAN
    elif has_aadhaar_number or has_uidai:
        doc_type = DocumentType.AADHAAR
    elif has_govt_india_aadhaar or has_partial_aadhaar:
        doc_type = DocumentType.AADHAAR
    elif doc_type == DocumentType.UNKNOWN:
        return ExtractionResult(
            document_type=DocumentType.UNKNOWN,
            error="unknown_document: Could not classify as PAN or Aadhaar",
        )

    # Stage 6-7: Positional field assignment + validation
    h, _w = card_image.shape[:2]
    if doc_type == DocumentType.PAN:
        extracted_fields = _extract_pan_fields_positional(all_blocks, h, config)
    else:
        extracted_fields = _extract_aadhaar_fields_positional(all_blocks, h, config)

    # Stage 8: Response assembly
    return assemble_response(doc_type, extracted_fields, processing_time_ms=0)


def _get_y_percent(block: TextBlock, card_height: int) -> float:
    """Get the vertical position of a text block as a percentage of card height."""
    if block.bbox and len(block.bbox) >= 2:
        # Use top-left Y coordinate
        y = block.bbox[0][1]
        return (y / card_height) * 100
    return 0.0


def _extract_pan_fields_positional(
    blocks: list[TextBlock], card_height: int, config: OCRConfig
) -> dict[str, FieldResult]:
    """Extract PAN card fields using content-based detection.

    Strategy: Identify fields by their content patterns, NOT by position.
    PAN cards come in many layouts (old blue, new e-PAN, different issuance years)
    and images may be rotated, cropped, or have varying whitespace.

    Detection rules:
    - PAN Number: regex [A-Z]{5}[0-9]{4}[A-Z] — unique, unambiguous
    - DOB: date pattern DD/MM/YYYY or DD-MM-YYYY
    - Names: Latin-script text blocks that aren't labels or identifiers
    - Name vs Father's Name: determined by order of appearance (top-to-bottom)
    """
    fields: dict[str, FieldResult] = {}

    pan_number_text = None
    pan_number_conf = 0.0
    dob_text = None
    dob_conf = 0.0
    name_candidates: list[tuple[str, float, float]] = []  # (text, conf, y_pct)
    father_label_y: float | None = None  # Y-position of "Father's Name" label
    name_label_y: float | None = None  # Y-position of "Name" label

    # Known labels to exclude from name detection
    skip_labels = {
        "INCOME TAX",
        "GOVT",
        "INDIA",
        "PERMANENT",
        "ACCOUNT",
        "NUMBER",
        "SIGNATURE",
        "DEPARTMENT",
        "DATE OF BIRTH",
        "DATE",
        "BIRTH",
        "FATHER",
        "NAME",
        "DOB",
        "CARD",
        "E-PERMANENT",
        "VALID",
        "REPUBLIC",
        "SIGN",
        "APPLICATION",
        "DIGITALLY",
        "PHYSICALLY",
        "COMMISSIONER",
        "COMPUTER",
        "OPERATIONS",
    }

    for block in blocks:
        y_pct = _get_y_percent(block, card_height)
        text = block.text.strip()
        conf = block.confidence

        if not text or conf < 0.3:
            continue

        text_upper = text.upper().strip()

        # 1. Detect PAN number ANYWHERE on the card (no Y constraint)
        pan_match = _PAN_PATTERN.search(text_upper)
        if pan_match:
            # Prefer higher confidence match if multiple found
            if pan_number_text is None or conf > pan_number_conf:
                pan_number_text = pan_match.group(0)
                pan_number_conf = conf
            continue

        # 2. Detect date ANYWHERE on the card
        date_match = _DATE_PATTERN.search(text)
        if date_match:
            if dob_text is None or conf > dob_conf:
                dob_text = date_match.group(0)
                dob_conf = conf
            continue

        # 3. Skip known labels (but note label positions for assignment)
        if "FATHER" in text_upper:
            father_label_y = y_pct
        # Detect "Name" label (various forms: "/Name", "Name", "नाम/Name")
        if text_upper == "NAME" or "/NAME" in text_upper or text_upper.endswith("/NAME"):
            name_label_y = y_pct
        # Also detect standalone "Name" mixed case (OCR often reads the label)
        if re.match(r"^(.*/)?\s*Name\s*$", text) and len(text) < 10:
            name_label_y = y_pct  # noqa: F841
        if any(label in text_upper for label in skip_labels):
            continue

        # 4. Skip very short text (OCR artifacts) or low-confidence garbage
        if len(text) < 3 or conf < 0.5:
            continue

        # 5. Skip text that's mostly digits or special chars
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < len(text) * 0.6:
            continue

        # 6. Name candidates — Latin script text that passes all filters
        if is_latin_text(text) and conf > 0.7:
            # Additional filters to reject mis-read Devanagari as Latin:
            # - Real names have at least one word >= 3 chars
            # - Real names contain only letters, spaces, hyphens, apostrophes
            # - Must have confidence > 0.9 OR be multi-word with > 0.8
            words = text.split()
            has_real_word = any(len(w) >= 3 for w in words)
            only_name_chars = bool(re.match(r"^[A-Za-z\s'\-\.]+$", text))
            is_high_quality = conf > 0.9 or (len(words) >= 2 and conf > 0.8)

            if has_real_word and only_name_chars and is_high_quality:
                # Reject single words <= 4 chars ONLY if low confidence (likely OCR fragment)
                if len(words) == 1 and len(words[0]) <= 4 and conf < 0.97:
                    continue
                name_candidates.append((text, conf, y_pct))

    # Sort name candidates by Y position (top to bottom on card)
    name_candidates.sort(key=lambda x: x[2])

    # Assign names using label positions if detected
    # Simple strategy:
    #   - If "Father's Name" label found: names before it = cardholder, names after it = father
    #   - If only "Name" label found: first name after it = cardholder, second = father
    #   - If no labels: first = cardholder, second = father (by Y-order)
    cardholder_name = None
    cardholder_conf = 0.0
    father_name = None
    father_conf = 0.0

    if father_label_y is not None and len(name_candidates) >= 1:
        # Split names by the "Father's Name" label
        before_father = [(t, c, y) for t, c, y in name_candidates if y < father_label_y]
        after_father = [(t, c, y) for t, c, y in name_candidates if y >= father_label_y]

        # Cardholder = last name before father label (closest to label from above)
        if before_father:
            cardholder_name = normalize_name(before_father[-1][0])
            cardholder_conf = before_father[-1][1]
        # Father = first name after father label
        if after_father:
            father_name = normalize_name(after_father[0][0])
            father_conf = after_father[0][1]
    else:
        # No father label — use Y-order: first = cardholder, second = father
        if len(name_candidates) >= 1:
            cardholder_name = normalize_name(name_candidates[0][0])
            cardholder_conf = name_candidates[0][1]
        if len(name_candidates) >= 2:
            father_name = normalize_name(name_candidates[1][0])
            father_conf = name_candidates[1][1]

    # Build field results
    fields["name"] = build_field_result(
        cardholder_name, cardholder_conf, bool(cardholder_name), config
    )
    fields["fatherName"] = build_field_result(father_name, father_conf, bool(father_name), config)

    # DOB
    if dob_text:
        normalized = normalize_date(dob_text)
        if normalized:
            is_valid, norm_val = validate_dob(normalized)
            fields["dob"] = build_field_result(norm_val, dob_conf, is_valid, config)
        else:
            fields["dob"] = build_field_result(dob_text, dob_conf, False, config)
    else:
        fields["dob"] = build_field_result(None, 0.0, False, config)

    # PAN Number
    if pan_number_text:
        normalized = normalize_pan_number(pan_number_text)
        is_valid, norm_val = validate_pan(normalized)
        fields["panNumber"] = build_field_result(norm_val, pan_number_conf, is_valid, config)
    else:
        fields["panNumber"] = build_field_result(None, 0.0, False, config)

    return fields


def _extract_aadhaar_fields_positional(
    blocks: list[TextBlock], card_height: int, config: OCRConfig
) -> dict[str, FieldResult]:
    """Extract Aadhaar card fields using positional assignment.

    Handles both standard Aadhaar cards and enrollment slips by looking
    for content patterns rather than strict Y-position bands.
    """
    fields: dict[str, FieldResult] = {}

    aadhaar_full_text = None
    aadhaar_full_conf = 0.0
    aadhaar_masked_text = None
    aadhaar_masked_conf = 0.0
    dob_text = None
    dob_conf = 0.0
    gender_text = None
    gender_conf = 0.0
    name_candidates: list[tuple[str, float, float]] = []
    guardian_candidates: list[tuple[str, float, float]] = []

    for block in blocks:
        y_pct = _get_y_percent(block, card_height)
        text = block.text.strip()
        conf = block.confidence

        if not text or conf < 0.3:
            continue

        # Check for Aadhaar number (12 digits, possibly with spaces)
        aadhaar_match = _AADHAAR_PATTERN.search(text)
        if aadhaar_match:
            candidate = aadhaar_match.group(0)
            digits = re.sub(r"\D", "", candidate)
            if len(digits) == 12:
                aadhaar_full_text = digits
                aadhaar_full_conf = conf
                continue

        # Check for partial/masked Aadhaar (only last 4 digits visible)
        # These appear as standalone 4-digit blocks (not part of dates, enrollment, etc.)
        if re.match(r"^\d{4}$", text) and conf > 0.9:
            # Avoid matching year-like numbers (1900-2025) and enrollment fragments
            num = int(text)
            if not (1900 <= num <= 2025):
                # Keep the LAST such match (Aadhaar digits are typically at the bottom)
                aadhaar_masked_text = f"XXXX XXXX {text}"
                aadhaar_masked_conf = conf
                continue

        # Check for date (DOB pattern with label prefix like "/DOB:" or "DOB:")
        text_upper = text.upper()
        date_match = _DATE_PATTERN.search(text)
        if date_match:
            dob_text = date_match.group(0)
            dob_conf = conf
            continue

        # Check for year-only (some Aadhaar cards)
        if not dob_text:
            year_match = _YEAR_PATTERN.search(text)
            if year_match and "ENROLLMENT" not in text_upper:
                year_val = int(year_match.group(0))
                if 1900 <= year_val <= 2010:
                    dob_text = year_match.group(0)
                    dob_conf = conf

        # Check for gender
        if any(g in text_upper for g in ["MALE", "FEMALE", "TRANSGENDER"]):
            gender_text = text_upper
            gender_conf = conf
            continue

        # Skip known labels and non-name content
        if any(
            label in text_upper
            for label in [
                "GOVERNMENT",
                "INDIA",
                "UNIQUE",
                "IDENTIFICATION",
                "AUTHORITY",
                "UIDAI",
                "AADHAAR",
                "DOB",
                "YEAR",
                "BIRTH",
                "VID",
                "ENROLLMENT",
                "ENROLMENT",
                "ADDRESS",
                "HELP",
                "WWW",
                "HTTP",
                "MERA",
                "TRIPURA",
                "ROAD",
                "NEAR",
                "MUNICIPALITY",
                "NAGAR",
                "AUTHENTICATION",
                "VERIFICATION",
                "SCANNING",
                "OFFLINE",
                "ONLINE",
                "PROOF",
                "IDENTITY",
                "CITIZENSHIP",
                "QR CODE",
                "XML",
                "SHOULD",
                "PRADESH",
                "MOBILE",
                "ISSUED",
                "WEST BENGAL",
                "MAHARASHTRA",
                "KARNATAKA",
                "TAMIL NADU",
                "GUJARAT",
            ]
        ):
            continue

        # Skip numeric-only or too-short blocks
        stripped_digits = re.sub(r"[\d\s/\-:.]", "", text)
        if len(stripped_digits) < 3:
            continue

        # Guardian detection (W/o, S/o, D/o, C/o patterns) — check BEFORE word-count filter
        # since guardian lines often include address info making them long
        if re.search(r"[SDWC]\s*[/\\]\s*[oO0]", text, re.IGNORECASE):
            # Extract name after the S/o, D/o prefix up to the first comma
            so_match = re.search(
                r"[SDWC]\s*[/\\]\s*[oO0]\s*[:\-]?\s*(.+)", text, re.IGNORECASE
            )
            if so_match:
                guardian_raw = so_match.group(1).split(",")[0].strip()
                if guardian_raw and len(guardian_raw) > 2:
                    guardian_candidates.append((guardian_raw, conf, y_pct))
            continue

        # Skip text that looks like a sentence (too many words = disclaimer/notice)
        words_count = len(text.split())
        if words_count > 5:
            continue

        # Name candidates (Latin script, max 4 words typical for Indian names)
        if is_latin_text(text) and len(text) > 2 and conf > 0.8:
            name_candidates.append((text, conf, y_pct))

    # Sort by Y position
    name_candidates.sort(key=lambda x: x[2])
    guardian_candidates.sort(key=lambda x: x[2])

    # Name — first high-confidence Latin name block that isn't a place name
    if name_candidates:
        cardholder_name = normalize_name(name_candidates[0][0])
        cardholder_conf = name_candidates[0][1]
        fields["name"] = build_field_result(
            cardholder_name, cardholder_conf, bool(cardholder_name), config
        )
    else:
        fields["name"] = build_field_result(None, 0.0, False, config)

    # Guardian Name
    if guardian_candidates:
        guardian_name = normalize_name(guardian_candidates[0][0])
        guardian_conf = guardian_candidates[0][1]
        fields["guardianName"] = build_field_result(
            guardian_name, guardian_conf, bool(guardian_name), config
        )
    else:
        fields["guardianName"] = build_field_result(None, 0.0, False, config)

    # DOB
    if dob_text:
        normalized = normalize_date(dob_text)
        if normalized:
            is_valid, norm_val = validate_dob(normalized)
            fields["dob"] = build_field_result(norm_val, dob_conf, is_valid, config)
        else:
            fields["dob"] = build_field_result(dob_text, dob_conf, False, config)
    else:
        fields["dob"] = build_field_result(None, 0.0, False, config)

    # Gender
    if gender_text:
        normalized = normalize_gender(gender_text)
        if normalized:
            is_valid, norm_val = validate_gender(normalized)
            fields["gender"] = build_field_result(norm_val, gender_conf, is_valid, config)
        else:
            fields["gender"] = build_field_result(None, 0.0, False, config)
    else:
        fields["gender"] = build_field_result(None, 0.0, False, config)

    # Aadhaar Number — prioritize masked (last-4) over full 12-digit
    # Most real Aadhaar cards are masked; full 12-digit matches are often
    # enrollment numbers, VIDs, or other identifiers on the slip.
    if aadhaar_masked_text:
        # Masked card — report last 4 visible digits
        fields["aadhaarNumber"] = build_field_result(
            aadhaar_masked_text, aadhaar_masked_conf, False, config
        )
    elif aadhaar_full_text:
        # Full number visible (unmasked card)
        normalized = normalize_aadhaar_number(aadhaar_full_text)
        is_valid, norm_val = validate_aadhaar(normalized)
        fields["aadhaarNumber"] = build_field_result(
            norm_val, aadhaar_full_conf, is_valid, config
        )
    else:
        fields["aadhaarNumber"] = build_field_result(None, 0.0, False, config)

    return fields
