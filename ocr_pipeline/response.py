"""Response assembly: constructs the structured JSON output.

Builds per-field responses with value, confidence, and requiresReview flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ocr_pipeline.classifier import DocumentType
from ocr_pipeline.config import OCRConfig


@dataclass
class FieldResult:
    """Result for a single extracted field."""

    value: str | None
    confidence: float
    requires_review: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "value": self.value,
            "confidence": round(self.confidence, 4),
            "requiresReview": self.requires_review,
        }


@dataclass
class ExtractionResult:
    """Complete extraction result for a document."""

    document_type: DocumentType
    fields: dict[str, FieldResult] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result: dict[str, Any] = {
            "documentType": self.document_type.value,
            "fields": {name: f.to_dict() for name, f in self.fields.items()},
            "processingTimeMs": round(self.processing_time_ms, 1),
        }
        if self.error:
            result["error"] = self.error
        return result


# Expected fields per document type
PAN_FIELDS = ["name", "fatherName", "dob", "panNumber"]
AADHAAR_FIELDS = ["name", "guardianName", "dob", "gender", "aadhaarNumber"]


def build_field_result(
    value: str | None,
    confidence: float,
    is_valid: bool,
    config: OCRConfig | None = None,
) -> FieldResult:
    """Build a FieldResult with appropriate requiresReview flag.

    A field requires review if:
    - value is None (extraction failed)
    - confidence is below threshold
    - validation failed

    Args:
        value: Extracted field value (or None).
        confidence: OCR confidence score (0-1).
        is_valid: Whether the value passed format validation.
        config: Pipeline configuration for threshold.

    Returns:
        FieldResult with requiresReview appropriately set.
    """
    if config is None:
        config = OCRConfig()

    threshold = config.confidence_threshold

    requires_review = value is None or confidence < threshold or not is_valid

    return FieldResult(
        value=value,
        confidence=confidence if value is not None else 0.0,
        requires_review=requires_review,
    )


def assemble_response(
    doc_type: DocumentType,
    extracted_fields: dict[str, FieldResult],
    processing_time_ms: float,
) -> ExtractionResult:
    """Assemble the final extraction response.

    Ensures all expected fields for the document type are present,
    even if extraction failed (with null value and requiresReview=true).

    Args:
        doc_type: Detected document type.
        extracted_fields: Dictionary of successfully extracted fields.
        processing_time_ms: Total processing time in milliseconds.

    Returns:
        Complete ExtractionResult.
    """
    expected = PAN_FIELDS if doc_type == DocumentType.PAN else AADHAAR_FIELDS

    fields: dict[str, FieldResult] = {}
    for field_name in expected:
        if field_name in extracted_fields:
            fields[field_name] = extracted_fields[field_name]
        else:
            # Field not extracted — mark as requiring review
            fields[field_name] = FieldResult(value=None, confidence=0.0, requires_review=True)

    return ExtractionResult(
        document_type=doc_type,
        fields=fields,
        processing_time_ms=processing_time_ms,
    )
