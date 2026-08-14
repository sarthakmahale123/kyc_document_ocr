# Requirements: Real-Time OCR Auto-Fill for Indian PAN and Aadhaar Cards

## 1. Overview

Build an on-premises, CPU-only OCR pipeline that extracts structured identity
fields from photographs or scans of Indian PAN cards and Aadhaar cards, and
returns a per-field JSON payload suitable for auto-populating a KYC form.

## 2. Functional Requirements

### FR-1: Document Classification

| ID | Requirement |
|--------|-------------|
| FR-1.1 | The system SHALL classify an input image as one of: `PAN`, `AADHAAR`, or `UNKNOWN`. |
| FR-1.2 | Classification SHALL precede any OCR processing; `UNKNOWN` documents SHALL be rejected with an appropriate error. |
| FR-1.3 | Classification SHALL complete within 200 ms on target CPU hardware. |

### FR-2: Field Extraction — PAN Card

| ID | Requirement |
|--------|-------------|
| FR-2.1 | Extract **Name** (English/Latin script only). |
| FR-2.2 | Extract **Father's Name** (English/Latin script only). |
| FR-2.3 | Extract **Date of Birth** (DD/MM/YYYY). |
| FR-2.4 | Extract **PAN Number** (10-character alphanumeric). |

### FR-3: Field Extraction — Aadhaar Card

| ID | Requirement |
|--------|-------------|
| FR-3.1 | Extract **Name** (English/Latin script only). |
| FR-3.2 | Extract **Date of Birth** (DD/MM/YYYY or YYYY for year-of-birth-only cards). |
| FR-3.3 | Extract **Aadhaar Number** (12-digit numeric, may be printed as XXXX XXXX XXXX). |
| FR-3.4 | Extract **Gender** (MALE / FEMALE / TRANSGENDER) if present on front face. |
| FR-3.5 | Extract **Guardian Name** (S/o, D/o, W/o, C/o prefix + name) in English/Latin script. Field key: `guardianName`. |

### FR-4: Multi-Script Name Selection

| ID | Requirement |
|--------|-------------|
| FR-4.1 | The system SHALL detect that a name zone contains text in multiple scripts. |
| FR-4.2 | Selection of the English/Latin name block SHALL use Unicode script-range detection: a text block qualifies as English if ≥ 90% of its non-whitespace, non-punctuation characters (apostrophe, hyphen, period) fall within U+0041–U+005A and U+0061–U+007A. |
| FR-4.3 | Script selection SHALL be position-independent — it must work regardless of whether the English name appears above or below the regional-script name. |
| FR-4.4 | The regional-script block SHALL be discarded and NOT included in the output payload. |
| FR-4.5 | The system SHALL be tested against all major Indian scripts: Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Odia, and Punjabi/Gurmukhi. |

### FR-5: Field Validation

| ID | Requirement |
|--------|-------------|
| FR-5.1 | PAN Number SHALL match regex `^[A-Z]{5}[0-9]{4}[A-Z]$`. |
| FR-5.2 | Aadhaar Number SHALL pass Verhoeff checksum validation after stripping spaces. |
| FR-5.3 | Date of Birth SHALL be parseable and within a plausible range (1900–present year, age ≥ 0). |
| FR-5.4 | Fields failing validation OR falling below the confidence threshold SHALL be flagged with `requiresReview: true`. |
| FR-5.5 | Fields SHALL never be silently auto-filled if validation or confidence checks fail. |
| FR-5.6 | Default confidence threshold: **0.85**. Configurable via environment variable or config file. |

### FR-6: Output Format

| ID | Requirement |
|--------|-------------|
| FR-6.1 | Output SHALL be a JSON object with a top-level `documentType` field. |
| FR-6.2 | Each extracted field SHALL be represented as: `{ "value": <string>, "confidence": <float 0–1>, "requiresReview": <bool> }`. |
| FR-6.3 | The system SHALL return all expected fields for the detected document type, even if extraction failed (with `value: null` and `requiresReview: true`). |
| FR-6.4 | Include `processingTimeMs` in the response for observability. |

Example output — PAN:

```json
{
  "documentType": "PAN",
  "fields": {
    "name": { "value": "RAJESH KUMAR", "confidence": 0.97, "requiresReview": false },
    "fatherName": { "value": "SURESH KUMAR", "confidence": 0.94, "requiresReview": false },
    "dob": { "value": "15/06/1985", "confidence": 0.99, "requiresReview": false },
    "panNumber": { "value": "ABCDE1234F", "confidence": 0.99, "requiresReview": false }
  },
  "processingTimeMs": 1820
}
```

Example output — Aadhaar:

```json
{
  "documentType": "AADHAAR",
  "fields": {
    "name": { "value": "PRIYA SHARMA", "confidence": 0.96, "requiresReview": false },
    "guardianName": { "value": "RAMESH SHARMA", "confidence": 0.91, "requiresReview": false },
    "dob": { "value": "22/03/1992", "confidence": 0.98, "requiresReview": false },
    "gender": { "value": "FEMALE", "confidence": 0.99, "requiresReview": false },
    "aadhaarNumber": { "value": "234567891234", "confidence": 0.99, "requiresReview": false }
  },
  "processingTimeMs": 2150
}
```

### FR-7: Benchmark / Evaluation Harness

| ID | Requirement |
|--------|-------------|
| FR-7.1 | Include a benchmark harness that measures field-level accuracy against a labeled test set. |
| FR-7.2 | Report per-field accuracy (exact-match and character-error-rate for identifiers). |
| FR-7.3 | Report p50 and p95 end-to-end latency. |
| FR-7.4 | Provide a baseline results file so future pipeline changes can be compared. |
| FR-7.5 | Include a small synthetic/sample test set for development; real labeled data to be added later. |

### FR-8: Input Handling

| ID | Requirement |
|--------|-------------|
| FR-8.1 | The system SHALL accept static file uploads (JPEG, PNG). |
| FR-8.2 | The system SHALL accept camera-captured images (same formats, from a snapshot endpoint or file picker). |
| FR-8.3 | The system SHALL handle low-quality inputs gracefully: poor lighting, significant skew (> 15°), motion blur, and partial occlusion SHALL degrade to lower confidence / `requiresReview: true` rather than crash. |
| FR-8.4 | Automatic deskew/rotation correction SHALL be attempted for images with detectable skew. |
| FR-8.5 | Minimum supported resolution: 640 × 480 pixels. |

### FR-9: Batch Processing

| ID | Requirement |
|--------|-------------|
| FR-9.1 | Primary mode: single card per request. |
| FR-9.2 | Optional batch mode: accept multiple images, return an array of results. |
| FR-9.3 | Batch endpoint SHALL process images sequentially or with bounded concurrency to avoid CPU exhaustion. |

## 3. Non-Functional Requirements

### NFR-1: Performance

| ID | Requirement |
|--------|-------------|
| NFR-1.1 | End-to-end latency SHALL be under 3 seconds on CPU-only hardware (hard ceiling). |
| NFR-1.2 | Target latency is sub-1-second where achievable. |
| NFR-1.3 | The system SHALL support parallel OCR of independent card zones to minimize wall-clock time. |

### NFR-2: Accuracy

| ID | Requirement |
|--------|-------------|
| NFR-2.1 | PAN Number field-level accuracy SHALL be ≥ 98% (exact match). |
| NFR-2.2 | Aadhaar Number field-level accuracy SHALL be ≥ 98% (exact match). |
| NFR-2.3 | Name, Father's/Guardian Name, DOB fields should target ≥ 95% accuracy; lower-confidence results trigger manual review. |

### NFR-3: Privacy and Deployment

| ID | Requirement |
|--------|-------------|
| NFR-3.1 | All processing SHALL occur locally / on-premises. No cloud OCR API calls. |
| NFR-3.2 | No image data SHALL be transmitted externally. |
| NFR-3.3 | Extracted PII SHALL not be logged to disk by default (configurable for debugging). |
| NFR-3.4 | Input images SHALL not be persisted after processing unless explicitly configured. |

### NFR-4: Portability

| ID | Requirement |
|--------|-------------|
| NFR-4.1 | The system SHALL run on Linux (primary) and macOS (development). |
| NFR-4.2 | No GPU SHALL be required; GPU acceleration may be used opportunistically if available. |
| NFR-4.3 | Dependencies SHALL be installable via pip/conda without proprietary SDKs. |

### NFR-5: Security

| ID | Requirement |
|--------|-------------|
| NFR-5.1 | Input images SHALL be validated (file type, size limits, magic bytes) before processing. |
| NFR-5.2 | The API SHALL enforce rate limiting to prevent abuse. |
| NFR-5.3 | No sensitive data (PII, card images) SHALL appear in application logs at INFO level or below. |
| NFR-5.4 | Dependencies SHALL be pinned to exact versions. |
| NFR-5.5 | The system SHALL handle malformed/adversarial inputs without crashing (graceful error responses). |

### NFR-6: Resiliency

| ID | Requirement |
|--------|-------------|
| NFR-6.1 | The pipeline SHALL use timeouts at each stage to prevent indefinite hangs. |
| NFR-6.2 | Individual stage failures SHALL be isolated — a crash in OCR zone extraction SHALL not bring down the entire service. |
| NFR-6.3 | Health check endpoint SHALL be provided for readiness/liveness probing. |
| NFR-6.4 | The system SHALL degrade gracefully under load (reject with 503 rather than OOM). |

## 4. Architecture Constraints (Mandated)

| Constraint | Detail |
|------------|--------|
| No VLM full-page transcription | Do NOT use a general-purpose vision-language model for transcription. |
| Pipeline stages | Classification → boundary detection → zone cropping → parallel OCR → validation → JSON output. |
| OCR engine | PaddleOCR or Tesseract — to be determined during design based on latency/accuracy benchmarks. |
| Zone-based extraction | Exploit fixed card layouts for known-zone cropping rather than free-form text detection. |
| Script filtering | Unicode range detection, not NLP/language-ID classifiers. |

## 5. Integration Surface

| Interface | Detail |
|-----------|--------|
| CLI | `ocr-extract --input image.jpg` → JSON to stdout. Supports `--batch` for multiple files. |
| REST API | FastAPI server. `POST /extract` (single), `POST /extract/batch` (multiple). |
| Python library | Importable module: `from ocr_pipeline import extract_card`. |

## 6. Input Assumptions

| Assumption | Detail |
|------------|--------|
| Input source | Static file upload OR camera snapshot (JPEG, PNG). |
| Card side | Front side only (all target fields are on the front). |
| Image quality | Variable: good scans to poor phone captures with skew, blur, occlusion. System degrades gracefully. |
| Card variants | Standard PAN (blue laminated), new e-PAN, standard Aadhaar (front). |

## 7. Testing Extensions

- Property-based testing SHALL be applied to pure functions and serialization
  round-trips (validation logic, JSON schema conformance, Unicode filtering).
- Integration tests SHALL cover the full pipeline with representative images.

## 8. Test Data

The `pan_aadhar/` directory contains ~104 sample images:

- ~80 PAN card images (jpg, jpeg, png)
- ~24 Aadhaar card images (prefixed `PER_UID`)

A small synthetic/sample test set will be included for development. Real labeled
ground-truth data to be added incrementally.
