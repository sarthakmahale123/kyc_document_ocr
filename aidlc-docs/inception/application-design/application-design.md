# Application Design: PAN/Aadhaar OCR Auto-Fill System

## 1. High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                    │
│  CLI (click)  │  REST API (FastAPI)  │  Python Library (importable)     │
└───────┬───────┴──────────┬───────────┴──────────┬───────────────────────┘
        │                  │                      │
        ▼                  ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       INPUT GATEWAY                                      │
│  • File-type validation (magic bytes + extension)                        │
│  • Size limit enforcement (max 10 MB)                                    │
│  • Image decode + normalize to RGB numpy array                           │
│  • Rate limiting (API mode only)                                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   STAGE 1: PREPROCESSING                                 │
│  • Auto-rotation (EXIF orientation)                                      │
│  • Deskew detection + correction (Hough transform, < 45°)                │
│  • Adaptive contrast enhancement (CLAHE)                                 │
│  • Noise reduction (bilateral filter for edge preservation)              │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               STAGE 2: DOCUMENT CLASSIFICATION                           │
│  • Lightweight CNN classifier (MobileNetV3-Small, ~5 MB)                 │
│  • Classes: PAN / AADHAAR / UNKNOWN                                      │
│  • Target: < 100 ms inference on CPU                                     │
│  • Fallback: template-matching heuristic if model unavailable            │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           STAGE 3: CARD BOUNDARY DETECTION + ZONE MAPPING                │
│  • Contour detection → largest quadrilateral → perspective transform     │
│  • Map to normalized card coordinates (standard aspect ratio)            │
│  • Apply card-type-specific zone template (see §3)                       │
│  • Output: dict of named ROI crops (numpy arrays)                        │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STAGE 4: PARALLEL OCR PER ZONE                              │
│  • ThreadPoolExecutor (bounded concurrency = 4)                          │
│  • Each zone → OCR engine → raw text + per-char confidence              │
│  • Engine selection: PaddleOCR (primary) or Tesseract (fallback)         │
│  • Per-zone timeout: 2 seconds                                           │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         STAGE 5: POST-PROCESSING + SCRIPT FILTERING                      │
│  • Unicode Latin-range filter (≥ 90% Latin → English name)               │
│  • Guardian prefix extraction (S/o, D/o, W/o, C/o)                       │
│  • Whitespace normalization, case normalization                           │
│  • Date format unification → DD/MM/YYYY                                  │
│  • Number normalization (strip spaces from Aadhaar)                      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STAGE 6: VALIDATION + CONFIDENCE SCORING                     │
│  • PAN regex: ^[A-Z]{5}[0-9]{4}[A-Z]$                                   │
│  • Aadhaar: 12-digit + Verhoeff checksum                                 │
│  • DOB: parseable + plausible range (1900–current year)                  │
│  • Confidence threshold check (default 0.85, configurable)               │
│  • Flag requiresReview for failures                                      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              STAGE 7: STRUCTURED JSON OUTPUT                              │
│  • Assemble per-field response objects                                    │
│  • Include processingTimeMs                                              │
│  • Return all expected fields (null + requiresReview if missing)         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Component Breakdown

### 2.1 Input Gateway (`ocr_pipeline/input_gateway.py`)

| Responsibility | Implementation |
|----------------|----------------|
| File validation | python-magic for MIME type; reject non-image types |
| Size enforcement | Max 10 MB configurable via `MAX_IMAGE_SIZE_BYTES` |
| Image decode | OpenCV `imdecode` from bytes buffer |
| Rate limiting | `slowapi` middleware (API mode); no-op for CLI/library |

### 2.2 Preprocessor (`ocr_pipeline/preprocessing.py`)

| Responsibility | Implementation |
|----------------|----------------|
| EXIF rotation | `PIL.ImageOps.exif_transpose` |
| Deskew | Hough line detection → compute median angle → `cv2.warpAffine` |
| Contrast | CLAHE on L-channel of LAB color space |
| Denoise | `cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)` |

Design note: Each preprocessing step is a standalone function. The pipeline
applies them in sequence, but individual steps can be disabled via config for
benchmarking/debugging.

### 2.3 Document Classifier (`ocr_pipeline/classifier.py`)

| Approach | Details |
|----------|---------|
| Primary | MobileNetV3-Small fine-tuned on PAN/Aadhaar/Other (3-class). ONNX Runtime inference for CPU speed. |
| Fallback | Template matching: detect Income Tax logo (PAN) or UIDAI emblem (Aadhaar) via ORB keypoints. |
| Training data | The ~104 samples + augmented copies (rotation, brightness, crop). |
| Model size | ~5 MB (ONNX quantized INT8). |
| Latency target | < 100 ms. |

### 2.4 Boundary Detector + Zone Mapper (`ocr_pipeline/zone_mapper.py`)

**Card boundary detection:**

1. Convert to grayscale → Gaussian blur → Canny edges
2. Find contours → filter by area (> 20% of image) → approximate polygon
3. If 4-point quadrilateral found → perspective warp to standard rectangle
4. If no quad found → assume card fills frame, apply margin crop

**Zone templates** (relative coordinates as % of card width/height):

PAN Card zones:

```text
┌──────────────────────────────────────────┐
│  INCOME TAX DEPARTMENT / GOVT OF INDIA   │  (header — skip)
│                                          │
│  ┌──────┐  Name Zone (regional)         │
│  │PHOTO │  Name Zone (English)           │
│  │      │  Father's Name (regional)      │
│  │      │  Father's Name (English)       │
│  └──────┘  Date of Birth                 │
│                                          │
│            PAN NUMBER                     │
│                                          │
└──────────────────────────────────────────┘
```

| Zone | Top % | Left % | Width % | Height % |
|------|-------|--------|---------|----------|
| name_block | 22 | 30 | 68 | 22 |
| father_name_block | 44 | 30 | 68 | 14 |
| dob_zone | 58 | 30 | 40 | 10 |
| pan_number_zone | 72 | 20 | 60 | 12 |

Aadhaar Card zones:

```text
┌──────────────────────────────────────────┐
│  GOVT OF INDIA / UIDAI HEADER            │  (header — skip)
│                                          │
│  ┌──────┐  Name (regional + English)     │
│  │PHOTO │  Guardian (S/o, D/o...)        │
│  │      │  DOB / Year of Birth           │
│  │      │  Gender                         │
│  └──────┘                                │
│                                          │
│         XXXX XXXX XXXX                   │
│                                          │
└──────────────────────────────────────────┘
```

| Zone | Top % | Left % | Width % | Height % |
|------|-------|--------|---------|----------|
| name_block | 28 | 32 | 66 | 14 |
| guardian_block | 42 | 32 | 66 | 10 |
| dob_gender_zone | 52 | 32 | 50 | 12 |
| aadhaar_number_zone | 72 | 15 | 70 | 14 |

**Adaptive zone refinement:** If OCR on a zone returns empty/garbage, expand the
zone by 10% in each direction and retry once. This handles minor layout
variations across card issuance years.

### 2.5 OCR Engine Wrapper (`ocr_pipeline/ocr_engine.py`)

```text
┌─────────────────────────────────────┐
│         OCREngine (Protocol)         │
│  • recognize(image: ndarray)         │
│    → list[TextBlock(text, conf, box)]│
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐  ┌──────────┐
│Paddle  │  │Tesseract │
│Backend │  │Backend   │
└────────┘  └──────────┘
```

| Engine | Pros | Cons |
|--------|------|------|
| PaddleOCR | Higher accuracy on Indic text, good angle tolerance | Heavier (~150 MB models), slower cold start |
| Tesseract | Lightweight, fast, widely available | Lower accuracy on degraded images, needs lang packs |

**Strategy:** Benchmark both during design validation. Use PaddleOCR (PP-OCRv4
mobile) as primary. If latency exceeds budget, fall back to Tesseract for
non-critical zones (name, DOB) while keeping PaddleOCR for identifier zones
(PAN number, Aadhaar number) where 98% accuracy is mandatory.

### 2.6 Script Filter (`ocr_pipeline/script_filter.py`)

```python
def is_latin_text(text: str, threshold: float = 0.90) -> bool:
    """
    Returns True if >= threshold fraction of significant characters
    are in the Basic Latin alphabet range (A-Z, a-z).
    Strips whitespace and allowed punctuation (', -, .) before counting.
    """
    stripped = re.sub(r"[\s'\-.]", "", text)
    if not stripped:
        return False
    latin_count = sum(1 for c in stripped if '\u0041' <= c <= '\u005A' or '\u0061' <= c <= '\u007A')
    return (latin_count / len(stripped)) >= threshold
```

Applied to name zones: OCR returns multiple text lines → filter each line →
select the one(s) that pass `is_latin_text` → join as the English name.

### 2.7 Validators (`ocr_pipeline/validators.py`)

| Validator | Logic |
|-----------|-------|
| `validate_pan(s)` | Regex `^[A-Z]{5}[0-9]{4}[A-Z]$` after strip + upper |
| `validate_aadhaar(s)` | Strip spaces → 12 digits → Verhoeff checksum |
| `validate_dob(s)` | Parse DD/MM/YYYY or YYYY → check 1900 ≤ year ≤ current |
| `validate_gender(s)` | Must be in {"MALE", "FEMALE", "TRANSGENDER"} |

Verhoeff checksum implementation included as a self-contained module (no
external dependency) following the standard multiplication/permutation tables.

### 2.8 Response Assembler (`ocr_pipeline/response.py`)

Constructs the final JSON. Logic:

1. For each expected field (based on `documentType`):
   - If extracted and validated and confidence ≥ threshold → normal field
   - If extracted but validation fails → `requiresReview: true`
   - If extracted but confidence < threshold → `requiresReview: true`
   - If not extracted (OCR returned nothing) → `value: null, requiresReview: true`
2. Attach `processingTimeMs` from wall-clock timer started at input receipt.

## 3. API Design

### 3.1 REST Endpoints (FastAPI)

```text
POST /extract
  Content-Type: multipart/form-data
  Body: file=<image>
  Response: 200 → ExtractionResult JSON
            400 → {"error": "invalid_input", "detail": "..."}
            422 → {"error": "unknown_document", "detail": "..."}
            503 → {"error": "overloaded", "detail": "..."}

POST /extract/batch
  Content-Type: multipart/form-data
  Body: files=<image1>&files=<image2>&...
  Response: 200 → {"results": [ExtractionResult, ...]}

GET /health
  Response: 200 → {"status": "healthy", "version": "...", "engine": "..."}
```

### 3.2 CLI Interface

```text
Usage: ocr-extract [OPTIONS] INPUT_PATH

Options:
  --output, -o       Output file (default: stdout)
  --batch            Treat INPUT_PATH as directory, process all images
  --confidence       Confidence threshold override (default: 0.85)
  --engine           OCR engine: paddle|tesseract|auto (default: auto)
  --format           Output format: json|jsonl (default: json)
  --verbose, -v      Enable debug logging
  --help             Show help
```

### 3.3 Python Library

```python
from ocr_pipeline import extract_card, extract_batch, OCRConfig

config = OCRConfig(confidence_threshold=0.85, engine="auto")
result = extract_card("path/to/pan.jpg", config=config)
# result is a dict matching the JSON schema

results = extract_batch(["img1.jpg", "img2.jpg"], config=config)
```

## 4. Data Flow Diagram

```text
Image bytes
    │
    ▼
[Input Gateway] ──invalid──▶ Error Response (400)
    │ valid
    ▼
[Preprocessor] ──▶ normalized image (numpy array)
    │
    ▼
[Classifier] ──UNKNOWN──▶ Error Response (422)
    │ PAN or AADHAAR
    ▼
[Zone Mapper] ──▶ dict[zone_name → cropped_image]
    │
    ▼
[OCR Pool] ──parallel──▶ dict[zone_name → TextBlock[]]
    │
    ▼
[Script Filter + Post-process] ──▶ dict[field_name → (text, confidence)]
    │
    ▼
[Validators] ──▶ dict[field_name → (text, confidence, valid: bool)]
    │
    ▼
[Response Assembler] ──▶ ExtractionResult JSON
```

## 5. Technology Stack

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| Language | Python | 3.11+ | Ecosystem compatibility with all OCR libraries |
| Image processing | OpenCV | 4.9+ | Industry standard, fast CPU ops |
| OCR (primary) | PaddleOCR | 2.7+ | Best accuracy on Indic scripts, mobile models available |
| OCR (fallback) | Tesseract + pytesseract | 5.x | Lightweight fallback, widely deployed |
| ML inference | ONNX Runtime | 1.17+ | Fast CPU inference for classifier |
| REST framework | FastAPI | 0.110+ | Async, fast, auto-docs |
| ASGI server | Uvicorn | 0.29+ | Production-grade ASGI |
| CLI framework | Click | 8.x | Clean CLI construction |
| Validation | Pydantic | 2.x | JSON schema, type safety |
| Testing | pytest + hypothesis | latest | Unit + property-based tests |
| Benchmark | Custom harness | — | Field accuracy + latency measurement |
| Package mgmt | uv | latest | Fast, reproducible installs |

## 6. Project Structure

```text
ocr_pipeline/
├── __init__.py              # Public API: extract_card, extract_batch, OCRConfig
├── input_gateway.py         # File validation, decode, rate limit
├── preprocessing.py         # Deskew, CLAHE, denoise, EXIF
├── classifier.py            # Document type classification
├── zone_mapper.py           # Boundary detection + zone templates
├── ocr_engine.py            # OCR protocol + PaddleOCR/Tesseract backends
├── script_filter.py         # Unicode Latin-range filtering
├── postprocessing.py        # Text cleanup, date normalization
├── validators.py            # PAN regex, Aadhaar Verhoeff, DOB check
├── response.py              # JSON assembly
├── config.py                # OCRConfig, thresholds, env vars
├── models/                  # ONNX classifier model + zone templates
│   ├── classifier.onnx
│   ├── pan_zones.json
│   └── aadhaar_zones.json
└── server.py                # FastAPI app

cli/
├── __init__.py
└── main.py                  # Click CLI entry point

benchmark/
├── __init__.py
├── harness.py               # Evaluation runner
├── metrics.py               # Accuracy + latency metrics
├── test_data/               # Labeled synthetic test set
│   ├── images/
│   └── ground_truth.json
└── baseline_results.json    # Last-known-good results

tests/
├── unit/
│   ├── test_script_filter.py
│   ├── test_validators.py
│   ├── test_preprocessing.py
│   └── test_postprocessing.py
├── integration/
│   ├── test_pipeline_pan.py
│   └── test_pipeline_aadhaar.py
└── property/
    ├── test_validators_pbt.py
    └── test_script_filter_pbt.py
```

## 7. Latency Budget

Target: < 3000 ms total. Aspirational: < 1000 ms.

| Stage | Budget (ms) | Notes |
|-------|-------------|-------|
| Input validation + decode | 20 | Trivial I/O |
| Preprocessing | 80 | CLAHE + deskew on ~2 MP image |
| Classification | 100 | MobileNetV3-Small ONNX INT8 |
| Boundary + zone crop | 50 | OpenCV contour ops |
| OCR (parallel, 4 zones) | 600–2000 | Dominant cost; PaddleOCR mobile ~500 ms/zone, parallelized to ~600 ms |
| Post-processing + filter | 10 | String ops |
| Validation | 5 | Regex + Verhoeff |
| Response assembly | 5 | Dict construction |
| **Total (optimistic)** | **~870 ms** | Parallel OCR on fast CPU |
| **Total (pessimistic)** | **~2270 ms** | Slower CPU or degraded image requiring retries |

Key optimization levers:
- Use PaddleOCR PP-OCRv4 mobile (server model is 3× slower)
- Parallel zone OCR via ThreadPoolExecutor
- ONNX INT8 quantization for classifier
- Downscale zones to optimal OCR input size (reduce unnecessary resolution)
- Lazy model loading with warm-up on first request

## 8. Error Handling Strategy

| Scenario | Behavior |
|----------|----------|
| Invalid file type | Reject at gateway with 400 + descriptive message |
| Image too large | Reject at gateway with 400 |
| Unrecognized document | Return 422 with `documentType: "UNKNOWN"` |
| OCR zone timeout (2s) | Mark that field as `null, requiresReview: true` |
| OCR engine crash | Catch exception, fall back to other engine for that zone |
| All zones fail | Return partial result with all fields `requiresReview: true` |
| Server overloaded | Return 503 (backpressure via semaphore-limited concurrency) |

## 9. Configuration

Environment variables (also loadable from `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_CONFIDENCE_THRESHOLD` | `0.85` | Below this → requiresReview |
| `OCR_ENGINE` | `auto` | `paddle`, `tesseract`, or `auto` |
| `OCR_MAX_IMAGE_SIZE_MB` | `10` | Reject images larger than this |
| `OCR_PARALLEL_WORKERS` | `4` | Thread pool size for zone OCR |
| `OCR_ZONE_TIMEOUT_S` | `2.0` | Per-zone OCR timeout |
| `OCR_LOG_LEVEL` | `INFO` | Logging level |
| `OCR_LOG_PII` | `false` | If true, log extracted field values (debug only) |
| `OCR_RATE_LIMIT` | `30/minute` | API rate limit per client |
| `OCR_HOST` | `0.0.0.0` | Server bind address |
| `OCR_PORT` | `8000` | Server bind port |

## 10. Security Considerations

1. **Input sanitization:** Validate magic bytes, reject polyglot files.
2. **No PII in logs:** Extracted values redacted at INFO level; only shown at DEBUG
   when `OCR_LOG_PII=true`.
3. **Dependency pinning:** All packages pinned to exact versions in `uv.lock`.
4. **No network egress:** Pipeline makes zero outbound connections; enforce via
   process sandboxing in production.
5. **Temp file cleanup:** All intermediate image buffers are in-memory numpy arrays;
   no temp files written to disk.
6. **Rate limiting:** `slowapi` with configurable limits prevents DoS.

## 11. Benchmark Harness Design

```text
benchmark/
├── harness.py          # Orchestrator
├── metrics.py          # Computation
├── test_data/
│   ├── images/         # Labeled test images
│   └── ground_truth.json
└── baseline_results.json
```

**ground_truth.json schema:**

```json
[
  {
    "filename": "pan_sample_01.jpg",
    "documentType": "PAN",
    "fields": {
      "name": "RAJESH KUMAR",
      "fatherName": "SURESH KUMAR",
      "dob": "15/06/1985",
      "panNumber": "ABCDE1234F"
    }
  }
]
```

**Metrics reported:**

| Metric | Scope |
|--------|-------|
| Exact match accuracy | Per field type |
| Character error rate (CER) | PAN number, Aadhaar number |
| Field extraction rate | % of fields successfully extracted (non-null) |
| p50 latency | End-to-end |
| p95 latency | End-to-end |
| Classification accuracy | Document type |

**Usage:**

```bash
# Run benchmark
python -m benchmark.harness --data benchmark/test_data --output results.json

# Compare against baseline
python -m benchmark.harness --data benchmark/test_data --baseline benchmark/baseline_results.json
```

## 12. Constraints Compliance Matrix

| Constraint | How Addressed |
|------------|---------------|
| No VLM | Pipeline uses lightweight CNN classifier + traditional OCR engines |
| Zone-based extraction | Fixed zone templates per card type with adaptive expansion |
| CPU-only | All models selected for CPU performance; GPU used opportunistically |
| < 3s latency | Latency budget shows ~870–2270 ms range; parallel OCR is key |
| 98% accuracy on IDs | Dedicated high-resolution zones for PAN/Aadhaar numbers; PaddleOCR for these zones |
| No cloud calls | All processing local; no network dependencies |
| Unicode script filter | `is_latin_text()` uses character-range detection, not NLP |
| Privacy | No logging of PII; no temp files; no egress |
| Structured JSON output | Pydantic models enforce schema; per-field confidence + requiresReview |
