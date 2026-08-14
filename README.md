# KYC Document OCR Auto-Fill

Real-time OCR pipeline for extracting structured fields from Indian **PAN** and **Aadhaar** cards. Upload a card image and get Name, Father's/Guardian Name, Date of Birth, PAN/Aadhaar Number, and Gender auto-populated in a structured JSON response.

## Features

- Extracts fields from PAN cards (Name, Father's Name, DOB, PAN Number)
- Extracts fields from Aadhaar cards (Name, Guardian Name, DOB, Gender, Aadhaar Number)
- Handles masked Aadhaar cards (last 4 digits visible)
- Content-based extraction — works with any card layout or orientation
- Multi-script support — filters English/Latin names from Hindi/regional text
- Per-field confidence scores and validation
- Fields flagged `requiresReview` when uncertain (never silently wrong)
- 100% on-premises — no cloud API calls, no data leaves your network
- Web UI with drag-and-drop upload

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| OCR Engine | PaddleOCR v3.7 (PP-OCRv6 medium models) |
| Image Processing | OpenCV, Pillow |
| Web Framework | FastAPI + Uvicorn |
| Validation | Verhoeff checksum (Aadhaar), regex (PAN) |
| Configuration | Pydantic Settings |
| Frontend | Single-page HTML/CSS/JS (no build step) |

## Prerequisites

- Python 3.11 or 3.12
- Tesseract OCR (optional fallback): `brew install tesseract` (macOS) or `apt install tesseract-ocr` (Linux)

## Setup

```bash
# Clone the project
cd ocr_project

# Create virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -e ".[dev]"

# Install PaddleOCR (recommended for best accuracy)
pip install paddlepaddle paddleocr paddlex
```

On first run, PaddleOCR will download ~136 MB of models to `~/.paddlex/official_models/`.

## Running

### Web UI (recommended for demo)

```bash
source .venv/bin/activate
uvicorn ocr_pipeline.server:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser. Drag and drop a PAN or Aadhaar card image.

### CLI

```bash
# Single image
ocr-extract pan_aadhar/PAN1_1_0.jpg

# Batch (all images in a directory)
ocr-extract pan_aadhar/ --batch

# With options
ocr-extract image.jpg --confidence 0.90 --engine paddle --format json
```

### Python Library

```python
from ocr_pipeline import extract_card, OCRConfig

config = OCRConfig(engine="paddle", confidence_threshold=0.85)
result = extract_card("path/to/card.jpg", config=config)

print(result["documentType"])  # "PAN" or "AADHAAR"
print(result["fields"]["name"]["value"])  # "RAJESH KUMAR"
print(result["fields"]["panNumber"]["confidence"])  # 0.9999
```

### REST API

```bash
# Single extraction
curl -X POST http://localhost:8000/extract \
  -F "file=@pan_card.jpg"

# Batch extraction
curl -X POST http://localhost:8000/extract/batch \
  -F "files=@card1.jpg" -F "files=@card2.jpg"

# Health check
curl http://localhost:8000/health
```

## Output Format

```json
{
  "documentType": "PAN",
  "fields": {
    "name": { "value": "RAJESH KUMAR", "confidence": 0.99, "requiresReview": false },
    "fatherName": { "value": "SURESH KUMAR", "confidence": 0.98, "requiresReview": false },
    "dob": { "value": "15/06/1985", "confidence": 1.0, "requiresReview": false },
    "panNumber": { "value": "ABCDE1234F", "confidence": 1.0, "requiresReview": false }
  },
  "processingTimeMs": 2540.3
}
```

For masked Aadhaar cards:

```json
{
  "documentType": "AADHAAR",
  "fields": {
    "name": { "value": "SUDEEP DUBEY", "confidence": 0.99, "requiresReview": false },
    "guardianName": { "value": "BHIM DUBEY", "confidence": 0.98, "requiresReview": false },
    "dob": { "value": "30/09/2000", "confidence": 0.99, "requiresReview": false },
    "gender": { "value": "MALE", "confidence": 0.99, "requiresReview": false },
    "aadhaarNumber": { "value": "XXXX XXXX 3160", "confidence": 1.0, "requiresReview": true }
  },
  "processingTimeMs": 4200.1
}
```

## Pipeline Architecture

```
Image → Preprocessing → OCR (PaddleOCR, single pass) → Content-Based Field Extraction → Validation → JSON
```

1. **Input Validation** — magic bytes check, size limit, EXIF orientation
2. **Preprocessing** — deskew (Hough lines), CLAHE contrast, bilateral denoise
3. **OCR** — PaddleOCR runs once on the full image (orientation detection + text detection + recognition)
4. **Classification** — determined from OCR content: "INCOME TAX" → PAN, "Government of India" + 4-digit block → Aadhaar
5. **Field Extraction** — content-based (regex for PAN/dates, Latin-script filter for names, S/o pattern for guardians)
6. **Validation** — PAN regex, Aadhaar Verhoeff checksum, DOB plausibility
7. **Response** — structured JSON with confidence and review flags

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OCR_CONFIDENCE_THRESHOLD` | `0.85` | Below this → `requiresReview: true` |
| `OCR_ENGINE` | `auto` | `paddle`, `tesseract`, or `auto` |
| `OCR_MAX_IMAGE_SIZE_MB` | `10` | Max upload size |
| `OCR_PARALLEL_WORKERS` | `4` | Thread pool size |
| `OCR_ZONE_TIMEOUT_S` | `5.0` | Per-zone OCR timeout |
| `OCR_LOG_PII` | `false` | Log extracted values (debug only) |
| `OCR_HOST` | `0.0.0.0` | Server bind address |
| `OCR_PORT` | `8000` | Server port |

## Project Structure

```
ocr_pipeline/
├── __init__.py          # Public API: extract_card, extract_batch
├── pipeline.py          # Main orchestrator — field extraction logic
├── ocr_engine.py        # PaddleOCR/Tesseract wrapper
├── preprocessing.py     # Deskew, CLAHE, denoise
├── input_gateway.py     # File validation, decode
├── classifier.py        # Heuristic classifier (fallback)
├── zone_mapper.py       # Card boundary detection
├── script_filter.py     # Unicode Latin-range filtering
├── postprocessing.py    # Name/date normalization
├── validators.py        # PAN regex, Aadhaar Verhoeff, DOB
├── response.py          # JSON response assembly
├── config.py            # Pydantic settings
├── server.py            # FastAPI REST API
├── static/index.html    # Web UI
└── models/              # ONNX model storage (optional)

cli/main.py              # Click CLI entry point
benchmark/               # Evaluation harness
tests/                   # Unit + property-based tests
```

## Performance

| Metric | Value |
|--------|-------|
| PAN card (warm) | 2-4 seconds |
| Aadhaar card (warm) | 3-6 seconds |
| Cold start overhead | +2-3 seconds (model loading, once) |
| PAN number accuracy | ~95%+ |
| DOB accuracy | ~98%+ |
| Model size (on disk) | ~136 MB |

## Limitations

- Heavily damaged/blurry cards may return null fields (flagged for review)
- Aadhaar numbers on masked cards only show last 4 visible digits
- Some old-style PAN cards with very small text may not have PAN number detected
- First request is slower due to model loading (~5-6s total)

## License

MIT-0
