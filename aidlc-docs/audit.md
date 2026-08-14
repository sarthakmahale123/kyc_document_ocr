# AI-DLC Audit Log

## Workspace Detection
**Timestamp**: 2026-08-12T00:00:00Z
**User Input**: "lets start aidlc"
**AI Response**: Displayed AI-DLC welcome message and began workspace detection.
**Context**: Initial workflow start.

---

## Workspace Detection - User Clarification
**Timestamp**: 2026-08-12T00:01:00Z
**User Input**: "Remove all packages referring to kyc, I am building something new from scratch"
**AI Response**: Removed app.py, configs/global.yaml, cleaned pyproject.toml and .gitignore of all KYC references. Workspace is now a blank slate.
**Context**: User clarified this is a greenfield project, not brownfield.

---

## Workspace Detection - Completion Confirmed
**Timestamp**: 2026-08-12T00:02:00Z
**User Input**: "Is it done"
**AI Response**: Confirmed cleanup complete. Offered to continue AI-DLC workflow.
**Context**: Workspace detection effectively complete — greenfield confirmed.

---

## Requirements Analysis - Initial User Request
**Timestamp**: 2026-08-12T00:03:00Z
**User Input**: "Build a real-time OCR auto-fill system for Indian PAN and Aadhaar cards.

Context:

- Use case: user uploads/photographs a PAN or Aadhaar card, system extracts Name, DOB, PAN/Aadhaar number, and Father's Name, and auto-populates a form
- Hard requirement: end-to-end latency under 3 seconds on CPU-only hardware (no GPU guaranteed in production), sub-1-second is the target if achievable
- Accuracy requirement: 98%+ field-level accuracy on PAN number and Aadhaar number specifically, since these are high-stakes identifiers; other fields can tolerate slightly lower confidence with manual-review fallback

Architecture constraints:
- Do NOT use a general-purpose vision-language model for full-page transcription — architecture should be: document classification → card boundary detection → known-zone cropping (PAN/Aadhaar have fixed layouts) → parallel OCR per zone using a lightweight engine (PaddleOCR or Tesseract) → field validation → structured JSON output with per-field confidence scores

Multi-script name handling:
- PAN and Aadhaar cards print the name in both a regional script (Devanagari, Tamil, Telugu, Bengali, Gujarati, etc. depending on issuing state) and in English/Latin script. The system must select only the English/Latin-script name for the auto-filled field, discarding the regional-script block.
- Use Unicode-script-range detection to identify the Latin-script block (not language-ID/NLP classification, which is unreliable on short proper nouns) — a text block counts as the English name if roughly ≥90% of its characters fall in the basic Latin alphabet range, after stripping whitespace and common name punctuation (apostrophes, hyphens, periods).
- This selection must work regardless of which line/position the English name appears in relative to the regional-script name, since card layouts vary.

Validation:
- PAN format: regex ^[A-Z]{5}[0-9]{4}[A-Z]$
- Aadhaar format: 12 digits + Verhoeff checksum validation
- DOB: parseable date, plausible range
- Fields that fail validation or fall below the confidence threshold must be flagged with requiresReview: true, not silently auto-filled

Output:
- Structured JSON per field: { value, confidence, requiresReview }, not a flat text dump

Other requirements:
- Deployment target: local/on-prem, no cloud OCR API calls (data privacy requirement for KYC documents)
- Include a benchmark/eval harness against a small labeled test set to measure field-level accuracy and p50/p95 latency, so future pipeline changes can be measured against a baseline rather than guessed at

Please generate requirements.md first and ask clarifying questions about anything ambiguous (e.g., input source — camera stream vs. static upload, expected image quality range, whether both card sides need scanning, which Indian regional scripts must be explicitly supported) before proceeding to design."
**AI Response**: Proceeding to create clarifying questions file for requirements analysis.
**Context**: INCEPTION - Requirements Analysis starting. User provided comprehensive requirements with explicit request for clarifying questions before proceeding.

---
