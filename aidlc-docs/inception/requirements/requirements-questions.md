# Requirements Clarification Questions

Please answer the following questions to help clarify the requirements. Fill in the letter choice after each `[Answer]:` tag. If none of the options match, choose the last option (Other) and describe your preference.

---

## Question 1
What is the primary input source for card images?

A) Static file upload only (user selects a file from disk)

B) Camera capture only (live viewfinder → snapshot)

C) Both static upload and camera capture

D) Other (please describe after [Answer]: tag below)

[Answer]: C

---

## Question 2
What image quality range should the system handle?

A) High quality only — scanned documents at 300+ DPI, well-lit, no skew

B) Mixed quality — scans plus smartphone photos with moderate skew/blur/uneven lighting

C) Low quality tolerated — poor lighting, significant skew (>15°), motion blur, partial occlusion; system should gracefully degrade and flag low-confidence fields

D) Other (please describe after [Answer]: tag below)

[Answer]: C

---

## Question 3
Should the system handle both sides of the card?

A) Front side only (PAN front / Aadhaar front) — sufficient for Name, DOB, PAN/Aadhaar number, Father's Name

B) Both sides — also extract address from Aadhaar back, or QR code data

C) Front side required, back side optional (future extension)

D) Other (please describe after [Answer]: tag below)

[Answer]: A

---

## Question 4
Which OCR engine do you prefer as the primary engine?

A) PaddleOCR (generally better accuracy on Indic scripts, heavier model footprint ~100MB)

B) Tesseract (lighter footprint, widely available, may need tuning for Indic text)

C) PaddleOCR primary with Tesseract as fallback for edge cases

D) Let the design phase determine the best option based on latency/accuracy benchmarks

E) Other (please describe after [Answer]: tag below)

[Answer]: D

---

## Question 5
For the benchmark/eval harness, do you have an existing labeled test dataset?

A) Yes — I have labeled card images with ground-truth field values ready to use

B) No — I need the project to include a labeling format/schema so I can create one

C) No — please include a small synthetic/sample test set with the project for development, and I'll add real data later

D) Other (please describe after [Answer]: tag below)

[Answer]: C

---

## Question 6
What is the deployment format / integration surface?

A) Python library with a CLI — `ocr-extract --input image.jpg` → JSON to stdout

B) REST API server (FastAPI/Flask) — POST image, receive JSON response

C) Both CLI and REST API

D) Python library only (importable module, no CLI or server)

E) Other (please describe after [Answer]: tag below)

[Answer]: C

---

## Question 7
For the "Father's Name" field — PAN cards include it, but Aadhaar cards do not (Aadhaar has "S/o", "D/o", "W/o" guardian info which varies). How should this be handled?

A) Extract Father's Name from PAN only; for Aadhaar, extract the guardian name (S/o, D/o, W/o, C/o) as a generic "guardian_name" field

B) Extract Father's Name from PAN only; skip guardian info on Aadhaar entirely

C) Extract guardian relationship + name from both card types where available

D) Other (please describe after [Answer]: tag below)

[Answer]: A

---

## Question 8
What confidence threshold should trigger `requiresReview: true`?

A) Below 0.90 — conservative, more fields flagged for review

B) Below 0.85 — balanced

C) Below 0.80 — permissive, fewer review flags

D) Make it configurable with a sensible default (e.g., 0.85)

E) Other (please describe after [Answer]: tag below)

[Answer]: D

---

## Question 9
Should the system support batch processing (multiple cards in one request)?

A) Single card per request only

B) Batch mode — accept multiple images, return array of results

C) Single card primary, batch as optional feature

D) Other (please describe after [Answer]: tag below)

[Answer]: C

---

## Question 10
Are there specific regional scripts that must be explicitly tested/supported for the multi-script filtering, or is the Latin-detection heuristic (≥90% Latin characters) sufficient regardless of which regional script appears?

A) The Latin-detection heuristic is sufficient — no need to explicitly handle specific regional scripts

B) Must explicitly test against Devanagari (Hindi), Tamil, Telugu, and Bengali — these cover the majority of issued cards

C) Must support all major Indian scripts (Devanagari, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi/Gurmukhi)

D) Other (please describe after [Answer]: tag below)

[Answer]: C

---

## Question: Security Extensions
Should security extension rules be enforced for this project?

A) Yes — enforce all SECURITY rules as blocking constraints (recommended for production-grade applications)

B) No — skip all SECURITY rules (suitable for PoCs, prototypes, and experimental projects)

X) Other (please describe after [Answer]: tag below)

[Answer]: A

---

## Question: Resiliency Extensions
Should the resiliency baseline be applied to this project?

A) Yes — apply the resiliency baseline as directional best practices and design-time guidance (recommended for business-critical workloads)

B) No — skip the resiliency baseline (suitable for PoCs, prototypes, and experimental projects where rapid iteration matters more than reliability)

X) Other (please describe after [Answer]: tag below)

[Answer]: A

---

## Question: Property-Based Testing Extension
Should property-based testing (PBT) rules be enforced for this project?

A) Yes — enforce all PBT rules as blocking constraints (recommended for projects with business logic, data transformations, serialization, or stateful components)

B) Partial — enforce PBT rules only for pure functions and serialization round-trips (suitable for projects with limited algorithmic complexity)

C) No — skip all PBT rules (suitable for simple CRUD applications, UI-only projects, or thin integration layers with no significant business logic)

X) Other (please describe after [Answer]: tag below)

[Answer]: B
