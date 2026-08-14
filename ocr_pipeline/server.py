"""FastAPI REST API server for OCR pipeline.

Endpoints:
    POST /extract       — Single card extraction
    POST /extract/batch — Batch extraction
    GET  /health        — Health check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ocr_pipeline import __version__
from ocr_pipeline.config import OCRConfig
from ocr_pipeline.input_gateway import InputValidationError
from ocr_pipeline.pipeline import extract_batch, extract_card

logger = logging.getLogger(__name__)

config = OCRConfig()

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=[config.rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    logger.info("OCR Pipeline API starting (version %s)", __version__)
    yield
    logger.info("OCR Pipeline API shutting down")


app = FastAPI(
    title="PAN/Aadhaar OCR Auto-Fill API",
    version=__version__,
    description="Extracts structured identity fields from Indian PAN and Aadhaar card images.",
    lifespan=lifespan,
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "detail": "Too many requests. Please try again later."},
    )


@app.post("/extract")
@limiter.limit(config.rate_limit)
async def extract_single(request: Request, file: UploadFile = File()):  # noqa: B008
    """Extract fields from a single PAN or Aadhaar card image.

    Accepts a multipart file upload (JPEG or PNG).
    Returns structured JSON with per-field extraction results.
    """
    # Read file content
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        result = extract_card(content, config)
    except InputValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e

    # Check if classification failed
    if result.get("error") and "unknown_document" in result.get("error", ""):
        raise HTTPException(
            status_code=422,
            detail=result["error"],
        )

    return result


@app.post("/extract/batch")
@limiter.limit(config.rate_limit)
async def extract_multiple(request: Request, files: list[UploadFile] = File()):  # noqa: B008
    """Extract fields from multiple card images.

    Accepts multiple multipart file uploads.
    Returns an array of extraction results.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files per batch request",
        )

    contents = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Empty file: {f.filename}")
        contents.append(data)

    results = extract_batch(contents, config)
    return {"results": results}


@app.get("/health")
async def health_check():
    """Health check endpoint for readiness/liveness probing."""
    return {
        "status": "healthy",
        "version": __version__,
        "engine": config.engine,
        "confidence_threshold": config.confidence_threshold,
    }


def create_app() -> FastAPI:
    """Factory function for creating the FastAPI app (useful for testing)."""
    return app


# Mount static files for the web UI
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_ui():
        """Serve the web UI."""
        return FileResponse(_static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
