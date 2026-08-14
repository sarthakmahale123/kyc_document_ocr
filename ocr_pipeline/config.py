"""Configuration management for OCR pipeline.

Loads settings from environment variables and .env files.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OCRConfig(BaseSettings):
    """OCR pipeline configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Confidence threshold — fields below this get requiresReview: true
    confidence_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence threshold below which fields are flagged for review",
    )

    # OCR engine selection
    engine: str = Field(
        default="auto",
        pattern=r"^(paddle|tesseract|auto)$",
        description="OCR engine: paddle, tesseract, or auto",
    )

    # Input limits
    max_image_size_mb: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum image file size in megabytes",
    )

    # Parallelism
    parallel_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Thread pool size for parallel zone OCR",
    )

    # Timeouts
    zone_timeout_s: float = Field(
        default=5.0,
        ge=0.5,
        le=30.0,
        description="Per-zone OCR timeout in seconds",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )
    log_pii: bool = Field(
        default=False,
        description="If true, log extracted PII values (debug only)",
    )

    # Server settings
    rate_limit: str = Field(
        default="30/minute",
        description="API rate limit per client",
    )
    host: str = Field(default="0.0.0.0", description="Server bind address")  # noqa: S104
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port")

    # Model paths
    models_dir: Path = Field(
        default=Path(__file__).parent / "models",
        description="Directory containing model files and zone templates",
    )

    @property
    def max_image_size_bytes(self) -> int:
        """Maximum image size in bytes."""
        return self.max_image_size_mb * 1024 * 1024
