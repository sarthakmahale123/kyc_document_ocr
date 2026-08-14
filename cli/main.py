"""CLI entry point for ocr-extract command.

Usage:
    ocr-extract INPUT_PATH [OPTIONS]
    ocr-extract --batch DIRECTORY [OPTIONS]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from ocr_pipeline import OCRConfig, extract_batch, extract_card


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path (default: stdout)",
)
@click.option(
    "--batch",
    is_flag=True,
    default=False,
    help="Treat INPUT_PATH as directory, process all images",
)
@click.option(
    "--confidence",
    type=float,
    default=None,
    help="Confidence threshold override (default: 0.85)",
)
@click.option(
    "--engine",
    type=click.Choice(["paddle", "tesseract", "auto"]),
    default=None,
    help="OCR engine selection (default: auto)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "jsonl"]),
    default="json",
    help="Output format (default: json)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging",
)
def cli(
    input_path: str,
    output: str | None,
    batch: bool,
    confidence: float | None,
    engine: str | None,
    output_format: str,
    verbose: bool,
):
    """Extract structured fields from Indian PAN and Aadhaar card images.

    INPUT_PATH can be a single image file or a directory (with --batch flag).
    """
    import logging

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Build config with overrides
    config_kwargs = {}
    if confidence is not None:
        config_kwargs["confidence_threshold"] = confidence
    if engine is not None:
        config_kwargs["engine"] = engine

    config = OCRConfig(**config_kwargs)

    path = Path(input_path)

    if batch:
        if not path.is_dir():
            click.echo(f"Error: {input_path} is not a directory (required with --batch)", err=True)
            sys.exit(1)

        # Collect image files
        image_extensions = {".jpg", ".jpeg", ".png"}
        image_files = sorted(
            f for f in path.iterdir() if f.suffix.lower() in image_extensions and f.is_file()
        )

        if not image_files:
            click.echo(f"Error: No image files found in {input_path}", err=True)
            sys.exit(1)

        click.echo(f"Processing {len(image_files)} images...", err=True)

        results = extract_batch([str(f) for f in image_files], config)

        # Add filenames to results
        for i, result in enumerate(results):
            result["filename"] = image_files[i].name

        output_data = _format_output(results, output_format, is_batch=True)
    else:
        if not path.is_file():
            click.echo(f"Error: {input_path} is not a file", err=True)
            sys.exit(1)

        result = extract_card(str(path), config)
        result["filename"] = path.name
        output_data = _format_output(result, output_format, is_batch=False)

    # Write output
    if output:
        Path(output).write_text(output_data)
        click.echo(f"Results written to {output}", err=True)
    else:
        click.echo(output_data)


def _format_output(data, output_format: str, is_batch: bool) -> str:
    """Format output data as JSON or JSONL."""
    if output_format == "jsonl":
        if is_batch:
            return "\n".join(json.dumps(item, ensure_ascii=False) for item in data)
        else:
            return json.dumps(data, ensure_ascii=False)
    else:
        return json.dumps(data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    cli()
