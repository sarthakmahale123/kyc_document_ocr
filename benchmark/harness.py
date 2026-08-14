"""Benchmark and evaluation harness.

Measures field-level accuracy and latency against a labeled test set.

Usage:
    python -m benchmark.harness --data benchmark/test_data --output results.json
    python -m benchmark.harness --data benchmark/test_data --baseline baseline_results.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from benchmark.metrics import (
    calculate_accuracy_metrics,
    calculate_latency_metrics,
    compare_with_baseline,
)
from ocr_pipeline import OCRConfig, extract_card


def run_benchmark(
    data_dir: Path,
    config: OCRConfig | None = None,
) -> dict:
    """Run the benchmark against a labeled test set.

    Args:
        data_dir: Directory containing images/ and ground_truth.json.
        config: Pipeline configuration.

    Returns:
        Benchmark results dictionary.
    """
    if config is None:
        config = OCRConfig()

    # Load ground truth
    gt_path = data_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with open(gt_path) as f:
        ground_truth = json.load(f)

    images_dir = data_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Run extraction on each image
    predictions = []
    latencies = []

    for entry in ground_truth:
        filename = entry["filename"]
        image_path = images_dir / filename

        if not image_path.exists():
            click.echo(f"Warning: Image not found: {image_path}", err=True)
            predictions.append(None)
            latencies.append(0)
            continue

        start_time = time.perf_counter()
        result = extract_card(str(image_path), config)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        predictions.append(result)
        latencies.append(elapsed_ms)

    # Calculate metrics
    accuracy_metrics = calculate_accuracy_metrics(ground_truth, predictions)
    latency_metrics = calculate_latency_metrics(latencies)

    return {
        "total_images": len(ground_truth),
        "processed_images": sum(1 for p in predictions if p is not None),
        "accuracy": accuracy_metrics,
        "latency": latency_metrics,
    }


@click.command()
@click.option(
    "--data",
    type=click.Path(exists=True),
    required=True,
    help="Path to test data directory (contains images/ and ground_truth.json)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file for results (default: stdout)",
)
@click.option(
    "--baseline",
    type=click.Path(exists=True),
    default=None,
    help="Baseline results file for comparison",
)
@click.option(
    "--engine",
    type=click.Choice(["paddle", "tesseract", "auto"]),
    default="auto",
    help="OCR engine to benchmark",
)
def main(data: str, output: str | None, baseline: str | None, engine: str):
    """Run OCR pipeline benchmark against labeled test data."""
    config = OCRConfig(engine=engine)
    data_path = Path(data)

    click.echo(f"Running benchmark on: {data_path}", err=True)
    click.echo(f"Engine: {engine}", err=True)

    results = run_benchmark(data_path, config)

    # Compare with baseline if provided
    if baseline:
        with open(baseline) as f:
            baseline_data = json.load(f)
        comparison = compare_with_baseline(results, baseline_data)
        results["baseline_comparison"] = comparison

    # Output results
    results_json = json.dumps(results, indent=2, ensure_ascii=False)

    if output:
        Path(output).write_text(results_json)
        click.echo(f"Results written to: {output}", err=True)
    else:
        click.echo(results_json)


if __name__ == "__main__":
    main()
