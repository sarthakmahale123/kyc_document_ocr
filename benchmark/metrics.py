"""Benchmark metrics: accuracy and latency calculations.

Computes:
- Exact match accuracy per field type
- Character error rate (CER) for identifier fields
- Field extraction rate
- p50/p95 latency
"""

from __future__ import annotations

import numpy as np


def character_error_rate(predicted: str, actual: str) -> float:
    """Calculate Character Error Rate (CER) using Levenshtein distance.

    CER = edit_distance(predicted, actual) / len(actual)

    Args:
        predicted: Predicted string.
        actual: Ground truth string.

    Returns:
        CER as a float (0 = perfect, >1 = very bad).
    """
    if not actual:
        return 0.0 if not predicted else 1.0

    # Dynamic programming Levenshtein distance
    m, n = len(predicted), len(actual)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i - 1] == actual[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / n


def calculate_accuracy_metrics(
    ground_truth: list[dict],
    predictions: list[dict | None],
) -> dict:
    """Calculate per-field accuracy metrics.

    Args:
        ground_truth: List of ground truth entries with expected field values.
        predictions: List of extraction results (or None for failed images).

    Returns:
        Dictionary of accuracy metrics per field.
    """
    field_metrics: dict[str, dict] = {}

    for gt_entry, pred in zip(ground_truth, predictions, strict=False):
        if pred is None:
            continue

        gt_fields = gt_entry.get("fields", {})
        pred_fields = pred.get("fields", {})

        for field_name, expected_value in gt_fields.items():
            if field_name not in field_metrics:
                field_metrics[field_name] = {
                    "total": 0,
                    "exact_matches": 0,
                    "extracted": 0,
                    "cer_values": [],
                }

            metrics = field_metrics[field_name]
            metrics["total"] += 1

            pred_field = pred_fields.get(field_name, {})
            pred_value = pred_field.get("value") if pred_field else None

            if pred_value is not None:
                metrics["extracted"] += 1

                # Normalize for comparison
                expected_norm = str(expected_value).strip().upper()
                pred_norm = str(pred_value).strip().upper()

                if expected_norm == pred_norm:
                    metrics["exact_matches"] += 1

                # CER for identifier fields
                if field_name in ("panNumber", "aadhaarNumber"):
                    cer = character_error_rate(pred_norm, expected_norm)
                    metrics["cer_values"].append(cer)

    # Compile final metrics
    result = {}
    for field_name, metrics in field_metrics.items():
        total = metrics["total"]
        field_result = {
            "total_samples": total,
            "exact_match_accuracy": metrics["exact_matches"] / total if total > 0 else 0,
            "extraction_rate": metrics["extracted"] / total if total > 0 else 0,
        }
        if metrics["cer_values"]:
            field_result["mean_cer"] = float(np.mean(metrics["cer_values"]))
            field_result["median_cer"] = float(np.median(metrics["cer_values"]))
        result[field_name] = field_result

    return result


def calculate_latency_metrics(latencies: list[float]) -> dict:
    """Calculate latency percentile metrics.

    Args:
        latencies: List of processing times in milliseconds.

    Returns:
        Dictionary with p50, p95, min, max, mean latency.
    """
    if not latencies:
        return {"p50_ms": 0, "p95_ms": 0, "min_ms": 0, "max_ms": 0, "mean_ms": 0}

    valid_latencies = [lat for lat in latencies if lat > 0]
    if not valid_latencies:
        return {"p50_ms": 0, "p95_ms": 0, "min_ms": 0, "max_ms": 0, "mean_ms": 0}

    arr = np.array(valid_latencies)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "min_ms": round(float(np.min(arr)), 1),
        "max_ms": round(float(np.max(arr)), 1),
        "mean_ms": round(float(np.mean(arr)), 1),
    }


def compare_with_baseline(current: dict, baseline: dict) -> dict:
    """Compare current results with a baseline.

    Args:
        current: Current benchmark results.
        baseline: Baseline benchmark results.

    Returns:
        Comparison dictionary showing improvements/regressions.
    """
    comparison = {"improvements": [], "regressions": [], "unchanged": []}

    current_accuracy = current.get("accuracy", {})
    baseline_accuracy = baseline.get("accuracy", {})

    for field_name in set(list(current_accuracy.keys()) + list(baseline_accuracy.keys())):
        curr = current_accuracy.get(field_name, {})
        base = baseline_accuracy.get(field_name, {})

        curr_acc = curr.get("exact_match_accuracy", 0)
        base_acc = base.get("exact_match_accuracy", 0)

        diff = curr_acc - base_acc
        entry = {
            "field": field_name,
            "current": round(curr_acc, 4),
            "baseline": round(base_acc, 4),
            "diff": round(diff, 4),
        }

        if diff > 0.01:
            comparison["improvements"].append(entry)
        elif diff < -0.01:
            comparison["regressions"].append(entry)
        else:
            comparison["unchanged"].append(entry)

    # Latency comparison
    curr_latency = current.get("latency", {})
    base_latency = baseline.get("latency", {})
    comparison["latency"] = {
        "current_p50": curr_latency.get("p50_ms", 0),
        "baseline_p50": base_latency.get("p50_ms", 0),
        "current_p95": curr_latency.get("p95_ms", 0),
        "baseline_p95": base_latency.get("p95_ms", 0),
    }

    return comparison
