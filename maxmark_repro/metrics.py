from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch


def bit_error_count(predicted: Sequence[int], expected: Sequence[int]) -> int:
    predicted_array = np.asarray(predicted, dtype=np.uint8)
    expected_array = np.asarray(expected, dtype=np.uint8)
    if predicted_array.shape != expected_array.shape:
        raise ValueError(f"Shape mismatch: {predicted_array.shape} != {expected_array.shape}")
    return int(np.count_nonzero(predicted_array != expected_array))


def normalization_gap(z: torch.Tensor, z_norm: torch.Tensor, epsilon: float = 1e-12) -> float:
    numerator = torch.linalg.vector_norm((z - z_norm).float())
    denominator = torch.linalg.vector_norm(z.float()).clamp_min(epsilon)
    return float((numerator / denominator).item())


def paired_bit_metrics(
    raw_data_bits: np.ndarray,
    secret: np.ndarray,
    final_payload: np.ndarray,
    ecc_decode_success: bool,
) -> dict[str, Any]:
    secret = np.asarray(secret, dtype=np.uint8)
    raw_data_bits = np.asarray(raw_data_bits, dtype=np.uint8)
    final_payload = np.asarray(final_payload, dtype=np.uint8)[: secret.size]
    if raw_data_bits.ndim != 2 or raw_data_bits.shape[1] != secret.size:
        raise ValueError("raw_data_bits must have shape (backups, secret_length)")

    majority = (raw_data_bits.mean(axis=0) > 0.5).astype(np.uint8)
    raw_errors = int(np.count_nonzero(raw_data_bits != secret[None, :]))
    majority_errors = bit_error_count(majority, secret)
    final_errors = bit_error_count(final_payload, secret)
    return {
        "raw_bit_errors": raw_errors,
        "raw_ber": raw_errors / raw_data_bits.size,
        "payload_majority_bit_errors": majority_errors,
        "payload_majority_ber": majority_errors / secret.size,
        "final_payload_bit_errors": final_errors,
        "final_payload_ber": final_errors / secret.size,
        "acc_noecc": 1.0 - majority_errors / secret.size,
        "acc_ecc": 1.0 - final_errors / secret.size,
        "ecc_decode_success": bool(ecc_decode_success),
        "ecc_payload_exact_match": bool(ecc_decode_success and final_errors == 0),
    }


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    summary: dict[str, Any] = {"num_samples": len(rows)}
    if not rows:
        return summary
    numeric_keys = sorted(
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        summary[key] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    for key in ("ecc_decode_success", "ecc_payload_exact_match"):
        values = [bool(row[key]) for row in rows if key in row]
        if values:
            summary[f"{key}_rate"] = sum(values) / len(values)
    return summary
