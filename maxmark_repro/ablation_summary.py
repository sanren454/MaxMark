from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_METRICS = (
    "payload_majority_ber",
    "parity_ber",
    "final_payload_ber",
    "acc_noecc",
    "acc_ecc",
    "ecc_decode_success",
    "ecc_payload_exact_match",
    "normalization_gap",
)


def _number(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"true", "false"}:
        return float(normalized == "true")
    return float(value)


def read_evaluation_rows(
    evaluation_root: Path,
    variants: Sequence[str],
    attacks: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[Path] = []
    for variant in variants:
        for attack in attacks:
            path = evaluation_root / variant / attack / "samples.csv"
            if not path.exists():
                missing.append(path)
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    row["variant"] = variant
                    row["attack"] = attack
                    row["sample_id"] = int(row["sample_id"])
                    row["seed"] = int(row["seed"])
                    rows.append(row)
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing expected evaluation files:\n{rendered}")
    return rows


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), str(row["attack"]))].append(row)

    result: list[dict[str, Any]] = []
    for (variant, attack), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "variant": variant,
            "attack": attack,
            "num_samples": len(group),
        }
        for metric in metrics:
            values = [_number(str(row[metric])) for row in group]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        result.append(summary)
    return result


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 2026,
    iterations: int = 10_000,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty sequence")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(iterations, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def paired_differences(
    rows: Iterable[dict[str, Any]],
    *,
    baseline: str = "mle_moment",
    metrics: Sequence[str] = DEFAULT_METRICS,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        group_key = (str(row["variant"]), str(row["attack"]))
        sample_key = (int(row["sample_id"]), int(row["seed"]))
        grouped[group_key][sample_key] = row

    attacks = sorted({attack for _, attack in grouped})
    variants = sorted({variant for variant, _ in grouped if variant != baseline})
    output: list[dict[str, Any]] = []
    for attack in attacks:
        baseline_rows = grouped.get((baseline, attack), {})
        if not baseline_rows:
            raise ValueError(f"Missing baseline rows for {baseline}/{attack}")
        for variant in variants:
            candidate_rows = grouped.get((variant, attack), {})
            if set(candidate_rows) != set(baseline_rows):
                raise ValueError(
                    f"Unpaired samples for {variant}/{attack}; sample IDs and seeds must match"
                )
            for metric_index, metric in enumerate(metrics):
                deltas = [
                    _number(str(candidate_rows[key][metric]))
                    - _number(str(baseline_rows[key][metric]))
                    for key in sorted(baseline_rows)
                ]
                low, high = bootstrap_mean_ci(
                    deltas,
                    seed=seed + metric_index,
                )
                output.append(
                    {
                        "baseline": baseline,
                        "variant": variant,
                        "attack": attack,
                        "metric": metric,
                        "num_pairs": len(deltas),
                        "mean_delta_variant_minus_baseline": mean(deltas),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
