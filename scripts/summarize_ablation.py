#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maxmark_repro.ablation_summary import (
    aggregate_rows,
    paired_differences,
    read_evaluation_rows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired MaxMark ablations")
    parser.add_argument("--evaluation_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=("mle_moment", "mle_only", "mle_mmd"),
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=("clean", "jpeg25", "resize25"),
    )
    parser.add_argument("--baseline", default="mle_moment")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_evaluation_rows(args.evaluation_root, args.variants, args.attacks)
    aggregate = aggregate_rows(rows)
    paired = paired_differences(rows, baseline=args.baseline, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ablation_summary.csv", aggregate)
    write_csv(args.output_dir / "paired_differences.csv", paired)
    payload = {
        "baseline": args.baseline,
        "variants": list(args.variants),
        "attacks": list(args.attacks),
        "aggregate": aggregate,
        "paired_differences": paired,
        "delta_definition": "variant minus baseline; lower is better for BER and normalization_gap",
    }
    (args.output_dir / "ablation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote ablation summary to {args.output_dir}")


if __name__ == "__main__":
    main()
