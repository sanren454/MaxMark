from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from maxmark_repro.ablation_summary import (
    aggregate_rows,
    paired_differences,
    read_evaluation_rows,
)


class AblationSummaryTests(unittest.TestCase):
    def test_aggregate_and_paired_delta(self) -> None:
        rows = []
        for variant, offset in (("mle_moment", 0.0), ("mle_only", 0.1)):
            for sample_id in range(2):
                rows.append(
                    {
                        "variant": variant,
                        "attack": "clean",
                        "sample_id": sample_id,
                        "seed": 100 + sample_id,
                        "payload_majority_ber": str(0.2 + offset),
                        "parity_ber": str(0.15 + offset),
                        "final_payload_ber": str(0.1 + offset),
                        "acc_noecc": str(0.8 - offset),
                        "acc_ecc": str(0.9 - offset),
                        "ecc_decode_success": "True",
                        "ecc_payload_exact_match": "False",
                        "normalization_gap": str(0.01 + offset),
                    }
                )

        aggregate = aggregate_rows(rows)
        self.assertEqual(len(aggregate), 2)
        paired = paired_differences(rows, seed=7)
        final_ber = next(row for row in paired if row["metric"] == "final_payload_ber")
        self.assertAlmostEqual(final_ber["mean_delta_variant_minus_baseline"], 0.1)
        self.assertAlmostEqual(final_ber["ci95_low"], 0.1)
        self.assertAlmostEqual(final_ber["ci95_high"], 0.1)

    def test_reader_requires_every_expected_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                read_evaluation_rows(root, ["mle_moment"], ["clean"])

    def test_reader_tags_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "mle_moment" / "clean"
            target.mkdir(parents=True)
            with (target / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["sample_id", "seed"])
                writer.writeheader()
                writer.writerow({"sample_id": 0, "seed": 2026})
            rows = read_evaluation_rows(root, ["mle_moment"], ["clean"])
            self.assertEqual(rows[0]["variant"], "mle_moment")
            self.assertEqual(rows[0]["attack"], "clean")


if __name__ == "__main__":
    unittest.main()
