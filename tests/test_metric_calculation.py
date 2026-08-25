import unittest

import numpy as np
import torch

from maxmark_repro.metrics import normalization_gap, paired_bit_metrics, summarize_rows


class MetricTests(unittest.TestCase):
    def test_paired_bit_metrics(self):
        secret = np.array([0, 1, 1, 0], dtype=np.uint8)
        raw = np.array(
            [
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 1, 1, 0],
            ],
            dtype=np.uint8,
        )
        result = paired_bit_metrics(raw, secret, secret.copy(), True)
        self.assertEqual(result["raw_bit_errors"], 2)
        self.assertEqual(result["payload_majority_bit_errors"], 0)
        self.assertTrue(result["ecc_payload_exact_match"])

    def test_normalization_gap_identity(self):
        values = torch.randn(1, 4, 8, 8)
        self.assertEqual(normalization_gap(values, values.clone()), 0.0)

    def test_summary_success_rates(self):
        summary = summarize_rows(
            [
                {"acc_ecc": 1.0, "ecc_decode_success": True, "ecc_payload_exact_match": True},
                {"acc_ecc": 0.5, "ecc_decode_success": False, "ecc_payload_exact_match": False},
            ]
        )
        self.assertEqual(summary["ecc_decode_success_rate"], 0.5)
        self.assertEqual(summary["acc_ecc"]["mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
