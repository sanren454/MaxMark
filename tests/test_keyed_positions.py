import unittest

import numpy as np

from maxmark_repro.keyed_positions import (
    keyed_positions,
    position_fingerprint,
    position_overlap_ratio,
)


class KeyedPositionTests(unittest.TestCase):
    def setUp(self):
        self.key = "research-position-key-v1"

    def test_same_inputs_reproduce_same_positions(self):
        first = keyed_positions(16384, 4096, self.key, "owner-a")
        second = keyed_positions(16384, 4096, self.key, "owner-a")
        np.testing.assert_array_equal(first, second)
        self.assertEqual(position_fingerprint(first), position_fingerprint(second))

    def test_positions_are_unique_and_in_range(self):
        positions = keyed_positions(512, 200, self.key, "owner-a")
        self.assertEqual(len(np.unique(positions)), 200)
        self.assertGreaterEqual(int(positions.min()), 0)
        self.assertLess(int(positions.max()), 512)

    def test_wrong_trigger_changes_subset_and_read_order(self):
        correct = keyed_positions(16384, 4096, self.key, "owner-a")
        wrong = keyed_positions(16384, 4096, self.key, "owner-b")
        self.assertFalse(np.array_equal(correct, wrong))
        self.assertAlmostEqual(position_overlap_ratio(correct, wrong), 0.25, delta=0.02)

    def test_wrong_nonce_changes_positions(self):
        correct = keyed_positions(2048, 512, self.key, "owner-a", nonce="image-a")
        wrong = keyed_positions(2048, 512, self.key, "owner-a", nonce="image-b")
        self.assertFalse(np.array_equal(correct, wrong))

    def test_only_correct_positions_restore_embedded_bit_order(self):
        rng = np.random.default_rng(2026)
        total_positions = 16384
        embedded_length = 8192
        payload = rng.integers(0, 2, embedded_length, dtype=np.uint8)
        latent = rng.standard_normal(total_positions).astype(np.float32)
        correct = keyed_positions(total_positions, embedded_length, self.key, "owner-a")
        wrong = keyed_positions(total_positions, embedded_length, self.key, "owner-b")
        latent[correct] = np.where(payload == 0, -10.0, 10.0)

        correct_bits = (latent[correct] >= 0).astype(np.uint8)
        wrong_bits = (latent[wrong] >= 0).astype(np.uint8)
        np.testing.assert_array_equal(correct_bits, payload)
        wrong_accuracy = float(np.mean(wrong_bits == payload))
        self.assertGreater(wrong_accuracy, 0.46)
        self.assertLess(wrong_accuracy, 0.54)

    def test_short_key_is_rejected(self):
        with self.assertRaises(ValueError):
            keyed_positions(128, 32, "short", "owner-a")


if __name__ == "__main__":
    unittest.main()
