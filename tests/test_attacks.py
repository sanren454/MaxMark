import unittest

import numpy as np
from PIL import Image

from maxmark_repro.attacks import apply_attack


class AttackTests(unittest.TestCase):
    def setUp(self):
        gradient = np.arange(64 * 64 * 3, dtype=np.uint16).reshape(64, 64, 3) % 256
        self.image = Image.fromarray(gradient.astype(np.uint8), mode="RGB")

    def test_clean_is_pixel_identical(self):
        attacked = apply_attack(self.image, "clean", seed=1)
        np.testing.assert_array_equal(np.asarray(attacked), np.asarray(self.image))

    def test_noise_is_deterministic_and_not_identity(self):
        first = np.asarray(apply_attack(self.image, "noise005", seed=7))
        second = np.asarray(apply_attack(self.image, "noise005", seed=7))
        np.testing.assert_array_equal(first, second)
        self.assertGreater(np.count_nonzero(first != np.asarray(self.image)), 0)

    def test_resize_returns_original_size(self):
        attacked = apply_attack(self.image, "resize25", seed=1)
        self.assertEqual(attacked.size, self.image.size)

    def test_unknown_attack_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_attack(self.image, "jpeg_and_resize", seed=1)


if __name__ == "__main__":
    unittest.main()
