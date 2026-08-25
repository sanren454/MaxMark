import unittest

import torch

from maxmark_repro.losses import multiscale_channel_mmd


class MmdTests(unittest.TestCase):
    def test_shifted_distribution_has_larger_mmd(self):
        torch.manual_seed(11)
        gaussian = torch.randn(1, 4, 16, 16)
        shifted = gaussian + 4.0
        gaussian_loss = multiscale_channel_mmd(
            gaussian,
            max_samples=128,
            generator=torch.Generator().manual_seed(5),
        )
        shifted_loss = multiscale_channel_mmd(
            shifted,
            max_samples=128,
            generator=torch.Generator().manual_seed(5),
        )
        self.assertGreater(float(shifted_loss), float(gaussian_loss))

    def test_mmd_has_nonzero_gradient(self):
        values = (torch.randn(1, 4, 16, 16) + 2.0).requires_grad_(True)
        loss = multiscale_channel_mmd(
            values,
            max_samples=128,
            generator=torch.Generator().manual_seed(9),
        )
        loss.backward()
        self.assertIsNotNone(values.grad)
        self.assertGreater(float(values.grad.abs().sum()), 0.0)

    def test_fixed_seed_is_repeatable(self):
        values = torch.randn(1, 4, 16, 16)
        first = multiscale_channel_mmd(
            values, max_samples=64, generator=torch.Generator().manual_seed(3)
        )
        second = multiscale_channel_mmd(
            values, max_samples=64, generator=torch.Generator().manual_seed(3)
        )
        self.assertAlmostEqual(float(first), float(second), places=7)


if __name__ == "__main__":
    unittest.main()
