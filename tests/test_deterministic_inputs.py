import unittest

import numpy as np

from maxmark_repro.latent import embed_secret_in_latent


class DeterministicInputTests(unittest.TestCase):
    def test_same_seed_produces_same_secret_and_latent(self):
        first = embed_secret_in_latent(
            64, (4, 8, 8), "PLACE_SEQUENTIAL", 10.0, np.random.default_rng(123)
        )
        second = embed_secret_in_latent(
            64, (4, 8, 8), "PLACE_SEQUENTIAL", 10.0, np.random.default_rng(123)
        )
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_embedded_sign_matches_secret(self):
        secret, positions, latent = embed_secret_in_latent(
            64, (4, 8, 8), "PLACE_SEQUENTIAL", 10.0, np.random.default_rng(8)
        )
        recovered = (latent.reshape(-1)[positions] > 0).astype(np.uint8)
        np.testing.assert_array_equal(recovered, secret)


if __name__ == "__main__":
    unittest.main()
