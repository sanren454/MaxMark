import unittest

import galois
import numpy as np

from util.ecc_utils import (
    bitarray_to_parity_blocks_8bit,
    decode_rs_blocks,
    encode_rs_bitstring,
    parity_blocks_to_bitarray_8bit,
)


class ReedSolomonRoundTripTests(unittest.TestCase):
    def test_one_corrupted_data_bit_is_corrected(self):
        parameters = (8, 15, 11, 4, 2, 88, 0.0, 0.0, 1)
        field = galois.GF(2**8)
        codec = galois.ReedSolomon(15, 11, field=field)
        secret = np.random.default_rng(4).integers(0, 2, 100, dtype=np.uint8)
        parity_blocks = encode_rs_bitstring(secret, parameters, field, codec)
        parity_bits = parity_blocks_to_bitarray_8bit(parity_blocks)
        restored_parity = bitarray_to_parity_blocks_8bit(parity_bits, parameters, field)
        corrupted = secret.copy()
        corrupted[3] ^= 1
        decoded = decode_rs_blocks(corrupted, restored_parity, parameters, field, codec)
        np.testing.assert_array_equal(decoded, secret)


if __name__ == "__main__":
    unittest.main()
