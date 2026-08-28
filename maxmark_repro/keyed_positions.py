from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence

import numpy as np


_DOMAIN = b"MaxMark-PLACE_KEYED-v1"


def _encode_part(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def _master_key_bytes(master_key: str) -> bytes:
    value = master_key.strip()
    if value.startswith("hex:"):
        value = value[4:]
        try:
            decoded = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError("Position key after 'hex:' must contain valid hexadecimal") from error
        if len(decoded) < 16:
            raise ValueError("Hex position key must contain at least 16 bytes")
        return decoded
    if len(value.encode("utf-8")) < 16:
        raise ValueError("Position key must contain at least 16 UTF-8 bytes")
    return value.encode("utf-8")


def derive_position_key(
    master_key: str,
    trigger: str,
    nonce: str,
    model_id: str,
) -> bytes:
    """Derive a domain-separated position key without exposing it in outputs."""
    context = b"".join(
        (
            _DOMAIN,
            _encode_part(trigger),
            _encode_part(nonce),
            _encode_part(model_id),
        )
    )
    return hmac.new(_master_key_bytes(master_key), context, hashlib.sha256).digest()


def keyed_positions(
    total_positions: int,
    count: int,
    master_key: str,
    trigger: str,
    nonce: str = "experiment-v1",
    model_id: str = "stable-diffusion-v1-5",
) -> np.ndarray:
    """Return a deterministic keyed subset whose order also carries information."""
    if total_positions <= 0:
        raise ValueError("total_positions must be positive")
    if count <= 0 or count > total_positions:
        raise ValueError("count must be in [1, total_positions]")

    derived_key = derive_position_key(master_key, trigger, nonce, model_id)

    def rank(index: int) -> bytes:
        return hashlib.blake2b(
            index.to_bytes(8, "big"),
            key=derived_key,
            digest_size=16,
            person=b"MaxMarkPosV1",
        ).digest()

    ordered = sorted(range(total_positions), key=rank)
    return np.asarray(ordered[:count], dtype=np.int64)


def position_overlap_ratio(left: Sequence[int], right: Sequence[int]) -> float:
    left_array = np.asarray(left, dtype=np.int64)
    right_array = np.asarray(right, dtype=np.int64)
    if left_array.size == 0:
        raise ValueError("left positions must not be empty")
    return len(set(left_array.tolist()).intersection(right_array.tolist())) / left_array.size


def position_fingerprint(positions: Sequence[int]) -> str:
    values = np.asarray(positions, dtype=">i8")
    return hashlib.sha256(values.tobytes()).hexdigest()
