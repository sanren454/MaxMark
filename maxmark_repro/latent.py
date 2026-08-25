from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def signed_values(bits: np.ndarray, margin: float, rng: np.random.Generator) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    magnitudes = np.abs(rng.standard_normal(bits.shape)).astype(np.float32) + float(margin)
    return np.where(bits == 0, -magnitudes, magnitudes).astype(np.float32)


def placement_indices(
    info_length: int,
    latent_shape: Sequence[int],
    place_mode: str,
) -> np.ndarray:
    total_positions = int(np.prod(latent_shape))
    if info_length > total_positions:
        raise ValueError(f"info length {info_length} exceeds latent capacity {total_positions}")
    if place_mode == "PLACE_SEQUENTIAL":
        return np.arange(info_length, dtype=np.int64)
    if place_mode == "PLACE_LINSPACE":
        return np.linspace(0, total_positions - 1, num=info_length, dtype=np.int64)
    if place_mode == "PLACE_CHANNEL_SEQ":
        channels, height, width = map(int, latent_shape)
        per_channel = info_length // channels
        remainder = info_length % channels
        positions: list[np.ndarray] = []
        channel_size = height * width
        for channel in range(channels):
            count = per_channel + (1 if channel < remainder else 0)
            positions.append(np.arange(count, dtype=np.int64) + channel * channel_size)
        return np.concatenate(positions)
    raise ValueError(f"Unknown place mode: {place_mode}")


def embed_secret_in_latent(
    secret_length: int,
    latent_shape: Sequence[int] = (4, 64, 64),
    place_mode: str = "PLACE_SEQUENTIAL",
    margin: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the same sign-based training input without importing Diffusers."""
    rng = rng or np.random.default_rng()
    secret = rng.integers(0, 2, size=secret_length, dtype=np.uint8)
    latent = rng.standard_normal(tuple(latent_shape)).astype(np.float32)
    positions = placement_indices(secret_length, latent_shape, place_mode)
    flat = latent.reshape(-1)
    flat[positions] = signed_values(secret, margin, rng)
    return secret, positions, flat.reshape(tuple(latent_shape))
