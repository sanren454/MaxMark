from __future__ import annotations

from io import BytesIO
from typing import Final

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


ATTACK_NAMES: Final[tuple[str, ...]] = (
    "clean",
    "jpeg25",
    "resize25",
    "blur5",
    "brightness3",
    "noise005",
)


def _ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image.copy()


def apply_attack(image: Image.Image, attack: str, seed: int = 0) -> Image.Image:
    """Apply one deterministic paper-style attack to a PIL image."""
    if attack not in ATTACK_NAMES:
        raise ValueError(f"Unknown attack {attack!r}; choose from {ATTACK_NAMES}")

    image = _ensure_rgb(image)
    if attack == "clean":
        return image
    if attack == "jpeg25":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=25)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if attack == "resize25":
        width, height = image.size
        small = image.resize(
            (max(1, round(width * 0.25)), max(1, round(height * 0.25))),
            Image.Resampling.BICUBIC,
        )
        return small.resize((width, height), Image.Resampling.BICUBIC)
    if attack == "blur5":
        return image.filter(ImageFilter.GaussianBlur(radius=5))
    if attack == "brightness3":
        return ImageEnhance.Brightness(image).enhance(3.0)

    rng = np.random.default_rng(seed)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    noise = rng.normal(loc=0.0, scale=0.05, size=pixels.shape).astype(np.float32)
    attacked = np.clip(pixels + noise, 0.0, 1.0)
    return Image.fromarray(np.rint(attacked * 255.0).astype(np.uint8), mode="RGB")
