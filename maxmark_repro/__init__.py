"""Reproducible training and evaluation helpers for MaxMark."""

from .attacks import ATTACK_NAMES, apply_attack
from .keyed_positions import keyed_positions, position_fingerprint, position_overlap_ratio
from .latent import embed_secret_in_latent
from .losses import channel_moment_loss, flow_mle_loss, multiscale_channel_mmd
from .metrics import bit_error_count, normalization_gap, summarize_rows
from .reproducibility import set_global_seed

__all__ = [
    "ATTACK_NAMES",
    "apply_attack",
    "bit_error_count",
    "channel_moment_loss",
    "embed_secret_in_latent",
    "flow_mle_loss",
    "keyed_positions",
    "multiscale_channel_mmd",
    "normalization_gap",
    "position_fingerprint",
    "position_overlap_ratio",
    "set_global_seed",
    "summarize_rows",
]
