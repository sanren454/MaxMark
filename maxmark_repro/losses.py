from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def flow_mle_loss(z: torch.Tensor, log_jac_det: torch.Tensor) -> torch.Tensor:
    dimensions = z[0].numel()
    log_prob = -0.5 * z.square().flatten(1).sum(1) - 0.5 * dimensions * np.log(2.0 * np.pi)
    return -(log_prob + log_jac_det).mean() / dimensions


def channel_moment_loss(z: torch.Tensor) -> torch.Tensor:
    channel_mean = z.mean(dim=(0, 2, 3))
    channel_var = z.var(dim=(0, 2, 3), correction=0)
    return channel_mean.square().mean() + (channel_var - 1.0).square().mean()


def _sample_scalars(
    values: torch.Tensor,
    max_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    values = values.reshape(-1)
    sample_count = min(max_samples, values.numel())
    indices = torch.randperm(values.numel(), generator=generator)[:sample_count]
    return values[indices.to(values.device)]


def multiscale_channel_mmd(
    z: torch.Tensor,
    *,
    max_samples: int = 512,
    bandwidths: Iterable[float] = (0.2, 0.5, 1.0, 2.0, 5.0),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """MMD over scalar spatial samples, valid even when the batch size is one."""
    if z.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, received shape {tuple(z.shape)}")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")

    bandwidths = tuple(float(value) for value in bandwidths)
    if not bandwidths:
        raise ValueError("At least one MMD bandwidth is required")
    generator = generator or torch.Generator(device="cpu")
    losses: list[torch.Tensor] = []
    for channel in range(z.shape[1]):
        sample = _sample_scalars(z[:, channel], max_samples, generator)
        reference = torch.randn(
            sample.numel(), generator=generator, dtype=torch.float32, device="cpu"
        ).to(device=z.device, dtype=z.dtype)

        dist_xx = (sample[:, None] - sample[None, :]).square()
        dist_yy = (reference[:, None] - reference[None, :]).square()
        dist_xy = (sample[:, None] - reference[None, :]).square()
        channel_loss = z.new_zeros(())
        for bandwidth in bandwidths:
            if bandwidth <= 0:
                raise ValueError("MMD bandwidths must be positive")
            denominator = 2.0 * float(bandwidth) ** 2
            channel_loss = channel_loss + (
                torch.exp(-dist_xx / denominator).mean()
                + torch.exp(-dist_yy / denominator).mean()
                - 2.0 * torch.exp(-dist_xy / denominator).mean()
            )
        losses.append(channel_loss / len(bandwidths))
    return torch.stack(losses).mean()


def loss_components(
    z: torch.Tensor,
    log_jac_det: torch.Tensor,
    variant: str,
    lambda_reg: float,
    mmd_generator: torch.Generator,
    mmd_max_samples: int,
) -> dict[str, torch.Tensor]:
    mle = flow_mle_loss(z, log_jac_det)
    zero = z.new_zeros(())
    moment = channel_moment_loss(z) if variant == "mle_moment" else zero
    mmd = (
        multiscale_channel_mmd(z, max_samples=mmd_max_samples, generator=mmd_generator)
        if variant == "mle_mmd"
        else zero
    )
    if variant == "mle_moment":
        total = mle + lambda_reg * moment
    elif variant == "mle_mmd":
        total = mle + lambda_reg * mmd
    elif variant == "mle_only":
        total = mle
    else:
        raise ValueError(f"Unknown loss variant: {variant}")
    return {"total": total, "mle": mle, "moment": moment, "mmd": mmd}
