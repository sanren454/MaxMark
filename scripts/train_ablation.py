#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maxmark_repro.latent import embed_secret_in_latent
from maxmark_repro.losses import loss_components
from maxmark_repro.metrics import normalization_gap
from maxmark_repro.reproducibility import set_global_seed


LOSS_VARIANTS = ("mle_moment", "mle_only", "mle_mmd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one controlled MaxMark loss ablation")
    parser.add_argument("--loss_variant", choices=LOSS_VARIANTS, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--initial_checkpoint", type=Path)
    parser.add_argument("--save_initial_checkpoint", type=Path)
    parser.add_argument("--secret_length", type=int, default=16384)
    parser.add_argument("--latent_shape", type=int, nargs=3, default=(4, 64, 64))
    parser.add_argument("--place_mode", default="PLACE_SEQUENTIAL")
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--mmd_max_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.secret_length > int(np.prod(args.latent_shape)):
        raise ValueError("secret_length exceeds the latent capacity")

    from newWm_v4 import newWatermark

    set_global_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training.jsonl"
    if log_path.exists():
        raise FileExistsError(f"Refusing to append to an existing run: {log_path}")

    model = newWatermark(tuple(args.latent_shape)).inn.to(device)
    if args.initial_checkpoint:
        model.load_state_dict(load_state_dict(args.initial_checkpoint, device))
    if args.save_initial_checkpoint:
        args.save_initial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_initial_checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    data_rng = np.random.default_rng(args.seed + 1000)
    mmd_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2000)
    config = vars(args).copy()
    config.update(
        {
            "output_dir": str(args.output_dir),
            "initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
            "save_initial_checkpoint": (
                str(args.save_initial_checkpoint) if args.save_initial_checkpoint else None
            ),
            "resolved_device": str(device),
            "torch_version": torch.__version__,
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    model.train()
    started = time.perf_counter()
    for epoch in range(args.epochs):
        _, _, latent_np = embed_secret_in_latent(
            args.secret_length,
            args.latent_shape,
            args.place_mode,
            args.margin,
            data_rng,
        )
        latent = torch.from_numpy(latent_np).unsqueeze(0).to(device=device, dtype=torch.float32)
        z, log_jac_det = model(latent)
        components = loss_components(
            z,
            log_jac_det,
            args.loss_variant,
            args.lambda_reg,
            mmd_generator,
            args.mmd_max_samples,
        )

        optimizer.zero_grad(set_to_none=True)
        components["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        with torch.no_grad():
            mean_error = z.mean(dim=(0, 2, 3)).abs().mean()
            variance_error = (z.var(dim=(0, 2, 3), correction=0) - 1.0).abs().mean()
            mu = z.mean(dim=(0, 2, 3), keepdim=True)
            std = z.std(dim=(0, 2, 3), correction=0, keepdim=True)
            z_norm = (z - mu) / (std + 1e-6)
            row = {
                "epoch": epoch + 1,
                "loss_total": float(components["total"].item()),
                "loss_mle": float(components["mle"].item()),
                "loss_moment": float(components["moment"].item()),
                "loss_mmd": float(components["mmd"].item()),
                "channel_mean_error": float(mean_error.item()),
                "channel_variance_error": float(variance_error.item()),
                "normalization_gap": normalization_gap(z, z_norm),
                "gradient_norm": float(gradient_norm.item()),
                "elapsed_seconds": time.perf_counter() - started,
            }
        append_jsonl(log_path, row)

        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), args.output_dir / f"epoch_{epoch + 1:04d}.pth")
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[{args.loss_variant}] epoch={epoch + 1}/{args.epochs} "
                f"total={row['loss_total']:.6f} mle={row['loss_mle']:.6f} "
                f"moment={row['loss_moment']:.6f} mmd={row['loss_mmd']:.6f}",
                flush=True,
            )

    torch.save(model.state_dict(), args.output_dir / "final.pth")
    summary = {
        "status": "complete",
        "loss_variant": args.loss_variant,
        "epochs": args.epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "final_metrics": row,
        "final_checkpoint": str(args.output_dir / "final.pth"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
