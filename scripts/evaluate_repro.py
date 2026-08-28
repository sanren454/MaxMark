#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maxmark_repro.attacks import ATTACK_NAMES, apply_attack
from maxmark_repro.keyed_positions import (
    derive_position_key,
    keyed_positions,
    position_fingerprint,
    position_overlap_ratio,
)
from maxmark_repro.metrics import normalization_gap, paired_bit_metrics, summarize_rows
from maxmark_repro.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible MaxMark image round trip")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--inn_checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", default="Gustavosta/Stable-Diffusion-Prompts")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt_file", type=Path)
    parser.add_argument("--prompt_key")
    parser.add_argument("--secret_length", type=int, default=1024)
    parser.add_argument("--total_size", type=int, default=16384)
    parser.add_argument("--data_backups", type=int, default=3)
    parser.add_argument("--ecc_backups", type=int, default=5)
    parser.add_argument("--latent_shape", type=int, nargs=3, default=(4, 64, 64))
    parser.add_argument("--place_mode", default="PLACE_SEQUENTIAL")
    parser.add_argument("--embedding_slots", type=int, default=8192)
    parser.add_argument("--position_key_env", default="MAXMARK_POSITION_KEY")
    parser.add_argument("--position_trigger_env", default="MAXMARK_POSITION_TRIGGER")
    parser.add_argument("--position_nonce", default="experiment-v1")
    parser.add_argument("--position_model_id", default="stable-diffusion-v1-5")
    parser.add_argument("--margin", type=float, default=10.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--generation_guidance_scale", type=float, default=7.5)
    parser.add_argument("--reverse_guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--reverse_inference_steps", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--attack", choices=ATTACK_NAMES, default="clean")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--local_files_only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--attention_slicing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [args.prompt] * args.num_samples
    if args.prompt_file:
        prompts = [
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if len(prompts) < args.num_samples:
            raise ValueError(
                f"Prompt file contains {len(prompts)} prompts, need {args.num_samples}"
            )
        return prompts[: args.num_samples]

    from datasets import load_dataset

    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        parquet_files = sorted(dataset_path.rglob("*.parquet"))
        if parquet_files:
            dataset = load_dataset(
                "parquet", data_files=[str(path) for path in parquet_files], split="train"
            )
        else:
            dataset = load_dataset(str(dataset_path), split=args.dataset_split)
    else:
        dataset = load_dataset(args.dataset, split=args.dataset_split)

    candidate_keys = [args.prompt_key, "Prompt", "prompt", "TEXT", "caption", "text"]
    prompt_key = next(
        (key for key in candidate_keys if key and key in dataset.column_names), None
    )
    if prompt_key is None:
        raise KeyError(f"No prompt column found in {dataset.column_names}")
    if len(dataset) < args.num_samples:
        raise ValueError(f"Dataset has {len(dataset)} rows, need {args.num_samples}")
    return [str(dataset[index][prompt_key])[:250] for index in range(args.num_samples)]


def image_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device=device, dtype=torch.float32)


def reverse_image(
    image: Image.Image,
    pipe,
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> torch.Tensor:
    from diffusers import DDIMInverseScheduler

    original_scheduler = pipe.scheduler
    original_vae_dtype = pipe.vae.dtype
    try:
        pipe.scheduler = DDIMInverseScheduler.from_pretrained(
            args.model_path,
            subfolder="scheduler",
            local_files_only=args.local_files_only,
        )
        pipe.vae.to(dtype=torch.float32)
        tensor = image_tensor(image, args.image_size, pipe.device)
        latent_distribution = pipe.vae.encode(tensor).latent_dist
        image_latent = latent_distribution.mode() * pipe.vae.config.scaling_factor
        image_latent = image_latent.to(dtype=pipe.unet.dtype)
        result = pipe(
            prompt="",
            latents=image_latent,
            num_inference_steps=args.reverse_inference_steps,
            output_type="latent",
            guidance_scale=args.reverse_guidance_scale,
        )
        return result.images.to(dtype=torch.float32)
    finally:
        pipe.scheduler = original_scheduler
        pipe.vae.to(dtype=original_vae_dtype if original_vae_dtype else torch_dtype)


def load_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    scheduler = DDIMScheduler.from_pretrained(
        args.model_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=args.local_files_only,
    ).to(device)
    if args.attention_slicing:
        pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=False)
    return pipe


def safe_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def decode_candidate(
    decoded_bits: np.ndarray,
    secret: np.ndarray,
    parity_length: int,
    parity_truth: np.ndarray,
    args: argparse.Namespace,
    rs_parameters,
    field,
    rs,
    bitarray_to_parity_blocks_8bit,
    decode_rs_blocks,
) -> dict[str, object]:
    data_length = args.secret_length * args.data_backups
    if decoded_bits.size < data_length:
        raise ValueError("Decoded candidate does not contain all data backups")
    raw_data = decoded_bits[:data_length].reshape(args.data_backups, args.secret_length)
    majority = (raw_data.mean(axis=0) > 0.5).astype(np.uint8)

    ecc_success = False
    final_payload = majority
    decode_error = None
    parity_bit_errors = 0
    parity_ber = 0.0
    actual_ecc_backups = 0
    try:
        if parity_length <= 0:
            raise ValueError("No RS parity bits were embedded")
        available_for_parity = decoded_bits.size - data_length
        actual_ecc_backups = min(
            args.ecc_backups,
            available_for_parity // parity_length,
        )
        if actual_ecc_backups < 1:
            raise ValueError("No selected latent slots remain for an RS parity backup")
        parity_end = data_length + parity_length * actual_ecc_backups
        raw_parity = decoded_bits[data_length:parity_end]
        if raw_parity.size != parity_length * actual_ecc_backups:
            raise ValueError("Decoded parity backup length is inconsistent")
        parity_matrix = raw_parity.reshape(actual_ecc_backups, parity_length)
        parity_vote = (parity_matrix.mean(axis=0) > 0.5).astype(np.uint8)
        parity_bit_errors = int(np.count_nonzero(parity_vote != parity_truth))
        parity_ber = parity_bit_errors / parity_length
        parity_blocks = bitarray_to_parity_blocks_8bit(parity_vote, rs_parameters, field)
        final_payload = decode_rs_blocks(majority, parity_blocks, rs_parameters, field, rs)
        ecc_success = True
    except Exception as error:
        decode_error = f"{type(error).__name__}: {error}"

    return {
        "actual_ecc_backups": actual_ecc_backups,
        "parity_bit_errors": parity_bit_errors,
        "parity_ber": parity_ber,
        "ecc_decode_error": decode_error or "",
        **paired_bit_metrics(raw_data, secret, final_payload, ecc_success),
    }


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not args.inn_checkpoint.exists():
        raise FileNotFoundError(args.inn_checkpoint)
    if args.total_size != int(np.prod(args.latent_shape)):
        raise ValueError("total_size must equal the latent capacity for this evaluator")
    keyed_mode = args.place_mode == "PLACE_KEYED"
    subset_mode = args.place_mode == "PLACE_SEQUENTIAL_SUBSET"
    reduced_slots_mode = keyed_mode or subset_mode
    position_key = None
    position_trigger = None
    if reduced_slots_mode:
        if args.embedding_slots <= 0 or args.embedding_slots >= args.total_size:
            raise ValueError(
                "Reduced-slot placement requires embedding_slots in [1, total_size - 1]"
            )
    if keyed_mode:
        position_key = os.environ.get(args.position_key_env)
        position_trigger = os.environ.get(args.position_trigger_env)
        if not position_key:
            raise ValueError(f"Missing position key environment variable: {args.position_key_env}")
        if not position_trigger:
            raise ValueError(
                f"Missing position trigger environment variable: {args.position_trigger_env}"
            )

    from newWm_v4 import newWatermark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This end-to-end evaluator requires a CUDA GPU")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.output_dir / "generated"
    attacked_dir = args.output_dir / "attacked"
    generated_dir.mkdir(exist_ok=True)
    attacked_dir.mkdir(exist_ok=True)

    set_global_seed(args.seed)
    prompts = load_prompts(args)
    pipe = load_pipeline(args, device, dtype)
    inn = newWatermark(tuple(args.latent_shape)).inn.to(device=device, dtype=torch.float32)
    inn.load_state_dict(safe_state_dict(args.inn_checkpoint, device))
    inn.eval()

    import galois
    from util.ecc_utils import (
        bitarray_to_parity_blocks_8bit,
        decode_rs_blocks,
        get_rs_paras,
    )
    from util.utils import embed_secret_in_latent_rs_2

    p_error = 0.1 if args.secret_length <= 4096 else 0.04
    rs_parameters = get_rs_paras(args.secret_length, p_error=p_error, m=8, epsilon=1e-4)
    m, n_rs, k_rs, _, _, _, _, _, _ = rs_parameters
    if n_rs is None or k_rs is None:
        raise RuntimeError("Automatic RS search did not find a feasible configuration")
    field = galois.GF(2**m)
    rs = galois.ReedSolomon(n_rs, k_rs, field=field)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    command = shlex.join(sys.argv)
    (args.output_dir / "commands.txt").write_text(command + "\n", encoding="utf-8")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for sample_id, prompt in enumerate(prompts):
        sample_seed = args.seed + sample_id
        set_global_seed(sample_seed)
        np.random.seed(sample_seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            source_place_mode = "PLACE_SEQUENTIAL" if reduced_slots_mode else args.place_mode
            payload_slots = args.embedding_slots if reduced_slots_mode else args.total_size
            secret, positions, inn_input, parity_length, parity_truth = embed_secret_in_latent_rs_2(
                args.secret_length,
                payload_slots,
                rs_parameters,
                field,
                rs,
                args.ecc_backups,
                args.data_backups,
                tuple(args.latent_shape),
                source_place_mode,
                args.margin,
            )
            sample_nonce = args.position_nonce
            control_positions: dict[str, np.ndarray] = {}
            if keyed_mode:
                assert position_key is not None
                assert position_trigger is not None
                source_values = inn_input.reshape(-1)[positions].copy()
                positions = keyed_positions(
                    args.total_size,
                    payload_slots,
                    position_key,
                    position_trigger,
                    nonce=sample_nonce,
                    model_id=args.position_model_id,
                )
                relocated = np.random.randn(*tuple(args.latent_shape)).astype(np.float32)
                relocated.reshape(-1)[positions] = source_values
                inn_input = relocated

                wrong_master_key = "hex:" + derive_position_key(
                    position_key,
                    "wrong-key-control",
                    sample_nonce,
                    args.position_model_id,
                ).hex()
                control_positions = {
                    "wrong_key": keyed_positions(
                        args.total_size,
                        payload_slots,
                        wrong_master_key,
                        position_trigger,
                        nonce=sample_nonce,
                        model_id=args.position_model_id,
                    ),
                    "wrong_trigger": keyed_positions(
                        args.total_size,
                        payload_slots,
                        position_key,
                        position_trigger + "::wrong-trigger-control",
                        nonce=sample_nonce,
                        model_id=args.position_model_id,
                    ),
                    "wrong_nonce": keyed_positions(
                        args.total_size,
                        payload_slots,
                        position_key,
                        position_trigger,
                        nonce=sample_nonce + "::wrong-nonce-control",
                        model_id=args.position_model_id,
                    ),
                }
            inn_input_tensor = torch.from_numpy(inn_input).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            with torch.inference_mode():
                z, _ = inn(inn_input_tensor)
                mu = z.mean(dim=(0, 2, 3), keepdim=True)
                std = z.std(dim=(0, 2, 3), correction=0, keepdim=True)
                z_norm = (z - mu) / (std + 1e-6)
                gap = normalization_gap(z, z_norm)
                generated = pipe(
                    prompt=prompt,
                    latents=z_norm.to(dtype=pipe.unet.dtype),
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.generation_guidance_scale,
                ).images[0]
                attacked = apply_attack(generated, args.attack, seed=sample_seed)
                reverse_latent = reverse_image(attacked, pipe, args, dtype)
                decoded, _ = inn(reverse_latent.to(device=device, dtype=torch.float32), rev=True)

            candidate_positions = {"correct": positions, **control_positions}
            candidate_results: dict[str, dict[str, object]] = {}
            for candidate_name, candidate_indices in candidate_positions.items():
                decoded_scores = decoded.flatten()[
                    torch.as_tensor(candidate_indices, device=device)
                ]
                decoded_bits = (decoded_scores >= 0).to(torch.uint8).cpu().numpy()
                candidate_results[candidate_name] = decode_candidate(
                    decoded_bits,
                    secret,
                    parity_length,
                    parity_truth,
                    args,
                    rs_parameters,
                    field,
                    rs,
                    bitarray_to_parity_blocks_8bit,
                    decode_rs_blocks,
                )
            metrics = candidate_results["correct"]
            generated_path = generated_dir / f"sample_{sample_id:04d}.png"
            attacked_path = attacked_dir / f"sample_{sample_id:04d}_{args.attack}.png"
            generated.save(generated_path)
            attacked.save(attacked_path)
            row: dict[str, object] = {
                "sample_id": sample_id,
                "seed": sample_seed,
                "prompt": prompt,
                "attack": args.attack,
                "secret_length": args.secret_length,
                "place_mode": args.place_mode,
                "embedding_slots": payload_slots,
                "normalization_gap": gap,
                "runtime_seconds": time.perf_counter() - started,
                "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
                "generated_path": str(generated_path),
                "attacked_path": str(attacked_path),
                **metrics,
            }
            if keyed_mode:
                row["position_nonce"] = sample_nonce
                row["position_fingerprint"] = position_fingerprint(positions)
                for control_name, control_result in candidate_results.items():
                    if control_name == "correct":
                        continue
                    row[f"{control_name}_position_overlap"] = position_overlap_ratio(
                        positions, candidate_positions[control_name]
                    )
                    for metric_name, value in control_result.items():
                        if isinstance(value, bool):
                            value = int(value)
                        row[f"{control_name}_{metric_name}"] = value
            rows.append(row)
            print(
                f"sample={sample_id} attack={args.attack} "
                f"majority_ber={row['payload_majority_ber']:.6f} "
                f"final_ber={row['final_payload_ber']:.6f} "
                f"exact={row['ecc_payload_exact_match']}"
                + (
                    " "
                    + " ".join(
                        f"{name}_acc={candidate_results[name]['acc_noecc']:.4f}"
                        for name in control_positions
                    )
                    if keyed_mode
                    else ""
                ),
                flush=True,
            )
        except Exception as error:
            failure = {
                "sample_id": sample_id,
                "seed": sample_seed,
                "prompt": prompt,
                "attack": args.attack,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failures.append(failure)
            print(json.dumps(failure, ensure_ascii=False), file=sys.stderr, flush=True)
            break

    if rows:
        write_csv(args.output_dir / "samples.csv", rows)
    summary = summarize_rows(rows)
    summary.update({"attack": args.attack, "failures": failures, "status": "failed" if failures else "complete"})
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
