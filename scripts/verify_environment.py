#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "datasets",
    "galois",
    "FrEIA",
    "PIL",
    "numpy",
    "scipy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an offline AutoDL MaxMark runtime")
    parser.add_argument("--model_path", default=os.getenv("MAXMARK_MODEL_PATH"))
    parser.add_argument("--inn_checkpoint", default=os.getenv("MAXMARK_EXISTING_INN"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow_missing_inn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "modules": {},
        "paths": {},
    }
    failures: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            report["modules"][name] = getattr(module, "__version__", "available")
        except Exception as error:  # preflight must report every missing dependency
            report["modules"][name] = f"ERROR: {error}"
            failures.append(f"module {name}: {error}")

    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_version"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if not torch.cuda.is_available():
            failures.append("CUDA is not available")
    except Exception:
        report["cuda_available"] = False

    for label, raw_path in (
        ("model_path", args.model_path),
        ("inn_checkpoint", args.inn_checkpoint),
    ):
        exists = bool(raw_path and Path(raw_path).exists())
        report["paths"][label] = {"value": raw_path, "exists": exists}
        if not exists and not (label == "inn_checkpoint" and args.allow_missing_inn):
            failures.append(f"{label} does not exist: {raw_path}")

    report["status"] = "ok" if not failures else "failed"
    report["failures"] = failures
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
