# MaxMark reproduction evidence

This repository prepares the code only. A completed reproduction must include artifacts produced on AutoDL:

- `environment.json` with CUDA and dependency versions;
- minimal clean and JPEG25 `samples.csv`, `summary.json`, and images;
- three independently trained checkpoints sharing one initialization;
- per-epoch `training.jsonl` files;
- paired clean/JPEG25/Resize25 evaluation outputs;
- `summary/ablation_summary.csv` and `summary/paired_differences.csv`;
- command logs showing the exact successful invocations.

Do not treat environment setup, a loaded checkpoint, or a training-only checkpoint as an end-to-end reproduction.

## AutoDL usage

Configure the variables from `.env.example`, then run:

```bash
git pull --ff-only
bash scripts/run_autodl.sh preflight
bash scripts/run_autodl.sh minimal
bash scripts/run_autodl.sh train
bash scripts/run_autodl.sh evaluate
```

The runner loads the repository-local `.env` automatically. See `AUTODL.md` for
the private Git and AutoDL split workflow.

For one uninterrupted run:

```bash
bash scripts/run_autodl.sh all
```
