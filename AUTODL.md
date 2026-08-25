# Local Codex + private Git + AutoDL workflow

The repository contains code and configuration only. Model weights, datasets,
credentials, generated images, and experiment outputs stay outside Git.

## One-time local setup

Create a private repository, then add it without replacing the official source
remote:

```powershell
git remote rename origin upstream
git remote add origin <PRIVATE_REPOSITORY_URL>
git push -u origin codex/maxmark-repro
```

Use your normal local Git credential manager or SSH key. Do not put a token in a
remote URL or commit it to a file.

## One-time AutoDL setup

Clone the private repository with a read-only deploy key when possible. Copy
`.env.example` to `.env` and edit only the paths for the existing Stable
Diffusion model, dataset, INN checkpoint, and persistent output directory.

Install dependencies once. Keep the CUDA-matched PyTorch build supplied by the
AutoDL image:

```bash
python -m pip install -r requirements-autodl.txt
bash scripts/run_autodl.sh preflight
```

## Normal iteration

Local computer:

```powershell
git add -A
git commit -m "describe the experiment change"
git push
```

AutoDL:

```bash
git pull --ff-only
bash scripts/run_autodl.sh minimal
bash scripts/run_autodl.sh train
bash scripts/run_autodl.sh evaluate
```

`minimal` is the stop gate: it must produce one clean and one JPEG25 full
generate -> attack -> DDIM inversion -> INN reverse -> RS decode result before
running the three training ablations. `evaluate` writes both aggregate results
and paired differences against `mle_moment`.
