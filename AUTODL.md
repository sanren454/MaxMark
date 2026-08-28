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
bash scripts/run_autodl.sh subset-minimal
bash scripts/run_autodl.sh keyed-minimal
bash scripts/run_autodl.sh train
bash scripts/run_autodl.sh evaluate
```

`minimal` verifies the original full-slot path. `subset-minimal` uses the same
8192-slot payload as the keyed experiment but places it sequentially, isolating
the effect of reducing occupied slots from the effect of changing locations.
`keyed-minimal` reuses the existing checkpoint and tests correct keyed positions
against wrong-key, wrong-trigger, and wrong-nonce reads on the same recovered
latent. Its key and trigger are read from the server-side `.env` and are not
written to command or environment artifacts.
`evaluate` writes both aggregate results and paired differences against
`mle_moment`.

The experiment design and stop criteria are recorded in
`reports/keyed_latent_position_watermark_plan.md`.
