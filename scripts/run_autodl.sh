#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi
PYTHON_BIN="${MAXMARK_PYTHON:-python}"
MODEL_PATH="${MAXMARK_MODEL_PATH:-}"
DATASET_PATH="${MAXMARK_DATASET_PATH:-Gustavosta/Stable-Diffusion-Prompts}"
EXISTING_INN="${MAXMARK_EXISTING_INN:-}"
OUTPUT_ROOT="${MAXMARK_OUTPUT_DIR:-/root/autodl-tmp/maxmark-runs}"
SEED="${MAXMARK_SEED:-2026}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-400}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-10}"
PROMPT_FILE="${MAXMARK_PROMPT_FILE:-${ROOT_DIR}/configs/prompts.txt}"

mkdir -p "${OUTPUT_ROOT}/logs"

require_model_path() {
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "MAXMARK_MODEL_PATH is required" >&2
    exit 2
  fi
}

preflight() {
  require_model_path
  local args=(
    "${ROOT_DIR}/scripts/verify_environment.py"
    --model_path "${MODEL_PATH}"
    --output "${OUTPUT_ROOT}/environment.json"
  )
  if [[ -n "${EXISTING_INN}" ]]; then
    args+=(--inn_checkpoint "${EXISTING_INN}")
  else
    args+=(--allow_missing_inn)
  fi
  "${PYTHON_BIN}" "${args[@]}" | tee "${OUTPUT_ROOT}/logs/preflight.log"
}

evaluate_checkpoint() {
  local checkpoint="$1"
  local label="$2"
  local attack="$3"
  local samples="$4"
  local output_dir="${OUTPUT_ROOT}/evaluation/${label}/${attack}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/evaluate_repro.py" \
    --model_path "${MODEL_PATH}" \
    --inn_checkpoint "${checkpoint}" \
    --dataset "${DATASET_PATH}" \
    --prompt_file "${PROMPT_FILE}" \
    --secret_length 1024 \
    --total_size 16384 \
    --data_backups 3 \
    --ecc_backups 5 \
    --margin 10 \
    --num_samples "${samples}" \
    --seed "${SEED}" \
    --generation_guidance_scale 7.5 \
    --reverse_guidance_scale 1.0 \
    --num_inference_steps 50 \
    --reverse_inference_steps 50 \
    --dtype fp16 \
    --attack "${attack}" \
    --output_dir "${output_dir}" \
    --local_files_only \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${label}_${attack}.log"
}

minimal() {
  require_model_path
  if [[ -z "${EXISTING_INN}" || ! -f "${EXISTING_INN}" ]]; then
    echo "MAXMARK_EXISTING_INN must point to the existing INN checkpoint" >&2
    exit 2
  fi
  evaluate_checkpoint "${EXISTING_INN}" existing clean 1
  evaluate_checkpoint "${EXISTING_INN}" existing jpeg25 1
}

train_all() {
  local checkpoint_root="${OUTPUT_ROOT}/checkpoints"
  local shared_initial="${checkpoint_root}/shared_initial.pth"
  mkdir -p "${checkpoint_root}"

  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_ablation.py" \
    --loss_variant mle_moment \
    --output_dir "${checkpoint_root}/mle_moment" \
    --save_initial_checkpoint "${shared_initial}" \
    --epochs "${TRAIN_EPOCHS}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/train_mle_moment.log"

  for variant in mle_only mle_mmd; do
    "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_ablation.py" \
      --loss_variant "${variant}" \
      --output_dir "${checkpoint_root}/${variant}" \
      --initial_checkpoint "${shared_initial}" \
      --epochs "${TRAIN_EPOCHS}" \
      --seed "${SEED}" \
      2>&1 | tee "${OUTPUT_ROOT}/logs/train_${variant}.log"
  done
}

evaluate_all() {
  require_model_path
  for variant in mle_moment mle_only mle_mmd; do
    local checkpoint="${OUTPUT_ROOT}/checkpoints/${variant}/final.pth"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing checkpoint: ${checkpoint}" >&2
      exit 2
    fi
    for attack in clean jpeg25 resize25; do
      evaluate_checkpoint "${checkpoint}" "${variant}" "${attack}" "${EVAL_NUM_SAMPLES}"
    done
  done
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_ablation.py" \
    --evaluation_root "${OUTPUT_ROOT}/evaluation" \
    --output_dir "${OUTPUT_ROOT}/summary" \
    --seed "${SEED}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/summarize_ablation.log"
}

case "${PHASE}" in
  preflight)
    preflight
    ;;
  minimal)
    minimal
    ;;
  train)
    train_all
    ;;
  evaluate)
    evaluate_all
    ;;
  all)
    preflight
    minimal
    train_all
    evaluate_all
    ;;
  *)
    echo "Usage: $0 {preflight|minimal|train|evaluate|all}" >&2
    exit 2
    ;;
esac
