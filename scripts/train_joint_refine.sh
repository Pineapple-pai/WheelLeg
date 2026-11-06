#!/usr/bin/env bash
# Experimental, bounded continuation; not a validated replacement policy.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
export SOURCE="${SOURCE:-checkpoints/joint_precision_20261105_184905/checkpoint}"
export VERSION="${VERSION:-joint_refine_$(date +%Y%m%d_%H%M%S)_$$}"
export EVAL_SEEDS="${EVAL_SEEDS:-65600 65601 65602 65603 65604}"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-wheelleg-train}"
if [[ "$VERSION" == *'/'* || "$VERSION" == '.' || "$VERSION" == '..' ]]; then
  echo 'VERSION must be a single directory name.' >&2; exit 2
fi
for suffix in .zip _vecnormalize.pkl _contract.json; do
  [[ -f "${SOURCE}${suffix}" ]] || { echo "Missing source: ${SOURCE}${suffix}" >&2; exit 2; }
done
for folder in "checkpoints/$VERSION" "diagnostics/$VERSION" "diagnostics/${VERSION}_source_evaluation"; do
  [[ ! -e "$folder" ]] || { echo "Refusing to overwrite: $folder" >&2; exit 2; }
done
read -r -a EVAL_SEED_ARGS <<< "$EVAL_SEEDS"
mkdir -p logs
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/eval_new_model_training.py \
  "$SOURCE" --output "diagnostics/${VERSION}_source_evaluation" \
  --launch-speeds .1 .25 .5 --seeds "${EVAL_SEED_ARGS[@]}" \
  2>&1 | tee "logs/${VERSION}_source_evaluation.log"
bash scripts/train_joint_precision.sh \
  --command-sampling startup_balanced --learning-rate 0.00001 \
  --updates 25 --save-interval 25 "$@"
if conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/check_refinement.py \
  --source "diagnostics/${VERSION}_source_evaluation/results.json" \
  --candidate "diagnostics/$VERSION/evaluation/results.json" \
  --output "diagnostics/$VERSION/refinement_gate.json"; then
  echo '候选通过本轮配对门槛，仍需独立随机化验证；未自动替换源权重。'
else
  gate_status=$?
  echo "候选未通过验收，请保留源权重：$SOURCE" >&2
  exit "$gate_status"
fi
