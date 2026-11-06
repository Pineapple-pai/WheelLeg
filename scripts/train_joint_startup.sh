#!/usr/bin/env bash
# Bounded foundation adaptation from the new-model standing checkpoint.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-wheelleg-train}"
VERSION="${VERSION:-joint_startup_$(date +%Y%m%d_%H%M%S)_$$}"
SEED="${SEED:-0}"
SOURCE="${SOURCE:-checkpoints/joint_tracking_pos_v16_model20261104/checkpoint}"
INIT_ARGS=(--checkpoint "$SOURCE" --vecnormalize-path "${SOURCE}_vecnormalize.pkl" --preserve-action-std)
if [[ "${FROM_SCRATCH:-0}" == 1 ]]; then
  INIT_ARGS=(--action-std 0.06)
else
  for suffix in .zip _vecnormalize.pkl _contract.json; do
    if [[ ! -f "${SOURCE}${suffix}" ]]; then
      echo "Missing source: ${SOURCE}${suffix}" >&2
      exit 2
    fi
  done
fi
if [[ "$VERSION" == *'/'* || "$VERSION" == '.' || "$VERSION" == '..' ]]; then
  echo 'VERSION must be a single directory name.' >&2
  exit 2
fi
if [[ -e "checkpoints/$VERSION" || -e "diagnostics/$VERSION" ]]; then
  echo "Refusing to overwrite existing run: $VERSION" >&2
  exit 2
fi
mkdir -p logs
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python -u scripts/train_uz05.py \
  --stage joint_tracking_startup --command-sampling startup_balanced --version "$VERSION" \
  "${INIT_ARGS[@]}" \
  --updates 150 --rollout-steps 256 --batch-size 256 --num-envs 4 --vec-env subproc \
  --device cpu --learning-rate 0.0001 --disable-adaptive-lr \
  --entropy-coef 0.001 --save-interval 50 --log-interval 1 --seed "$SEED" "$@" \
  2>&1 | tee "logs/${VERSION}.log"
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/eval_new_model_training.py \
  "checkpoints/$VERSION/checkpoint" --output "diagnostics/$VERSION/evaluation" \
  --launch-speeds .1 .25 .5 --seeds 64010 64011 64012 64013 64014 \
  2>&1 | tee "logs/${VERSION}_evaluation.log"
printf 'Checkpoint: checkpoints/%s/checkpoint.zip\nEvaluation: diagnostics/%s/evaluation\n' "$VERSION" "$VERSION"
