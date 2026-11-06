#!/usr/bin/env bash
# Bounded low-speed continuation followed by fixed-policy acceptance checks.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-wheelleg-train}"
SOURCE="${SOURCE:-checkpoints/joint_tracking_pos_v16_model20261104/checkpoint}"
VERSION="${VERSION:-joint_low_speed_$(date +%Y%m%d_%H%M%S)_$$}"
SEED="${SEED:-74}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
if [[ "$VERSION" == *'/'* || "$VERSION" == '.' || "$VERSION" == '..' ]]; then
  echo 'VERSION must be a single directory name.' >&2
  exit 2
fi
OUTPUT="checkpoints/$VERSION"
EVALUATION="diagnostics/$VERSION"
if [[ -e "$OUTPUT" || -e "$EVALUATION" ]]; then
  echo "Refusing to overwrite existing run: $VERSION" >&2
  exit 2
fi
for suffix in .zip _vecnormalize.pkl _contract.json; do
  if [[ ! -f "${SOURCE}${suffix}" ]]; then
    echo "Missing source: ${SOURCE}${suffix}" >&2
    exit 2
  fi
done
mkdir -p logs
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python -u scripts/train_uz05.py \
  --stage joint_tracking_v16 --checkpoint "$SOURCE" --version "$VERSION" \
  --vecnormalize-path "${SOURCE}_vecnormalize.pkl" --preserve-action-std \
  --command-sampling low_band --updates 300 \
  --rollout-steps 256 --batch-size 256 --num-envs 4 --vec-env subproc --device cpu \
  --learning-rate "$LEARNING_RATE" --disable-adaptive-lr \
  --entropy-coef 0.001 --save-interval 100 --log-interval 25 --seed "$SEED" \
  2>&1 | tee "logs/${VERSION}.log"
# New evaluation seeds; test all milestones, retain failures, never auto-expand speed.
for step in 100 200 300; do
  conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/eval_new_model_training.py \
    "$OUTPUT/checkpoint_iter_$step" --output "$EVALUATION/evaluation_$step" \
    --launch-speeds .25 .5 --seeds 64010 64011 64012 64013 64014 \
    2>&1 | tee "logs/${VERSION}_eval_${step}.log"
done
printf 'Checkpoint: %s/checkpoint.zip\nEvaluation: %s\n' "$OUTPUT" "$EVALUATION"
