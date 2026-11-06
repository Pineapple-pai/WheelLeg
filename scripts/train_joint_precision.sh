#!/usr/bin/env bash
# Velocity-precision continuation from the more stable measured startup snapshot.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-wheelleg-train}"
SOURCE="${SOURCE:-checkpoints/joint_stand_translation_20261105_161432/checkpoint_iter_1000}"
VERSION="${VERSION:-joint_precision_$(date +%Y%m%d_%H%M%S)_$$}"
SEED="${SEED:-0}"
read -r -a EVAL_SEED_ARGS <<< "${EVAL_SEEDS:-65300 65301 65302 65303 65304}"
if [[ "$VERSION" == *'/'* || "$VERSION" == '.' || "$VERSION" == '..' ]]; then
  echo 'VERSION must be a single directory name.' >&2; exit 2
fi
for suffix in .zip _vecnormalize.pkl _contract.json; do
  if [[ ! -f "${SOURCE}${suffix}" ]]; then
    echo "Missing source: ${SOURCE}${suffix}" >&2; exit 2
  fi
done
if [[ -e "checkpoints/$VERSION" || -e "diagnostics/$VERSION" ]]; then
  echo "Refusing to overwrite existing run: $VERSION" >&2; exit 2
fi
# Reset reward RMS only when the objective changes. Same-stage continuation keeps it.
INIT_ARGS=(--checkpoint "$SOURCE" --preserve-action-std --reset-vecnormalize)
SOURCE_STAGE="$(conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["stage"])' "${SOURCE}_contract.json")"
if [[ "$SOURCE_STAGE" == joint_tracking_startup_precision ]]; then
  INIT_ARGS=(--checkpoint "$SOURCE" --preserve-action-std --vecnormalize-path "${SOURCE}_vecnormalize.pkl")
fi
mkdir -p logs
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python -u scripts/train_uz05.py \
  --stage joint_tracking_startup_precision --command-sampling startup_balanced --version "$VERSION" \
  "${INIT_ARGS[@]}" --updates 1000 --rollout-steps 256 --batch-size 256 \
  --num-envs 4 --vec-env subproc --device cpu --learning-rate 0.0001 --disable-adaptive-lr \
  --entropy-coef 0.001 --save-interval 250 --log-interval 1 --seed "$SEED" "$@" \
  2>&1 | tee "logs/${VERSION}.log"
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/eval_new_model_training.py \
  "checkpoints/$VERSION/checkpoint" --output "diagnostics/$VERSION/evaluation" \
  --launch-speeds .1 .25 .5 --seeds "${EVAL_SEED_ARGS[@]}" \
  2>&1 | tee "logs/${VERSION}_evaluation.log"
printf 'Checkpoint: checkpoints/%s/checkpoint.zip\nEvaluation: diagnostics/%s/evaluation\n' "$VERSION" "$VERSION"
