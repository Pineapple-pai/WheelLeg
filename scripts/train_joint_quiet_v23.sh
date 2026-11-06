#!/usr/bin/env bash
# Pure PPO: wheel velocity [L,R] + circular joint position [L1,L2,R1,R2].
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs
VERSION="${VERSION:-joint_tracking_pos_v16_armature001_$(date +%Y%m%d_%H%M%S)_$$}"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-wheelleg-train}"
# Pass additional train_uz05.py arguments after this script (last value wins).
exec conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python -u scripts/train_uz05.py \
  --stage joint_tracking_v16 --version "$VERSION" \
  --updates 6000 --rollout-steps 512 --batch-size 512 \
  --num-envs 8 --vec-env subproc --device cpu \
  --learning-rate 0.0001 --disable-adaptive-lr \
  --entropy-coef 0.001 --action-std 0.06 \
  --save-interval 500 --seed 0 "$@"
