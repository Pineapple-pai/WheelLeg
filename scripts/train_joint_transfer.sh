#!/usr/bin/env bash
# Explicitly initialized0.001 policy; bounded PPO adaptation with matched evaluation.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-wheelleg-train}"
export UZ05_MODEL_XML="${UZ05_MODEL_XML:-$REPO_DIR/models/uz05_20261105_armature001/uz05_rl.xml}"
DEFAULT_INIT_VERSION=armature001_transfer_legacy_20261106
DEFAULT_SOURCE="checkpoints/$DEFAULT_INIT_VERSION/checkpoint"
SOURCE="${SOURCE:-$DEFAULT_SOURCE}"
TARGET_STAGE="${TARGET_STAGE:-joint_tracking_startup_precision}"
VERSION="${VERSION:-joint_transfer_armature001_$(date +%Y%m%d_%H%M%S)_$$}"
SEED="${SEED:-0}"
read -r -a EVAL_SEED_ARGS <<< "${EVAL_SEEDS:-66300 66301 66302}"
if [[ "$VERSION" == *'/'* || "$VERSION" == '.' || "$VERSION" == '..' ]]; then
  echo 'VERSION must be a single directory name.' >&2; exit 2
fi
case "$TARGET_STAGE" in joint_tracking_startup|joint_tracking_startup_precision) ;; *) echo 'Unsupported foundation stage.' >&2; exit 2;; esac
for folder in "checkpoints/$VERSION" "diagnostics/$VERSION" "diagnostics/${VERSION}_source_evaluation"; do
  [[ ! -e "$folder" ]] || { echo "Refusing to overwrite: $folder" >&2; exit 2; }
done
mkdir -p logs
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python - <<'PY'
import sys
sys.path.insert(0,'scripts')
from uz05.spec import MODEL_XML,DEFAULT_MODEL_XML
if MODEL_XML.read_bytes()!=DEFAULT_MODEL_XML.read_bytes():
 raise SystemExit('This launcher requires the current0.001 plant. Set UZ05_MODEL_XML to models/uz05_20261105_armature001/uz05_rl.xml.')
PY
if [[ ! -f "${SOURCE}.zip" && "$SOURCE" == "$DEFAULT_SOURCE" ]]; then
  conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/initialize_model_transfer.py \
    --source checkpoints/joint_tracking_pos_v16/checkpoint --version "$DEFAULT_INIT_VERSION" \
    --allow-cross-model-initialization
fi
SOURCE_INFO="$(conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python - "$SOURCE" "$TARGET_STAGE" <<'PY'
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from train_uz05 import require_compatible_contract
p=Path(sys.argv[1])
if (p.parent/'INITIALIZATION_FAILED').exists():raise SystemExit('Incomplete initialization; refusing training.')
require_compatible_contract(p,sys.argv[2])
for suffix in ('.zip','_vecnormalize.pkl','_contract.json'):
 if not Path(str(p)+suffix).is_file():raise SystemExit('Missing source artifact: '+str(p)+suffix)
c=json.loads(Path(str(p)+'_contract.json').read_text())
initial=c.get('total_updates')==0 and 'initialization_provenance' in c
if initial:
 audit=json.loads(Path(str(p)+'_initialization.json').read_text())
 for suffix,expected in audit['output_files'].items():
  if hashlib.sha256(Path(str(p)+suffix).read_bytes()).hexdigest()!=expected:
   raise SystemExit('Initialization artifact changed: '+str(p)+suffix)
print(c['stage'],int(initial))
PY
)"
SOURCE_INFO="${SOURCE_INFO##*$'\n'}"
read -r SOURCE_STAGE SOURCE_IS_INITIAL <<< "$SOURCE_INFO"
STD_ARGS=(--preserve-action-std)
if [[ "$SOURCE_IS_INITIAL" == 1 ]]; then STD_ARGS=(--action-std 0.02); fi
NORMALIZATION_ARGS=(--vecnormalize-path "${SOURCE}_vecnormalize.pkl")
if [[ "$SOURCE_STAGE" != "$TARGET_STAGE" ]]; then NORMALIZATION_ARGS=(--reset-vecnormalize); fi
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/eval_new_model_training.py \
  "$SOURCE" --output "diagnostics/${VERSION}_source_evaluation" \
  --launch-speeds .1 .25 .5 --seeds "${EVAL_SEED_ARGS[@]}" \
  2>&1 | tee "logs/${VERSION}_source_evaluation.log"
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python - "diagnostics/${VERSION}_source_evaluation/results.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))['task_metrics']
if not all(x['posture_success'] for x in r):raise SystemExit('Source failed foundation posture coverage. Keep artifacts for diagnosis; training not started.')
PY
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python -u scripts/train_uz05.py \
  --stage "$TARGET_STAGE" --command-sampling startup_balanced --checkpoint "$SOURCE" \
  "${STD_ARGS[@]}" "${NORMALIZATION_ARGS[@]}" --version "$VERSION" \
  --updates 50 --rollout-steps 256 --batch-size 256 --num-envs 8 --vec-env subproc \
  --device cpu --learning-rate 0.00003 --disable-adaptive-lr --entropy-coef 0.001 \
  --save-interval 25 --log-interval 10 --seed "$SEED" \
  --initial-state-audit "diagnostics/$VERSION/initial_state.json" "$@" \
  2>&1 | tee "logs/${VERSION}.log"
conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/eval_new_model_training.py \
  "checkpoints/$VERSION/checkpoint" --output "diagnostics/$VERSION/evaluation" \
  --launch-speeds .1 .25 .5 --seeds "${EVAL_SEED_ARGS[@]}" \
  2>&1 | tee "logs/${VERSION}_evaluation.log"
if conda run --no-capture-output -n "$TRAIN_CONDA_ENV" python scripts/check_transfer_progress.py \
  --source "diagnostics/${VERSION}_source_evaluation/results.json" \
  --candidate "diagnostics/$VERSION/evaluation/results.json" \
  --output "diagnostics/$VERSION/progress_gate.json"; then
  echo "候选显示训练进展，请分别查看完整速度验收与后续随机化验证：checkpoints/$VERSION/checkpoint"
else
  gate_status=$?
  echo "本段未达到进展门槛，保留训练起点：$SOURCE；候选及诊断文件已保存。" >&2
  exit "$gate_status"
fi
