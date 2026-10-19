# Reproduce the Pure PPO stand experiment

## Runtime

Use the recorded `sim` Conda environment. In this workspace the Python executable is `/home/p/NavRL/isaac-training/yes/envs/sim/bin/python`. It uses PyTorch `2.2.2+cu118` and an NVIDIA RTX 4060 Laptop GPU. Isaac Sim 4.0.0 provides the CUDA 11.8 shared libraries used by that wheel.

For this machine, set the following before Python commands:

```bash
export LD_LIBRARY_PATH=/home/p/isaac-sim-4.0.0/kit/kernel/plugins:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cuda_runtime/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cublas/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cudnn/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cusparse/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cusolver/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cufft/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/curand/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cuda_nvrtc/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cuda_cupti/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvtx/lib:/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nccl/lib
export PYTHONPATH=scripts
SIM_PYTHON=/home/p/NavRL/isaac-training/yes/envs/sim/bin/python
```

The runtime lacked two import dependencies used by SB3's logger path. The recorded additions were `matplotlib==3.8.4`, `pillow==11.3.0`, and `pytz==2025.2`.

## Gates and tests

Run these in order. Training refuses to start if a gate report is missing, failed, or stale for the selected MJCF/control code.

```bash
$SIM_PYTHON scripts/preflight_pure_ppo_stand.py
$SIM_PYTHON scripts/smoke_pure_ppo_stand.py
$SIM_PYTHON -m unittest discover -s tests -p 'test_pure_ppo_*.py' -v
$SIM_PYTHON scripts/train_pure_ppo_stand.py --dry-run --seed 0 --device cuda
```

The preflight compiles and resets 100 independent model instances, then checks closure, contacts, angular frames, zero action, all six action coordinates, wheel response/current/torque, and the 125/500 Hz timing. The smoke test runs 1000 random action transitions and checks both Gymnasium and SB3 vector-environment contracts.

## Formal training and evaluation

Each seed uses an independent fresh `PPO(...)` construction. Runs refuse to overwrite existing run/checkpoint directories.

```bash
$SIM_PYTHON scripts/train_pure_ppo_stand.py --seed 0 --updates 800 --num-envs 8 --device cuda
$SIM_PYTHON scripts/train_pure_ppo_stand.py --seed 1 --updates 800 --num-envs 8 --device cuda
$SIM_PYTHON scripts/train_pure_ppo_stand.py --seed 2 --updates 800 --num-envs 8 --device cuda
```

Each seed writes `config.json`, launch command, `metrics.csv`, per-episode JSONL, fixed validation records, `last.zip`, `best.zip`, and matching contract JSON beneath `runs/pure_ppo_stand_v1/` and `checkpoints/pure_ppo_stand_v1/`.

After each seed's training completes, evaluate its `best.zip` on 100 held-out ID episodes and the three separate stress distributions:

```bash
$SIM_PYTHON scripts/eval_pure_ppo_stand.py --checkpoint checkpoints/pure_ppo_stand_v1/seed0/best.zip --training-seed 0 --mode all --episodes 100
```

Repeat for seed 1 and 2 with their matching checkpoint and seed. Only a checkpoint with a passing independent ID report can be given to the ONNX exporter:

```bash
$SIM_PYTHON scripts/export_pure_ppo_stand_onnx.py --checkpoint checkpoints/pure_ppo_stand_v1/seed0/best.zip --training-seed 0 --evaluation-summary runs/pure_ppo_stand_v1/seed0/test_all/evaluation_summary.json
```

The exporter checks 1000 actual simulation observations and compares full MuJoCo episodes between SB3 and ONNX for every evaluation mode. It writes a versioned contract and refuses to overwrite an artifact.

## Evidence-driven follow-ups

The baseline three-seed outcome and the first fresh leg-scale ablation are documented in `TRAINING_REPORT.md` and `ABLATIONS.md`. The scale-only paired diagnostic can be reproduced with:

```bash
$SIM_PYTHON scripts/train_pure_ppo_leg_scale_ablation.py --seed 0 --updates 800 --num-envs 8 --device cuda
$SIM_PYTHON scripts/eval_pure_ppo_leg_scale_ablation.py --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_v1/seed0/best.zip --training-seed 0 --mode all --episodes 100
$SIM_PYTHON scripts/trace_pure_ppo_leg_scale_ablation.py --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_v1/seed0/best.zip --training-seed 0 --episodes 10 --output-dir runs/pure_ppo_stand_leg_scale_035_v1/seed0/test_trace
$SIM_PYTHON scripts/ablate_pure_ppo_leg_scale_levels.py --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_v1/seed0/best.zip --training-seed 0 --episodes 100 --output-dir runs/pure_ppo_stand_leg_scale_035_v1/seed0/scale_levels_counterfactual
```

The final one-factor training run holds the 0.35 rad action map fixed and uses `reward.position_drift=0.50` instead of `0.10`:

```bash
$SIM_PYTHON scripts/train_pure_ppo_leg_scale_drift_weight_ablation.py --seed 0 --updates 800 --num-envs 8 --device cuda
$SIM_PYTHON scripts/eval_pure_ppo_drift_weight_ablation.py --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed0/best.zip --training-seed 0 --mode all --episodes 100
$SIM_PYTHON scripts/train_pure_ppo_leg_scale_drift_weight_ablation.py --seed 1 --updates 800 --num-envs 8 --device cuda --reuse-gates
$SIM_PYTHON scripts/eval_pure_ppo_drift_weight_ablation.py --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed1/best.zip --training-seed 1 --mode all --episodes 100
$SIM_PYTHON scripts/train_pure_ppo_leg_scale_drift_weight_ablation.py --seed 2 --updates 800 --num-envs 8 --device cuda --reuse-gates
$SIM_PYTHON scripts/eval_pure_ppo_drift_weight_ablation.py --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed2/best.zip --training-seed 2 --mode all --episodes 100
```

`--reuse-gates` validates and reuses the same passing physical preflight, random-action smoke, and PPO dry-run reports; it does not skip checkpoint or evaluation checks. All configs, checkpoints, validation records, held-out episode JSONL, and evaluation summaries are retained under the matching experiment ID in `runs/`, `checkpoints/`, and `artifacts/`. Only seed 0 of the 5x drift-weight follow-up met independent ID acceptance, so the cross-seed criterion failed and no final ONNX was produced.
