# 复现本次失败诊断

## 数据与代码身份

- 诊断入口：[trace_pure_ppo_diagnosis.py](../../scripts/trace_pure_ppo_diagnosis.py)。它只加载 PPO checkpoint 做 deterministic inference，不执行 optimizer/update，不写 checkpoint；六维动作只经过任务定义的 `[-1,1]` action space clipping，不施加其它动作修正。
- MuJoCo model XML：`uz05_修正后(3).xml`，SHA256 `22fc08a5f900ffc20d977c29851618fb0ca03c1d6b01dd2da0f32209ba505c54`。
- Drift-weight experiment task ID：`pure_ppo_stand_leg_scale_035_drift_weight_5x_v1`；所有诊断的腿映射 `0.35 rad`、drift reward `0.50`。
- Checkpoint：seed 0 `checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed0/best.zip` SHA256 `ec4c3c422103aab57cd01d10bc865e0693329ed7b93cd80c9b00b1c5cf1de89a`；seed 1 `.../seed1/best.zip` SHA256 `fd2ca93a7d507dab9269144ddbfdc20c3d161ff6465367a3cfd4b65d74f255c4`；seed 2 `.../seed2/best.zip` SHA256 `d378a718d95d0b842f3912275182c22f573508bafbe93de9a58bb58f058fad49`。
- 记录的 training commit 是 `ff19e52cf6dd1b309adb2c07e575ddd2be4d870e`；保存的训练 source fingerprint `77531bd427e59fff9dd7f53ac4bd1f4c992d86cff8fe82a9311b4da8eadb7799`，当前诊断源 fingerprint `36864f138974c210cafcb33e52e9db14d337939d72c387d54d7c305bdc9c2e45`。完整训练源码快照未归档；差别与模型/actuator contract 限制见 [ROOT_CAUSE_REPORT.md](ROOT_CAUSE_REPORT.md)。训练 source fingerprint 不应被当前工作树 hash 替代。
- checkpoint contract 与当前 actuator 合约相符：wheel speed KP `1.25 A/(rad/s)`、速度误差 clip `8 rad/s`、wheel current limit `10 A`、wheel torque limit `2.46 Nm`；leg PD `kp=100`、`kd=4`、torque limit `20 Nm`；policy step `8 ms`，motor step `2 ms × 4`。验收仍为 saturation fraction `≤1%` 且 max consecutive saturation `≤3`。
- Runtime：`/home/p/NavRL/isaac-training/yes/envs/sim/bin/python`，Python 3.10.20、PyTorch 2.2.2+cu118、MuJoCo 3.2.3、Stable-Baselines3 2.3.2。此解释器加载 torch 时需要 Isaac Sim 预装 NVIDIA shared libraries。

## 执行环境

在仓库根目录运行。示例输出目录使用 `/tmp` 并且必须是不存在的新目录；诊断脚本会拒绝覆盖已有目录。重跑时换一个输出路径。

```bash
ISAAC_NVIDIA=/home/p/isaac-sim-4.0.0/exts/omni.isaac.ml_archive/pip_prebundle/nvidia
SIM_PYTHON=/home/p/NavRL/isaac-training/yes/envs/sim/bin/python
export LD_LIBRARY_PATH="$ISAAC_NVIDIA/cuda_runtime/lib:$ISAAC_NVIDIA/cublas/lib:$ISAAC_NVIDIA/cudnn/lib:$ISAAC_NVIDIA/cufft/lib:$ISAAC_NVIDIA/curand/lib:$ISAAC_NVIDIA/cusolver/lib:$ISAAC_NVIDIA/cusparse/lib:$ISAAC_NVIDIA/nccl/lib:$ISAAC_NVIDIA/cuda_nvrtc/lib:$ISAAC_NVIDIA/cuda_cupti/lib:$ISAAC_NVIDIA/nvtx/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH=scripts
```

## 重放完整独立测试并与历史 summary 对照

seed 1，reset IDs 24100–24199：

```bash
"$SIM_PYTHON" scripts/trace_pure_ppo_diagnosis.py \
  --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed1/best.zip \
  --training-seed 1 \
  --task-id pure_ppo_stand_leg_scale_035_drift_weight_5x_v1 \
  --leg-scale 0.35 --drift-weight 0.5 \
  --seed-start 24100 --episodes 100 \
  --reference-episodes runs/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed1/test_all/id_episodes.jsonl \
  --output-dir /tmp/pureppo_diag_seed1_replay
```

seed 2，reset IDs 24200–24299：

```bash
"$SIM_PYTHON" scripts/trace_pure_ppo_diagnosis.py \
  --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed2/best.zip \
  --training-seed 2 \
  --task-id pure_ppo_stand_leg_scale_035_drift_weight_5x_v1 \
  --leg-scale 0.35 --drift-weight 0.5 \
  --seed-start 24200 --episodes 100 \
  --reference-episodes runs/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed2/test_all/id_episodes.jsonl \
  --output-dir /tmp/pureppo_diag_seed2_replay
```

预期各输出 `run_manifest.json` 中 `reference_comparison_mismatch_count=0`。本次已对 200 回合全部逐项完成此对照。生成的轨迹字段包括 Actor 原始均值、裁剪后和实际动作、轮速目标/实测、电流/力矩、腿目标/实测、接触、饱和 flag/run、drift 和 reward 分项。

## 复现 `.10` 与 `.50` 的固定 seed 0 checkpoint 配对行为

`.50` seed 0 checkpoint 用和历史 `.10` fresh seed 0 测试相同的 reset seed 23000–23009 做确定性推理。该 10 条用于行为对照，不作为正式 ID 接受率。

```bash
"$SIM_PYTHON" scripts/trace_pure_ppo_diagnosis.py \
  --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed0/best.zip \
  --training-seed 0 \
  --task-id pure_ppo_stand_leg_scale_035_drift_weight_5x_v1 \
  --leg-scale 0.35 --drift-weight 0.5 \
  --seed-start 23000 --episodes 10 \
  --output-dir /tmp/pureppo_diag_seed0_paired_23000
```

`.10` 对照历史 trace 在 `runs/pure_ppo_stand_leg_scale_035_v1/seed0/test_trace/step_trace.csv.gz`；完整配对轨迹见 `artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed0_paired_23000/`。

## 复现 seed 1 镜像 Pitch/rate 推理

以下生成 32 个 case：8 个相同 reset seed × 4 种显式 Pitch/rate 初始状态。相同 seed 的四个 case 除 Pitch/rate 外 reset 状态相同。重复的不同 seed 在该固定环境不引入物理参数随机化；因此它们用于检查配对结果可复现，不应解释为 8 个独立训练/泛化样本。

```bash
python3 - <<'PY'
import json, math
from pathlib import Path
rows = []
for seed in range(30000, 30008):
    for p_sign in (-1, 1):
        for r_sign in (-1, 1):
            rows.append({
                "seed": seed,
                "label": f"pair{seed}_p{p_sign:+d}_r{r_sign:+d}",
                "initial_pitch_rad": math.radians(2.5 * p_sign),
                "initial_pitch_rate_rad_s": 0.08 * r_sign,
            })
Path("/tmp/pure_ppo_diagnosis_seed1_mirror_states.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
PY

"$SIM_PYTHON" scripts/trace_pure_ppo_diagnosis.py \
  --checkpoint checkpoints/pure_ppo_stand_leg_scale_035_drift_weight_5x_v1/seed1/best.zip \
  --training-seed 1 \
  --task-id pure_ppo_stand_leg_scale_035_drift_weight_5x_v1 \
  --leg-scale 0.35 --drift-weight 0.5 \
  --initial-states-json /tmp/pure_ppo_diagnosis_seed1_mirror_states.json \
  --output-dir /tmp/pureppo_diag_seed1_mirror_pairs
```

本次执行成功，输出 32 个完整 episode、32,000 步；每个 base seed 的轨迹均复现同一方向模式：两种负 Pitch 条件失败，两种正 Pitch 条件通过。

## 饱和计数的独立检查逻辑

诊断脚本每个 8 ms policy step 以 `abs(current) >= 0.99 × 10 A` 或 `abs(wheel torque) >= 0.99 × 2.46 Nm` 标记 saturation。每个新策略步在该标志为真时 run 加 1，为假时立刻归零；再与 `episode_passes_acceptance()` 使用的阈值 `max_consecutive_steps > 3` 对照。本次 seed 1/2 的历史 JSONL 与重放 summary、step trace 重新算出的 wheel run/fraction 一致，无 off-by-one。

## 本次没有执行的操作

没有调用训练入口，没有加载 checkpoint 接续训练，没有更改权重、真实执行器限值、验收标准或历史配置，没有导出 ONNX。`output-dir` 下生成的是只读推理诊断数据。
