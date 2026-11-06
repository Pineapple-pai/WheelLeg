# v7：实际腿长、前后构型和 roll 的软约束

启动脚本现默认 v11，见 [v11 平移与姿态配置](joint_tracking_v11.md)。复现本页 v7 时需设置
`VERSION=joint_tracking_pos_v7` 并显式添加 `--stage joint_tracking_v7`。

阶段 `joint_tracking_v7`，启动脚本默认输出 `checkpoints/joint_tracking_pos_v7/`。
实现、38 项测试、短训练与奖励审计已完成，正式 6000 轮训练尚未启动。
本页说明配置，不代表对称支撑或抗扰恢复已经达标。

## 为什么需要三种量

[v6 实测](joint_tracking_pos_v6_leg_symmetry.md)中，高位站立左右有效腿长约 23.60/24.85cm，
roll 约 −2.13°，左右轮相对髋部的前后位置相差约 3.81cm。
仅有较小 pitch、双轮接地或竖向投影接近，都不能排除这些不对称构型。

v7 直接约束以下几何量：

* **实际腿长差**：左右 hip 站点到 wheel 站点的三维距离之差。
* **前后构型错位**：左右轮心相对各自髋部、在机体系 X 方向偏移之差。
  只要求左右接近，不指定两条腿必须前伸或后缩到某个位置。
* **roll**：机身侧倾角，正负方向完全对称。

左右轮速度与四个关节位置仍由 PPO 独立输出。奖励没有读取或规定左右关节角相等，
也没有把同一动作复制到另一侧。关节仍按周期位置处理，允许 360° 旋转。

## 固定软目标

通用代价为：

```text
x = max(0, abs(error) - tolerance) / scale
cost = -weight * H(x)
H(x) = x²（x <= 1），否则 2*x - 1
```

| 项目 | 容差 | 归一化尺度 | 固定权重 |
|---|---:|---:|---:|
| 实际腿长差 | 3mm | 10mm | 0.35 |
| 左右前后构型错位 | 5mm | 20mm | 0.25 |
| roll 偏差 | 1° | 4° | 0.25 |
| 竖向落差差（沿用并收紧） | 3mm | 15mm | 0.25 |

roll 的最高 1 分正收益仍保留，但指数尺度由 12° 改为 5°，鼓励接近水平；
pitch 奖励与偏移代价沿用 v6，最高正向任务收益仍为 9.5/步。
vx、yaw、高度、停止锚点、支撑、平滑、实际轮速变化和饱和目标均保留。

上述权重从第一批采样起固定，在站立、前进、倒退、转向、起停和受扰时都计算。
容差内不扣对应偏差代价，大偏差代价线性增长，不设置几何硬边界、不修改电机目标。
策略可以选择付出暂时的代价来恢复平衡；这保留了动作空间，但恢复成功率需要训练后验证。

v6 的原有较宽几何保护仍保留，新目标补足小幅但持续的不对称。
没有通过收紧旧几何乘子来同时大幅削减所有跟踪收益。
观测噪声、执行器参数、500Hz 执行环、动力学随机化与随机推力均沿用 v6。
全范围高度/vx/yaw 和长短命令分布从第一步同时启用，无课程或退火。

## 同轨迹量级审计

冻结 v6 最终 PPO，使用 v7 的名义环境，种子 20000；每种命令 6 秒，排除首秒。
两版观测和物理相同，仅奖励定义变化，因此可在相同轨迹上比较姿态得分。
审计脚本不会训练 PPO，也不会向正式训练提供旧策略动作或数据。

| 旧策略轨迹 | 新对称代价 / 正向任务收益 |
|---|---:|
| 0.245m 站立 / 前进 | 4.7% / 7.5% |
| 0.270m 站立 / 前进 | 5.4% / 5.0% |
| 0.295m 站立 / 前进 | 12.3% / 5.5% |
| 0.270m 左转 / 混合 | 5.4% / 1.9% |

对称代价分组包括实际腿长差、前后错位、roll 负代价和竖向落差差；
roll 正收益收紧造成的变化另有记录。
高位站立相对 v6 的总姿态得分变化约 −1.225/步，形成了明确优化压力，
但这些轨迹中未出现对称代价压过任务收益。该审计没有覆盖全部严重扰动和恢复状态。

[原始审计](../logs/joint_tracking_v7_symmetry_audit.json)可重现：

```bash
conda run --no-capture-output -n sim python scripts/audit_support_symmetry.py
```

38 项测试覆盖：几何腿长与前后错位分别产生代价、容差、左右交换与 roll 正负对称性、
大偏差线性代价、所有命令模式都执行目标，以及六路不相等动作不会被覆盖。
还完成 30 次更新从头训练、保存与独立评估入口检查，以及两次更新的 v6 迁移链路检查。
短跑模型仍会很快倒下，不能用它判断正式学习效果。

## 训练

默认从头训练，保留 v6 模型作为基线：

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh 2>&1 | tee logs/joint_tracking_pos_v7.log
```

默认 6000 次更新、8 个环境、每环境 rollout 512，共 24,576,000 个采样步。
学习率固定 1e-4、熵系数固定 0.001，初始动作标准差 0.06，之后由 PPO 自己学习。
每 100 轮保存，最终模型位于 `checkpoints/joint_tracking_pos_v7/checkpoint`。
没有教师、残差、动作滤波或平衡外环；策略输出两轮速度和四关节周期位置。

v6→v7 的观测和执行器契约兼容，技术上可以迁移，但跨版本必须加
`--reset-vecnormalize` 重置奖励归一化；默认命令没有加载旧模型。
checkpoint 的 `precision_objective` 记录全部实际生效的软目标参数。

## 日志与验收

新增训练日志：

* `Support/leg_length_difference_rms_m`
* `Support/fore_aft_difference_rms_m`
* `Posture/roll_rms_deg`
* `Posture/roll_within_2deg_fraction`
* `Reward/symmetry_cost_mean`

原有 pitch、速度/高度跟踪、支撑、漂移、输出变化和饱和指标继续记录。
几何长度差与 roll 用 RMS，避免左右/正负偏差平均后相消。

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v7/checkpoint --duration 24 --episodes-per-case 3 \
  --output logs/joint_tracking_pos_v7_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v7/checkpoint --duration 24 --episodes-per-case 3 \
  --domain-randomization --perturbations --output logs/joint_tracking_pos_v7_stress_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_random.py \
  checkpoints/joint_tracking_pos_v7/checkpoint --episodes 10 \
  --output logs/joint_tracking_pos_v7_random_eval.json

bash scripts/replay_joint_tracking.sh --keyboard --nominal \
  --checkpoint checkpoints/joint_tracking_pos_v7/checkpoint
```

键盘：1/2 前进/后退，3/4 左/右转，5/6 降低/升高 1cm 并进入站立调高度模式，
7 站立，R 重置。`- / +` 调当前模式参数：速度每次 0.05m/s、转速每次 0.1rad/s、
站立/高度每次 1cm。后退和右转时 `+` 增大速度绝对值；主键盘 `=` 也可加，支持小键盘加减。
切换保留已调数值，数值限制在训练范围内。旧回放窗口需关闭后重启，不必重新训练。

回放默认采用浅色天空和带轻微反射的棋盘地面，仅改变渲染资产。窗口顶部三行小字显示：
pitch/yaw/roll（度）、机体系 vx 实际/目标（m/s）、yaw rate 实际/目标（rad/s），
以及相对地面高度实际/目标、模式与时间。Yaw 是世界航向角，范围约 −180° 至 +180°；
yaw rate 是转向角速度，两者分开标注。

交互回放使用原生 `mujoco.viewer.launch_passive`，左右 File/Joint/Control 等侧栏均保留。
本机 MuJoCo 3.2.x 没有自定义屏幕叠字接口，顶部信息通过原生 viewer 的视觉标签显示，
按当前摄像机位置更新。鼠标仍采用原生视角操作；摄像机跟随机器人并保留手动平移偏移。
视频导出也显示同一组实时量。
背景不改变碰撞、摩擦、质量或训练接口，不需重训。

按三个高度和正反方向检查稳定段实际腿长差、竖向差、前后错位及 roll，
并复测 v6 已有的 24 秒存活与速度跟踪，不能用对称换取倒下、欠速或失去恢复能力。
建议平地稳态先以腿长差约 5mm、前后错位约 10mm、roll 约 2°作为筛查目标，
随后扩大种子并结合完整时序检查；这些是待验证目标，不是已达到成绩。
停止后的 5cm/5° 锚点保持、受扰恢复与轮腿高频输出仍应一并验收。
