# WheelLeg · UZ-05 MuJoCo 训练

当前主线是 MuJoCo 中的 UZ-05 两轮轮腿机器人，训练入口为
[`scripts/train_uz05.py`](scripts/train_uz05.py)。环境、策略接口、奖励和执行器分别位于
`scripts/uz05/`。Isaac Sim 旧实现已从当前版本移除。

## 开始训练

默认物理模型是 [`models/uz05_20261105_armature001/uz05_rl.xml`](models/uz05_20261105_armature001/uz05_rl.xml)：
按用户指定，将左右轮侧附加惯量设为0.001kg·m²，保留其它新版物理与控制参数。
这是待标定候选，资料与计算边界见[改装M3508惯量说明](docs/wheel_armature001_20261105.md)。
原始0.01模型、网格与全部历史checkpoint保留；旧权重不能当作当前同模型直接续训。
当前已验证可用的CPU环境是 `wheelleg-train`，训练脚本默认使用它；可用 `TRAIN_CONDA_ENV` 覆盖。

2026-11-06的[0.001模型训练复盘](diagnostics/armature001_training_review/report.md)：
从头1000轮的回合约0.73秒，未进入3秒后的起步阶段；不要原样追加长训。
已新增显式跨模型参数初始化，成熟v16初始化后在标称与域随机化测试中各21/21回合姿态通过。
速度仍有方向偏差，延长短训也发现精度退化，尚无全任务5%验收通过的升级模型。
下一段使用[带来源审计和配对验收的训练入口](docs/armature001_transfer_training_20261106.md)
`bash scripts/train_joint_transfer.sh`，默认50轮、8环境、rollout256、batch256。
未通过进展门槛退出3，保存诊断并保留来源；不能把训练奖励提高当作模型升级。
该50轮配方的[实际运行复盘](diagnostics/joint_transfer_armature001_20261106_095202/report.md)：
姿态21/21保留、归一化速度误差下降30.65%，但-0.25与-0.5退化，25轮保存点也未通过。
当前入口保留作复现，不建议原样重跑或从本次候选继续长训。

新旧模型退化的[最新单项对照定位](docs/model_switch_causes_20261105.md)：固定旧策略时，
仅将新模型轮轴armature从0.01回退至0.0001，±2m/s起步存活由各0/3恢复为各3/3；
单独恢复旧云台惯量/质心未恢复起步。应先核实轮侧反射惯量，再启动新长训。
该结果不证明旧参数符合硬件或速度精度已合格；当前采用用户另行指定的0.001候选。
主训练shell默认输出带armature001与时间戳的新版本；以下训练成绩均对应此前0.01模型，
复现时需设置`UZ05_MODEL_XML="$PWD/models/uz05_20261104/uz05_rl.xml"`。
该版本6000轮训练后的[效果分析](diagnostics/new_model_training_effect/report.md)：固定站立通过，
非零起步仍快速失稳，尚不能作为运动策略使用。

最新[速度精度1000轮复盘](diagnostics/precision_completed_review/report.md)：13个独立标称种子中，
+0.25m/s综合通过12/13，站立姿态12/13；反向起步、±0.1精度、±0.5及域随机化尚未通过。
混合缓升与较小学习率的配对短训均未形成可靠整体升级，保留`joint_precision_20261105_184905/checkpoint`，
不建议直接追加长训。[实验入口与验收保护](docs/precision_review_20261105.md)默认仅25轮，
先后配对评估源与候选，发现已学能力退化或没有新增速度成功时退出3，不自动替换源权重。
历史[5000轮复盘](diagnostics/startup_5000_review/report.md)与
[上一轮速度精度训练配方](docs/startup_precision_20261105.md)保留供复现；后者1000轮已完成，不再作为未测推荐。
此前的[基础站立与起步](docs/startup_training_20261105.md)及[150轮短训](diagnostics/startup_warmstart/report.md)保留供复现。
此前的[300轮低速续训](docs/low_speed_refinement_20261105.md)保留作实验复现，不作为已验证修复。

此前的[奖励简化与2×2对照设计](docs/reward_simplification.md)供独立实验参考。
独立实验入口`python3 scripts/compare_simple_rewards.py`默认只生成计划，
不会启动训练；简化基线与可选同轴项尚未完成正式效果比较。

旧模型上的短训优化、失败候选及v17训练命令见
[速度校准与困难恢复采样](docs/optimization_20261104_priority.md)。
尚无全面通过验收的升级模型，训练配方与已验证性能请分别看待。

需要从第一步同时训练高度、vx、yaw 时，执行 `bash scripts/train_joint_quiet_v23.sh`。
当前默认是 v16，仍覆盖 ±2m/s、0.22–0.32m 车身高度，见
[v15 反向车身倾斜与错轴复测、v16 修改与续训命令](docs/joint_tracking_v16.md)。
v15仍有切换时摆腿、车身倾斜与整机质心倾斜方向相反、等腿长但轮轴错位的问题。
v16将车身零pitch目标改为与整机支撑倾角对齐，并直接约束真实轮轴偏移，
辅以较弱的周期关节目标协调成本。速度目标保持，完整训练后仍需联合验收。
它保留固定的漂移、平滑、饱和及抗扰目标。[v6 的 6000 轮实测与回放](docs/joint_tracking_pos_v6_analysis.md)
显示固定命令与 pitch 明显改善，运动停止后的锚点保持仍未达标。此前 v5 的结果见
[v5 实测分析与回放](docs/joint_tracking_pos_v5_analysis.md)。
六维动作为左右轮独立速度加四关节周期位置；完整命令、范围和回放方式见
[联合训练说明](docs/joint_tracking.md)。5000 轮结果、视频和原生 MuJoCo 回放命令见
[v1 评估报告](docs/joint_tracking_pos_v1_evaluation.md)。姿态修正版的新结果见
[v2 训练与复测分析](docs/joint_tracking_pos_v2_analysis.md)。最新 v4 结果与回放见
[v4 分析报告](docs/joint_tracking_pos_v4_analysis.md)。以下为保留的旧站立课程。

当前训练使用本机 Conda 环境 `wheelleg-train`。站立训练只用 PPO 输出动作，没有手写平衡外环、残差策略、教师策略或辅助退火。站立 S0 示例：

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
conda run --no-capture-output -n wheelleg-train python -u scripts/train_uz05.py \
  --stage stand --stand-level 0 \
  --updates 800 --rollout-steps 256 --batch-size 512 \
  --num-envs 8 --vec-env subproc --version stand_s0_v4 \
  > "logs/stand_s0_v4_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

S0 学稳后逐级训练 S1、S2，并将上一级 PPO checkpoint 传给 `--checkpoint`：

```bash
conda run --no-capture-output -n wheelleg-train python -u scripts/train_uz05.py \
  --stage stand --stand-level 1 --updates 800 --rollout-steps 256 \
  --batch-size 512 --num-envs 8 --vec-env subproc \
  --checkpoint checkpoints/stand_s0_v4/checkpoint --version stand_s1_v4

conda run --no-capture-output -n wheelleg-train python -u scripts/train_uz05.py \
  --stage stand --stand-level 2 --updates 800 --rollout-steps 256 \
  --batch-size 512 --num-envs 8 --vec-env subproc \
  --checkpoint checkpoints/stand_s1_v4/checkpoint --version stand_s2_v4
```

每个版本的策略、VecNormalize 统计量和接口契约都保存在 `checkpoints/<version>/`，TensorBoard 数据保存在 `runs/<version>/`。倾倒判定对正、负俯仰使用相同绝对角度边界，并给 40 个控制帧（320 ms）恢复时间；S2 的倾角边界为 11.5°。训练日志仍要求最近 50 个完整 episode 存活率达到 95%、最大漂移不超过 5 cm、偏航不超过 5°、三轴角速度 RMS 不超过 0.20 rad/s，且峰值倾角不越过当前课程边界，才显示 `accept_stand: 1`。当前契约为 `uz05_iface_v5`，除观测/动作接口外还校验执行器参数；v4 及更早 checkpoint 不兼容。

## 已训练站立 checkpoint

历史 checkpoint [`checkpoints/stand_nominal_v9/`](checkpoints/stand_nominal_v9/) 使用旧轮速环参数，满动作时仅输出 0.405 A，不能作为当前站立策略。当前回放候选为 [`stand_negative_rate_refine_v1/checkpoint.zip`](checkpoints/stand_negative_rate_refine_v1/checkpoint.zip)，使用 v5 契约。名义动力学、正负 ±0.08 rad 初态的 50 回合压力回放中，45 回合跑满 1000 步；其余 5 回合只因站位略超 5 cm（5.01–5.04 cm）终止，峰值倾角最高 10.06°、偏航峰值低于 5°。默认网页种子 91000 可跑满 1000 步，最大站位偏移 0.98 cm、峰值俯仰 2.86°。这版仍未满足 95% 全部验收条件，压力回放中角速度 RMS 的回合最大值为 0.325 rad/s，高于 0.20 rad/s 验收值。

## 当前接口

| 接口 | 维度 | 内容 |
|---|---:|---|
| 策略 actor 观测 | 38 | 当前单帧观测，含相对初始航向误差；无历史帧堆叠 |
| critic 专属特权观测 | 56 | 精确线速度/位移、腿长、接触、地形、力矩和域参数 |
| PPO 环境观测总计 | 94 | 当前 actor 观测 + 单帧 critic 特权观测 |
| 动作 | 6 | 共模轮速 1、差模轮速 1、腿关节位置偏置 4 |

训练中的策略只读取前 38 维；critic 读取全部 94 维。各阶段保持相同的张量尺寸。
actor 不再接收仿真任务标签；速度、位移、高度和地形输入仍是带噪仿真传感器代理，详见
[观测与任务适配审计](docs/observation_training_audit.md)。

## 目录

```text
scripts/
  train_uz05.py                 PPO 训练入口和 checkpoint 接口契约
  uz05/
    spec.py                     观测、动作、奖励、课程、物理参数
    policy.py                   非对称 actor-critic 网络
    model.py                    MJCF 加载、索引、接触、地形和域随机化
    actuators.py                PPO 轮速/关节位置目标的执行器模型
    observations.py              命名观测块、缩放/噪声和维度校验（无历史堆叠）
    env.py                      Gymnasium 环境编排、奖励和终止
  validate_uz05_deployment_data.py  外部实测 YAML 的部署数据校验器
models/uz05_20261104/            当前新版训练 MJCF 与 meshes/stl 显示资产
uz05_修正后(3).xml              旧训练模型（保留用于历史复现）
model_source/uz05/              SolidWorks 源装配及提取表格
docs/                           接口、模型、部署及训练审计
checkpoints/                    本机训练 checkpoint（生成目录）
runs/                           TensorBoard 输出（生成目录）
logs/                           本机运行日志
artifacts/                      保留的实验日志和复现记录
修改.md                         历史设计记录
```

`UZ05_VISUAL=1` 可启用 MJCF 的显示网格；常规训练会裁掉非碰撞显示网格。显示网格
目录由 `UZ05_MESH_DIR` 指定，默认查找 `models/uz05_20261104/meshes/stl/`。
`UZ05_MODEL_XML` 可覆盖模型路径，例如复现旧checkpoint时设为根目录的 `uz05_修正后(3).xml`；
该环境变量必须在启动Python之前设置，也会传给训练子进程。历史诊断命令需显式选择旧模型。
部署数据校验器也需要外部实测 YAML；当前仓库不含该 YAML 模板。
完整能力课程配置在 `scripts/uz05/spec.py`，但仓库当前没有自动串联所有阶段的 shell 脚本。

## 训练与模型边界

- 课程顺序：`stand → low_speed → high_speed → steering → rotation → airborne → stairs → jump → recovery`。
- 站立 S2 以纯 PPO 达到全程最大前后漂移 5 cm、相对初始航向 5°以内；训练从 S0 开始逐级收紧边界。
- 前后平移、旋转、跳跃和台阶的输入已纳入 actor；侧向命令虽预留但当前课程不采样，轮式底盘也不具备
  直接横移能力。高度/地形传感器代理、跳跃事件奖励和站立奖励仍有限制，不能仅凭张量维度固定就判定实机可部署。
- 电机增益、机械限位、摩擦和随机化范围中仍有手册值或待实测占位。实机部署结论见
  [模型与实物部署审计](docs/model_deployment_audit.md)。

## 文档

- [观测与任务适配审计](docs/observation_training_audit.md)：逐项评估站立、平移、旋转、跳跃等任务。
- [重构说明与训练配置](docs/uz05_refactor.md)：接口、坐标系、执行器、奖励和课程。
- [模型与实物部署审计](docs/model_deployment_audit.md)：模型可信度和实机阻塞项。
- [文档索引](docs/README.md)
