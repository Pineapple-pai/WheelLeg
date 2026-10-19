# WheelLeg · UZ-05 MuJoCo 训练

当前主线是 MuJoCo 中的 UZ-05 两轮轮腿机器人，训练入口为
[`scripts/train_uz05.py`](scripts/train_uz05.py)。环境、策略接口、奖励和执行器分别位于
`scripts/uz05/`。Isaac Sim 旧实现已从当前版本移除。

## 开始训练

训练依赖安装在本机 Conda 环境 `sim` 中。站立训练只用 PPO 输出动作，没有手写平衡外环、残差策略、教师策略或辅助退火。站立 S0 示例：

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
conda run --no-capture-output -n sim python -u scripts/train_uz05.py \
  --stage stand --stand-level 0 \
  --updates 800 --rollout-steps 256 --batch-size 512 \
  --num-envs 8 --vec-env subproc --version stand_s0_v4 \
  > "logs/stand_s0_v4_$(date +%Y%m%d_%H%M%S).log" 2>&1
```

S0 学稳后逐级训练 S1、S2，并将上一级 PPO checkpoint 传给 `--checkpoint`：

```bash
conda run --no-capture-output -n sim python -u scripts/train_uz05.py \
  --stage stand --stand-level 1 --updates 800 --rollout-steps 256 \
  --batch-size 512 --num-envs 8 --vec-env subproc \
  --checkpoint checkpoints/stand_s0_v4/checkpoint --version stand_s1_v4

conda run --no-capture-output -n sim python -u scripts/train_uz05.py \
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
diagnostics/uz05.xml            当前 MuJoCo MJCF（已同步用户提供的腿部惯性修正版）
model_source/uz05/              SolidWorks 源装配及提取表格
docs/                           接口、模型、部署及训练审计
checkpoints/                    本机训练 checkpoint（生成目录）
runs/                           TensorBoard 输出（生成目录）
logs/                           本机运行日志
artifacts/                      保留的实验日志和复现记录
修改.md                         历史设计记录
```

`UZ05_VISUAL=1` 可启用 MJCF 的显示网格；常规训练会裁掉非碰撞显示网格。显示网格
需另行提供，目录由 `UZ05_MESH_DIR` 指定，默认查找 `model_source/uz05/mjcf_meshes/`。
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
