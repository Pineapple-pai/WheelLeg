# Pure PPO 站立失败根因诊断

## 结论摘要

本次只做了日志审计和固定 checkpoint 的确定性 MuJoCo 重放；没有启动训练、没有改 checkpoint/历史测试结果，也没有导出 ONNX。

- 原始 `pure_ppo_stand_v1` 的正式 ID 结果复核为 seed 0：0/100、seed 1：100/100、seed 2：0/100。
- 原始 seed 0 checkpoint 在同一批 100 个 ID seed 上仅把腿目标映射尺度从 0.50 rad 改为 0.35 rad 后为 100/100。这是推理对照，不是训练结果。
- 随机初始化、0.35 rad、`position_drift=0.10` 的 seed 0 正式结果是 0/100。
- 随机初始化、0.35 rad、`position_drift=0.50` 的正式结果是 seed 0：100/100，seed 1：49/100，seed 2：92/100。当前可靠性仍不满足跨种子要求。
- seed 1 的 51 个失败都触发 `wheel_saturation_duration`；其中 8 个还触发 `planar_drift`。所有 51 个失败回合在第 1 个策略步后左轮失去接触，首次轮部饱和在第 3 个策略步；8 个漂移失败在第 26–28 步才超过 1 cm，随后在第 53–67 步超过 5 cm。首发异常顺序是左轮接触丢失、轮电流饱和、平面漂移增长。
- seed 2 的全部 8 个失败都是左轮电流饱和连续 4 个策略步（步 2–5）。连续步数实现和 JSONL 汇总均正确，失败是既定 `4 > 3` 验收结果；不是 off-by-one。
- 对 seed 1 checkpoint 追加了 32 次固定初态重放（4 组 `±2.5°` Pitch × `±0.08 rad/s` Pitch rate，重复 8 组固定 reset seed）。每一组负 Pitch 都失败、正 Pitch 都通过；重复轨迹完全一致。该结果确认了此 checkpoint 在这组受控初态下存在 Pitch 符号相关的失败机制，但重复次数不是独立样本，不能推成跨分布成功率。

## 实验、权重、配置与源码审计

所有正式模型的记录模型 SHA256 都是 `22fc08a5f900ffc20d977c29851618fb0ca03c1d6b01dd2da0f32209ba505c54`。训练 seed 0/1/2 的 `best.zip` SHA256 均与独立评估 JSON 的 `checkpoint_sha256` 一致，`best_contract.json` SHA256 也与评估 JSON 一致。三个 drift-weight checkpoint 分别为：

| 实验 | Seed / 最佳 update | 映射 / 漂移权重 | 独立 ID seeds / 通过 | best checkpoint SHA256 | 训练 wrapper SHA256 |
|---|---:|---|---|---|---|
| `pure_ppo_stand_v1` | 0 / 650 | 0.50 rad / 0.10 | 20000–20099 / 0/100 | `8eed2d85de19f17937e34326e7219eeb043e02dbef1c6f7087cb3f6cea982029` | 基础入口；见 `launch.json` |
| `pure_ppo_stand_v1` | 1 / 750 | 0.50 rad / 0.10 | 21000–21099 / 100/100 | `b11f76f6b8f67c1e014fed2f68020e010c900910436300020e482133465b407b` | 基础入口；见 `launch.json` |
| `pure_ppo_stand_v1` | 2 / 700 | 0.50 rad / 0.10 | 22000–22099 / 0/100 | `470f2f125ac4d10d7a91d7d9bc5634e68141df0eaa1f5a94abbfe2824f9f6cbc` | 基础入口；见 `launch.json` |
| `pure_ppo_stand_v1` 的 seed 0 推理对照 | 原 seed 0 checkpoint | 0.35 rad / 0.10 | 同一批 20000–20099 / 100/100 | 沿用 `8eed…2029` | `retrained=false`；非训练结果 |
| `pure_ppo_stand_leg_scale_035_v1` | 0 / 800 | 0.35 rad / 0.10 | 23000–23099 / 0/100 | `12653840c98b095d0c63b7df36c1e0878fc3e8477217b41cc4ed4259d600ab28` | `8b1d57a77c5a7392ba455372bdce69e4211459b04baae7297b10df15e08f275e` |
| `pure_ppo_stand_leg_scale_035_drift_weight_5x_v1` | 0 / 800 | 0.35 rad / 0.50 | 24000–24099 / 100/100 | `ec4c3c422103aab57cd01d10bc865e0693329ed7b93cd80c9b00b1c5cf1de89a` | `a8a3ddd8582b28cf68367152ccce5e69307e48359f36b9764373b39ba482fa80` |
| 同上 | 1 / 650 | 0.35 rad / 0.50 | 24100–24199 / 49/100 | `fd2ca93a7d507dab9269144ddbfdc20c3d161ff6465367a3cfd4b65d74f255c4` | `b1edda8a76a1139814d4a51581a061f8ca29ea21245e7c81bec74c790b8fb84b` |
| 同上 | 2 / 550 | 0.35 rad / 0.50 | 24200–24299 / 92/100 | `d378a718d95d0b842f3912275182c22f573508bafbe93de9a58bb58f058fad49` | `b1edda8a76a1139814d4a51581a061f8ca29ea21245e7c81bec74c790b8fb84b` |

上述记录的 source fingerprint 均为 `77531bd427e59fff9dd7f53ac4bd1f4c992d86cff8fe82a9311b4da8eadb7799`。测试入口 `scripts/eval_pure_ppo_drift_weight_ablation.py` 的记录 SHA256 为 `5a4a9396cc6f730ffd7cfa5ffa107ef57034628c199343c9d0c9efa55cc250de`。训练 wrapper 的 seed 0 与 seed 1/2 SHA 不同；保存的命令显示 seed 1/2 加了 `--reuse-gates`，seed 0 没有。三者实验配置中的任务因子、800 更新目标、随机权重初始化、模型 SHA、验收指标和 source fingerprint 相同；seed 0 wrapper 的原始字节快照未保存，只有配置记录的 SHA 和命令可核对。

**源码身份限制：**运行配置的 Git commit 是 `ff19e52cf6dd1b309adb2c07e575ddd2be4d870e`，但该干净 commit 的源码路径/哈希不能单独复现实验：该 commit 的 `spec.py` 指向 `diagnostics/uz05.xml`，当前该文件 SHA 是 `3d9713a373edfca6cd9c77f3304d0df71ef7970e230e2f9a7525f88d764b51a2`；与训练记录匹配的 SHA `22fc…05c54` 对应工作区 `uz05_修正后(3).xml`。checkpoint 的 actuator contract 记录 `speed_kp=1.25 A/(rad/s)`、speed error clip `8 rad/s`、wheel current limit `10 A`、wheel joint torque limit `2.46 Nm`、leg PD `kp=100`、`kd=4`、torque limit `20 Nm`，均与当前 Pure PPO 运行所加载的参数一致。当前工作树 fingerprint 是 `36864f138974c210cafcb33e52e9db14d337939d72c387d54d7c305bdc9c2e45`；原训练 fingerprint 对应的源码文件快照没有另存。既有训练报告称其差别仅为 `pure_ppo_eval.py` 末尾空白行。本次重放模型 SHA、actuator contract 与记录一致，且 seed 1/2 的 200 个逐回合汇总字段及失败原因均与历史 JSONL 一致；但重放是新生成的轨迹，不能声称拥有历史原始逐步日志。

## 已确认的问题与证据

### seed 1：初始姿态符号相关的左轮接触丢失及电流饱和

seed 1 的 100 条记录终止原因全部是 `time_limit`，所有失败是验收失败而非提前摔倒。按 signed pitch 划分，pitch `<0` 有 47 条、6 条通过；pitch `>=0` 有 53 条、43 条通过。按报告常用的三个绝对角区间划分：`[-3,-1)°` 为 0/31，`[-1,1)°` 为 16/33，`[1,3]°` 为 33/36。负 pitch 的 31 条大角样本全部失败，8 条 `planar_drift` 也全部在该组；另外 10 条 wheel-duration 失败来自 near-zero 17 条和 positive 3 条。

首次动作响应不是饱和。失败组 51/51 在第 1 个 8 ms 策略步后左轮接触为 0；第 3 步才首次出现轮饱和。每个失败的轮饱和比例最高为 0.7%，低于 1% 比例验收线；失效来自最长持续时间 4–7 步。51 个失败均无 leg torque saturation 验收失败，轮部 torque saturation 比例为 0。8 个漂移失败都先在第 3 步发生轮饱和，约第 26–28 步才超过 1 cm，第 53–67 步才超过 5 cm，峰值漂移为 0.1011–0.1385 m。也就是说，漂移是后发结果；第 1 步的左轮接触丢失更早。

已找到绝对初始 Pitch 和 Pitch rate 极接近的可比回合：失败 24132 为 `(-2.912°, +0.0903 rad/s)`，通过 24185 为 `(+2.825°, +0.0919 rad/s)`；其绝对 Pitch 差仅 0.088°，Pitch-rate 差 0.0016 rad/s，其他已知 reset 状态相同。第 1 步 Actor 六维输出分别为 `[-0.157, 0.576, -0.070, 0.155, -0.259, -0.254]` 和 `[-0.290, 0.419, -0.086, -0.002, -0.341, -0.303]`。在未发生电流饱和前，失败回合第 1 步后左轮脱离接触；后续左轮在第 3–7 步连续限流。通过回合只在第 2 步出现右轮单步电流饱和。

**基于数据的推断：**同一训练 checkpoint 对正负 pitch 产生了明显不同的左右轮目标，并且近乎匹配 pitch 幅值/rate 的负 pitch 回合失败、正 pitch 回合通过，因此“测试初态恰好抽得不同”不足以解释全部差别；Actor 的 pitch 符号相关响应与左轮启动瞬态相互作用，是主要候选机制。该结论目前是配对观察，不是精确镜像状态的受控因果实验。

**限制：**Pitch rate 也与结果有关。pitch `<0` 样本中，失败回合初始 rate 均值 `-0.0142 rad/s`、通过回合 `+0.0686 rad/s`；pitch `>=0` 中失败均值 `-0.0679`、通过 `+0.0130 rad/s`。因此不能把所有 seed 1 失败归结为 Pitch 符号单一因素。near-zero 的配对回合 24189/24144 首次原始 Actor 输出很接近，但前者左轮 5 连续饱和、后者只有 2 步，表明还需要观察状态沿途差异。

### 固定 checkpoint 的镜像初态配对结果

为区分初态抽样差异与策略/动力学方向响应，使用 seed 1 checkpoint，对 Pitch `±2.5°` 和 Pitch rate `±0.08 rad/s` 做了 2×2 配对重放。没有修改 Actor、动作或执行器；同一配对中的其他 reset 状态相同。四种初态分别重放 8 次，输出逐步一致（任务无域随机化或观测噪声，因此这些是确定性重复，不是 8 个独立试验）。

| 初始 Pitch | 初始 rate | 通过数 | 首步六维动作（wheel L/R，leg L1/L2/R1/R2） | 左轮接触/饱和结果 | 最大漂移 |
|---:|---:|---:|---|---|---:|
| `-2.5°` | `-0.08` | 0/8 | `[-.105,.641,-.056,.240,-.205,-.219]` | 第 1 步失去接触；第 3–9 步最长饱和 7 步 | 0.1011 m（该 seed） |
| `-2.5°` | `+0.08` | 0/8 | `[-.163,.570,-.070,.150,-.261,-.255]` | 第 1 步失去接触；第 3–6 步连续饱和 4 步 | 0.0753 m（该 seed） |
| `+2.5°` | `-0.08` | 8/8 | `[-.219,.507,-.069,.105,-.275,-.262]` | 有短暂单步接触丢失；最大饱和 run 为 2 步 | 0.0628 m |
| `+2.5°` | `+0.08` | 8/8 | `[-.278,.433,-.084,.014,-.333,-.298]` | 左轮第 1 步接触；轮饱和最长 1 步 | 0.0649 m |

以相同 `+0.08 rad/s` rate 为例，负 Pitch 的首步轮目标为 `[-1.30,+4.56] rad/s`，正 Pitch 为 `[-2.23,+3.47] rad/s`；负 Pitch 首个策略步后左轮脱离接触，而正 Pitch 保持接触。负 Pitch 两种 rate 的 Actor 首步输出都更偏向右轮高目标，后续出现左轮反向限流。**这排除了“仅仅是独立测试碰巧抽到不同 Pitch/rate”的解释对该固定初态成立；它确认了策略输入符号触发了不同动作和结果。**它仍不能单独区分网络学到的不对称、对称坐标约定中的正确/错误响应，以及相应接触动力学的因果贡献；不能据此宣称坐标系实现错误。

### seed 2：第 2–5 步左轮连续电流饱和

全部 8 个失败回合终止于 1000 步 time limit，失败列表只有 `wheel_saturation_duration`。它们的初始状态都在 `Pitch=+1.390°…+2.943°`、`Pitch rate=+0.0575…+0.0980 rad/s`。每个回合的饱和窗口完全相同：左轮步骤 2–5，右轮在步骤 2 和 5 各单独饱和一次；每回合 max consecutive `[4,1]`。左/右轮 current saturation fraction 为 `[0.004,0.002]`，均未超过 1%；wheel torque saturation 为 `[0,0]`；leg torque 无饱和；漂移最大值 0.0204–0.0424 m，接触和姿态验收均满足。

重放中用相似初态的通过回合配对后，发现 seed 2 的 100 个测试样本里没有 `Pitch>1° 且初始 rate>0.05 rad/s` 的通过对照；因此在这一联合初态子区间无法用同一 seed 的测试样本区分“状态本身敏感”与“策略响应”。最近的回合配对和逐步数据见 `EPISODE_COMPARISON.md`。

## 饱和计数与坐标/执行器审计

- `PurePPOStandEnv.step()` 在每个 8 ms 策略步开始时，读取轮速并计算电流/力矩，再执行 4 个 2 ms MuJoCo motor ticks。回合统计把本步 `abs(current) >= 0.99 * 10 A` 或 `abs(torque) >= 0.99 * 2.46 Nm` 作为 wheel saturation 标志。
- `_longest_true_run()` 对每个 wheel 按布尔序列逐样本计数；非饱和立即归零。因此 step 2、3、4、5 是四个饱和策略步，不是 3 个区间，也不是 16 个 motor tick。
- `episode_passes_acceptance()` 用 `max_consecutive_steps > 3` 判定失败，严格匹配每回合 max `[4,1]`。对 seed 1/2 的 200 条重放轨迹重新计算两轮饱和最长连续数和比例，与 episode 汇总全部匹配（比例绝对误差为 0）；原始 JSONL 的每个失败原因也逐条复现。没有 off-by-one 或旗标顺序错误的证据。
- action order 是 `[wheel_left, wheel_right, leg_left_1, leg_left_2, leg_right_1, leg_right_2]`。Actor 归一化输出先直接限于 `[-1,1]`，轮目标为 `8 × action[:2] rad/s`，腿目标为 `clip(q_ref + 0.35 × action[2:6], ±1.2 rad)`。重放同时记录了 Gaussian policy raw mean、送入环境的六维动作和 actuator 目标；未有额外规则对六维输出做姿态修正。
- Pitch 由四元数 `xyz` 欧拉角的 Y 分量读取；Pitch rate 由机体坐标的角速度 Y 分量读取。MuJoCo free-joint `qvel[3:6]` 的局部角速度先旋转到 world，再用旋转矩阵转回 body。reset 将 pitch rate 写在 body-Y 轴 `qvel[4]`。权重、映射、reset 定义和 preflight ±Pitch/±rate 符号探针一致；目前没有坐标轴交换或符号翻转的直接证据。

## 已排除或暂不支持的解释

- **验收统计 off-by-one：**排除；逐步布尔窗口、回合汇总和验收原因吻合。
- **腿部扭矩饱和是 drift-weight seed 1/2 失败原因：**不支持；失败字段没有 leg saturation，wheel torque saturation 为 0。失败在 wheel current 的持续步数，seed 1 的另一个附加失败为 drift。
- **Pitch RMS / 终止 / chassis 接触导致失败：**排除于本组具体失败；200 回合均完整跑到 1000 步，终止原因均 `time_limit`，无 body contact 失败。
- **wheel saturation fraction 超标：**排除；seed 1 最坏为 0.7%，seed 2 最坏为 0.4%，两者都低于 1%。
- **额外平衡控制器、动作混合或在线 reward 控制：**Pure PPO standalone 任务代码中未发现；reward 在评估中只记分，不反向修改当前动作。
- **只由位置漂移奖励决定所有种子成败：**排除；`position_drift=0.50` 三个新随机种子结果从 49 到 100/100，seed 2 的失败与 drift 无关。

## 未确认假设与缺失项

1. 历史 `runs/.../test_all` 原本没有 seed 1/2 的 `step_trace.csv.gz`。本次在对应 checkpoint、固定原 ID seed 上重放 200 回合补齐，并逐项匹配历史 episode 汇总；轨迹是新生成数据，不是原训练时保存的二进制日志。
2. 原训练 source fingerprint `77531…` 没有随 checkpoint 保存完整源码快照。训练配置所列 Git commit 的干净树 hash 不等于它；当前运行源为记录中的 post-run fingerprint `36864…`，训练报告说明是纯空白格式改动。模型内容与 actuator contract 已匹配，200 个 episode 汇总也匹配，但无法独立恢复历史源码每个字节。
3. drift-weight seed 2 的全部 8 个失败都处在 `Pitch>1° / rate>0.05 rad/s` 组合，100 个 ID seed 中没有同一组合内的通过回合。这限制了初态匹配的因果判断。
4. seed 1 的镜像初态试验确认负/正 Pitch 对该 checkpoint 的结果不同；但目前不能精确区分 Actor 方向响应、轮腿启动瞬态和接触状态耦合各自的因果贡献。没有发现坐标系符号实现错误的直接证据。
5. `position_drift=0.10`、scale 0.35 的正式随机训练只有 seed 0。虽可在相同 10 个 reset seed 上与 drift 0.50 checkpoint 做行为对照，但不能据此判断 0.50 对所有训练种子都有效。

完整数据审计、回合配对和轨迹路径见 [EPISODE_COMPARISON.md](EPISODE_COMPARISON.md)。
