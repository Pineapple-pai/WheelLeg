# Pure PPO 站立失败回合与逐步轨迹对照

本报告区分历史正式评估记录与本次固定 checkpoint 重放。历史 `test_all` 未保存 drift-weight 实验 seed 1/2 的逐步 CSV；本次重放补齐了数据，并将每个 episode 汇总与历史 JSONL 对照，200/200 均无字段或失败原因差异。以下链接中的 `artifacts/` 是新生成的诊断轨迹，不是历史原始 trace，也不是训练输出。

## 实验及测试分组复核

| 实验 / 训练 seed | 独立测试 episode seeds | 配置 | 验收 | 失败拆分 |
|---|---|---|---:|---|
| `pure_ppo_stand_v1` / 0 | 20000–20099 | fresh PPO，腿 scale 0.50，drift 0.10 | 0/100 | actuator 饱和问题；详见基线报告与旧 trace |
| `pure_ppo_stand_v1` / 1 | 21000–21099 | fresh PPO，0.50 / 0.10 | 100/100 | 无 |
| `pure_ppo_stand_v1` / 2 | 22000–22099 | fresh PPO，0.50 / 0.10 | 0/100 | actuator 饱和/漂移；详见历史日志 |
| 原 seed 0 checkpoint 推理对照 | 20000–20099 | 同一 checkpoint，scale 0.35 / drift 0.10 | 100/100 | 推理映射对照；非新训练 |
| `pure_ppo_stand_leg_scale_035_v1` / 0 | 23000–23099 | fresh PPO，0.35 / 0.10 | 0/100 | 漂移、轮部饱和/接触和腿部饱和；历史 trace 在 runs 下 |
| `pure_ppo_stand_leg_scale_035_drift_weight_5x_v1` / 0 | 24000–24099 | fresh PPO，0.35 / 0.50 | 100/100 | 无 |
| 同上 / 1 | 24100–24199 | fresh PPO，0.35 / 0.50 | 49/100 | 51 个 `wheel_saturation_duration`；其中 8 个兼有 `planar_drift` |
| 同上 / 2 | 24200–24299 | fresh PPO，0.35 / 0.50 | 92/100 | 8 个 `wheel_saturation_duration`，其余 92 个通过 |

本次使用的 drift-weight checkpoint SHA256：seed 0 `ec4c3c422103aab57cd01d10bc865e0693329ed7b93cd80c9b00b1c5cf1de89a`；seed 1 `fd2ca93a7d507dab9269144ddbfdc20c3d161ff6465367a3cfd4b65d74f255c4`；seed 2 `d378a718d95d0b842f3912275182c22f573508bafbe93de9a58bb58f058fad49`。三者均使用模型 XML SHA256 `22fc08a5f900ffc20d977c29851618fb0ca03c1d6b01dd2da0f32209ba505c54`。完整配置、合约和 source fingerprint 核对见 [ROOT_CAUSE_REPORT.md](ROOT_CAUSE_REPORT.md)。

## seed 1：失败与通过分组

100 个 ID episode 都跑满 1000 个策略步，终止原因为 `time_limit`。按初始 Pitch 分组：

| 初始 Pitch | 样本数 | 通过 / 失败 | 失败原因 |
|---|---:|---:|---|
| `[-3°, -1°)` | 31 | 0 / 31 | 31 个 wheel 连续饱和超限；其中 8 个同时超过漂移阈值 |
| `[-1°, +1°)` | 33 | 16 / 17 | 17 个 wheel 连续饱和超限 |
| `[+1°, +3°]` | 36 | 33 / 3 | 3 个 wheel 连续饱和超限 |

另外，signed 分组为 `<0°` 47 条（6 通过）与 `≥0°` 53 条（43 通过）；near-zero 区间跨越正负两侧。

负 Pitch 失败的 41 条均因连续饱和超限；其中 8 条还超过 drift acceptance。所有 51 个失败（41 个负 Pitch，10 个非负 Pitch）都在第 1 个策略步后的状态记录中显示左轮 contact 为 0，第一次轮饱和出现在第 3 步。这个先后顺序是：**接触丢失（post-step 1）→ 轮电流饱和（step 3）→ 漂移增长（仅 8 条在 step 26–28 超 1 cm、step 53–67 超 5 cm）**。这 51 条的轮饱和比例均低于 1% 验收线；失败由持续时间引起。

### 可比 episode 的完整首步动作与后续信号

| 回合 | 初始 Pitch / rate | 首步 Actor 六维输出 | 首步轮目标 | 首步后接触（L/R） | 轮饱和窗口 | 最大漂移 | 结果 |
|---|---|---|---|---|---|---:|---|
| 24132，fail | `−2.912° / +0.0903` | `[-.157,.576,-.070,.155,-.259,-.254]` | `[-1.26,+4.61] rad/s` | `0 / 1` | L: 3–7（5 步），R: 6–7、9（单步/2步） | 0.0940 m | duration fail |
| 24185，pass | `+2.825° / +0.0919` | `[-.290,.419,-.086,-.002,-.341,-.303]` | `[-2.32,+3.35] rad/s` | `1 / 1` | R: step 2 单步 | 0.0656 m | pass |

两回合初始绝对 Pitch 相差 0.088°、rate 相差 0.0016 rad/s。24132 的 step 1 轮速从 `[0,0]` 变为 `[-4.79,+12.41] rad/s`，电流 `[-1.6,+5.8] A`，此时尚未限流但左轮已失去接触；24185 首步后轮速 `[-1.59,+14.33] rad/s`、电流 `[-2.9,+4.2] A` 且两轮接触。24132 从 step 3 起左轮电流反复在 ±10 A 间切换，step 7 连续计数达到 5，step 8 清零。失败回合的漂移阈值明显晚于接触丢失和饱和。

near-zero 对照 24189 (`−0.336°`, `+0.0120 rad/s`, fail) 与 24144 (`−0.103°`, `+0.0304 rad/s`, pass) 首步输出接近；最大漂移为 0.0636/0.0630 m，但左轮饱和分别连续 5/2 步。正 Pitch 对照 24152 (`+1.180°`, `−0.0689`, fail) 与 24163 (`+1.469°`, `−0.0653`, pass) 的漂移都为 0.0618 m，左轮 run 则是 5/3 步。这说明“初始 Pitch 符号”是强分层因素，但连续饱和边界附近也受完整状态轨迹影响。

### seed 1 checkpoint 的镜像初态配对推理

构造 2×2 初态：Pitch `±2.5°`、Pitch rate `±0.08 rad/s`。对每个条件重复 8 次并保持相同的其它 reset 状态；由于环境无域随机化/噪声，重复结果逐项一致，不能作为独立样本估计置信区间。

| Pitch | Rate | 通过 | 首步六维动作 | 首步后 contact L/R | max wheel run | max drift |
|---:|---:|---:|---|---|---:|---:|
| −2.5° | −0.08 | 0/8 | `[-.105,.641,-.056,.240,-.205,-.219]` | 0/1 | 7 | seed 30000 为 0.1011 m（重放组最大 0.1289 m） |
| −2.5° | +0.08 | 0/8 | `[-.163,.570,-.070,.150,-.261,-.255]` | 0/1 | 4 | seed 30000 为 0.0753 m（重放组最大 0.0794 m） |
| +2.5° | −0.08 | 8/8 | `[-.219,.507,-.069,.105,-.275,-.262]` | 0/1 | 2 | 0.0628 m |
| +2.5° | +0.08 | 8/8 | `[-.278,.433,-.084,.014,-.333,-.298]` | 1/1 | 1 | 0.0649 m |

rate 固定为 `+0.08` 时，负/正 Pitch 的轮目标分别是 `[-1.30,+4.56]` 和 `[-2.23,+3.47] rad/s`。负 Pitch 的 step 1 左轮失去接触，step 3 开始限流并达到 4 步；正 Pitch 的 step 1 保持双轮接触且最多单步饱和。rate 为负时，正 Pitch 同样通过，而负 Pitch 失败并有 7 步连续饱和。**在这组受控初态里，符号不是随机初态抽样造成的混杂因素；符号改变了 Actor 输出，并沿着左轮接触/饱和路径改变验收结果。**这不证明网络在坐标语义上“输出错误”，也不证明是纯策略而非策略和物理耦合。

## seed 2：8 个连续 4 步饱和失败回合

| Episode | 初始 Pitch | 初始 rate | max drift | contact L/R | current sat frac L/R | max run L/R | 失败 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 24223 | +2.888° | +0.0575 | 0.0400 m | .998 / 1.000 | .004 / .002 | 4 / 1 | duration |
| 24230 | +2.754° | +0.0597 | 0.0382 m | .998 / 1.000 | .002 / .002 | 4 / 1 | duration |
| 24231 | +1.700° | +0.0741 | 0.0247 m | .998 / 1.000 | .004 / .002 | 4 / 1 | duration |
| 24239 | +1.390° | +0.0717 | 0.0204 m | .998 / 1.000 | .004 / .002 | 4 / 1 | duration |
| 24241 | +2.009° | +0.0780 | 0.0290 m | .998 / 1.000 | .004 / .002 | 4 / 1 | duration |
| 24249 | +2.134° | +0.0757 | 0.0305 m | .998 / 1.000 | .004 / .002 | 4 / 1 | duration |
| 24271 | +2.943° | +0.0791 | 0.0424 m | .998 / 1.000 | .004 / .002 | 4 / 1 | duration |
| 24299 | +2.598° | +0.0980 | 0.0389 m | .998 / 1.000 | .004 / .002 | 4 / 1 | duration |

全部 8 条只有 `wheel_saturation_duration` 失败；漂移都低于 0.05 m、接触合格、轮饱和占比远低于 1%、没有腿部力矩饱和或 wheel torque saturation。每条均是左轮 step 2–5 连续电流饱和；右轮在 step 2 和 step 5 各单独饱和。测试分布中没有 `Pitch > 1° 且 rate > 0.05 rad/s` 的通过回合，因此正式测试里没有同一联合初态区域的 natural pair。

### Episode 24223 与可比通过回合 24285：step 1–6

24285 的 Pitch `+2.979°` 与 24223 的 `+2.888°` 接近，但 rate 较低（`+0.0330` 对 `+0.0575 rad/s`）。表中动作顺序为 `[wheel L, wheel R, leg L1, leg L2, leg R1, leg R2]`；轮速是控制器计算时读取的 step 前速度，接触是该策略步积分后的状态。

| Step | Episode / outcome | Actor 六维动作 | 轮目标 L/R | step 前轮速 L/R | 电流 A L/R | 连续 run L/R | contact L/R | drift | Pitch / rate | reward / drift 项 |
|---:|---|---|---|---|---|---|---|---:|---|---|
| 1 | 24223 fail | `[.183,.245,-.349,-.224,-.320,-.318]` | `1.46/1.96` | `0/0` | `1.8/2.4` | `0/0` | `1/1` | .0002 | `2.86°/−.145` | .760 / ~0 |
| 1 | 24285 pass | `[.197,.255,-.348,-.217,-.313,-.313]` | `1.58/2.04` | `0/0` | `2.0/2.5` | `0/0` | `1/1` | .0002 | `2.94°/−.164` | .720 / ~0 |
| 2 | 24223 fail | `[.236,.300,-.374,-.360,-.442,-.436]` | `1.89/2.40` | `10.81/13.04` | `−10/−10` | `1/1` | `0/1` | .0007 | `2.77°/−.212` | .643 / −.00002 |
| 2 | 24285 pass | `[.266,.322,-.375,-.354,-.435,-.434]` | `2.13/2.57` | `10.62/12.81` | `−10/−10` | `1/1` | `0/1` | .0007 | `2.84°/−.232` | .590 / −.00002 |
| 3 | 24223 fail | `[.312,.347,-.302,-.128,-.209,-.376]` | `2.49/2.78` | `−9.75/4.03` | `10/−1.6` | `2/0` | `0/1` | .0011 | `2.69°/−.146` | .779 / −.00006 |
| 3 | 24285 pass | `[.356,.378,-.305,-.125,-.203,-.377]` | `2.85/3.03` | `−9.94/3.94` | `10/−1.1` | `2/0` | `0/1` | .0011 | `2.75°/−.166` | .739 / −.00006 |
| 4 | 24223 fail | `[.165,.243,-.382,-.558,-.613,-.457]` | `1.32/1.94` | `19.26/.47` | `−10/1.8` | `3/0` | `1/1` | .0017 | `2.57°/−.343` | .274 / −.00014 |
| 4 | 24285 pass | `[.216,.279,-.388,-.560,-.611,-.467]` | `1.73/2.23` | `19.15/1.10` | `−10/1.4` | `3/0` | `1/1` | .0016 | `2.63°/−.359` | .212 / −.00014 |
| 5 | 24223 fail | `[−.013,.142,-.330,-.355,-.472,-.428]` | `−.11/1.14` | `9.67/11.05` | `−10/−10` | `4/1` | `1/1` | .0024 | `2.42°/−.329` | .341 / −.00029 |
| 5 | 24285 pass | `[.073,.202,-.330,-.321,-.433,-.430]` | `.58/1.62` | `6.83/11.19` | `−7.8/−10` | `0/1` | `1/1` | .0023 | `2.47°/−.315` | .387 / −.00027 |
| 6 | 24223 fail | `[−.145,.034,-.277,-.230,-.347,-.339]` | `−1.16/.27` | `−2.86/−1.39` | `2.1/2.1` | `0/0` | `1/1` | .0031 | `2.27°/−.322` | .381 / −.00049 |
| 6 | 24285 pass | `[−.041,.105,-.287,-.237,-.341,-.357]` | `−.33/.84` | `−3.98/−1.90` | `4.6/3.4` | `0/0` | `1/1` | .0030 | `2.33°/−.307` | .429 / −.00044 |

最早明显的分岔发生在 step 5：两个回合都给出左轮限流需求，但 24223 的左轮电流达到 `−10 A`，run 从 3 变 4；24285 是 `−7.8 A`，run 清零。两回合在 step 2–4 都有相似的饱和和高实际轮速，不能把“前三步已饱和”本身当成失败充分条件。24223 step 5 的 wheel target 已改为 `−0.11 rad/s`，说明 Actor 有调整目标的机会，但当前轮速为 `+9.67 rad/s`，仍产生 `−10 A` 限流；step 6 目标继续调整后饱和解除。观察证明了动作可变、饱和可退出，不足以证明第四步由前三步动作单独因果造成。

step 1 的四个腿目标（24223）为 `[-.317,.113,-.303,.079] rad`；积分后实际位置 `[-.212,.177,-.211,.174] rad`，速度 `[-3.54,-2.77,-3.84,-3.24] rad/s`，力矩 `[-12.2,-7.9,-11.2,-11.1] Nm`，未发生腿力矩饱和。各步的腿关节状态、轮 torque、reward 全分项均完整保存在 trace 中。

## `position_drift=0.10` 与 `0.50` 的训练结果差异

两组在 scale `0.35 rad` 下使用 fresh PPO；唯一配置因素是 position-drift reward coefficient（`.10 → .50`）。这一对照改变 reward 后得到的策略行为，不可与固定权重的单步 counterfactual 混为一谈。

| 训练设置 / seed | ID 通过 | Pitch RMS mean | drift P95 | contact L/R mean | wheel current saturation | leg torque saturation | 主要观测 |
|---|---:|---:|---:|---:|---:|---:|---|
| drift `.10` / seed 0 | 0/100 | 0.372° | 0.1660 m | .511 / .568 | .4392 / .4117 | joint 3 mean .2692 | 100 回合存活但明显移动、轮速跟踪频繁限流、腿部 joint 3 也持续饱和 |
| drift `.50` / seed 0 | 100/100 | 0.577° | 0.0611 m | 约 1.0 / 1.0 | .0002 / 0 | 无显著腿饱和 | contact、drift 和 actuator gates 均通过 |
| drift `.50` / seed 1 | 49/100 | 2.074° | 0.1051 m | 约 1.0 / 1.0 | 失败 episodes 仍低于 1% fraction | 无显著腿饱和 | 51 duration failures，8 条兼有 drift |
| drift `.50` / seed 2 | 92/100 | 0.363° | 0.0466 m | .998 / 1.000（失败回合） | 失败 episodes .004 / .002 | 无腿饱和 | 8 条只因 4 步 duration 失败 |

同一 reset IDs 23000–23009 上，对 drift `.10` seed 0 checkpoint 与 drift `.50` seed 0 checkpoint 做了额外确定性推理比较（不是正式接受率）：`.10` 的 mean drift `0.12019 m`、P95 `0.15764 m`、contact `.5121/.5686`、current saturation `.4383/.4126`，十条均失败；`.50` 的 mean drift `0.03742 m`、P95 `0.06088 m`、contact `1.0/1.0`、current saturation `.0003/0`，十条均通过该配对诊断。动作均值从 `[.1407,.1841,-.4568,.0651,-.5190,-.1840]` 变为 `[-.1297,.1436,-.2081,-.5670,.0012,-.3868]`，且前者可见较长的轮部饱和 run，后者只在少数 step 单次饱和。这表明差异涉及动作幅值/方向、持续时间、接触和漂移，而不只是总回报变化。

**直接事实：**`.50` seed 0 的 posture RMS 稍高于 `.10` seed 0，但 drift、轮接触和执行器饱和明显改善；`.50` seeds 1/2 的饱和比例也很低，但尾部的持续时间门槛仍使部分回合失败。**推断：**较强 drift 惩罚在 seed 0 学到更少平移、更多稳定接触和更低 actuator use 的行为；它对 random initialization 的结果仍有显著方差（49–100/100）。只有一个 `.10` fresh training seed，不能据此量化系数对跨种子的平均因果效应；不能由某个 seed 成功推出普遍收敛。

## 步级轨迹、来源及字段

| 数据集 | 内容 | 轨迹文件 | 汇总 / manifest |
|---|---|---|---|
| seed 1 独立 ID 重放 | 24100–24199 全部 100,000 策略步 | [`step_trace.csv.gz`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed1_id/step_trace.csv.gz) | [`episode_metrics.json`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed1_id/episode_metrics.json), [`run_manifest.json`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed1_id/run_manifest.json) |
| seed 2 独立 ID 重放 | 24200–24299 全部 100,000 策略步 | [`step_trace.csv.gz`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed2_id/step_trace.csv.gz) | [`episode_metrics.json`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed2_id/episode_metrics.json), [`run_manifest.json`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed2_id/run_manifest.json) |
| seed 1 镜像初态 | 32 个确定性 episode，32,000 步 | [`step_trace.csv.gz`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed1_mirror_pairs_30000/step_trace.csv.gz) | [`episode_metrics.json`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed1_mirror_pairs_30000/episode_metrics.json), [`run_manifest.json`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed1_mirror_pairs_30000/run_manifest.json) |
| `.10` vs `.50` paired inference | 23000–23009 的 `.50` checkpoint replay；`.10` 参照历史 trace | [`step_trace.csv.gz`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed0_paired_23000/step_trace.csv.gz) | [`episode_metrics.json`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed0_paired_23000/episode_metrics.json), [`run_manifest.json`](../../artifacts/pure_ppo_stand_v1_diagnosis/drift5x_seed0_paired_23000/run_manifest.json) |

每行是一 8 ms 策略步；字段包含 Gaussian Actor raw mean、策略裁剪动作、实际交给环境的六维动作；目标/实际 wheel speed、电流、力矩、误差、限幅和连续计数；四个腿关节目标/位置/速度/力矩；接触、drift 及 reward 总值/分项。对 seed 2，每个回合重算的饱和最长连续段和饱和比例与历史 summary 完全一致。
