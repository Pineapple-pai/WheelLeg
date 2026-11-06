# v14：高速切换覆盖、腿长一致性与全误差速度跟踪

当前默认阶段为 [v16](joint_tracking_v16.md)。v14完整4000次更新已完成，
切换时大幅前后摆腿的复测及后续修改见v15说明；本文保留v14训练设置。

## v13 最终模型复测

2026-11-03，使用 `joint_tracking_pos_v13/checkpoint`（4000 次更新），
名义动力学、无外推、保留传感器噪声、近静止初态。
`logs/joint_tracking_pos_v13_user_diagnosis.json` 含每协议两个种子 20000/20001；
`logs/joint_tracking_pos_v13_high_speed_switch_diagnosis.json` 含种子 20000 的扩展协议与奖励分项。
这些是有限复现样本，不能视为总体成功率。

| 协议 | 触发时刻 | 结果 |
|---|---|---|
| +2m/s → 停车 | 10s | 10.376s 轮子站位越界，停车未完成 |
| −2m/s → 停车 | 10s | 10.728s 倾角越界，pitch 峰值 37.94° |
| +2 → −2m/s | 11s | 11.336s 轮子站位越界，腿长差 RMS 22.7mm |
| 保持 +2m/s，0.27 → 0.22m | 8s | 10.136s 倾角越界 |
| 保持 −2m/s，0.27 → 0.22 → 0.32m，再停车 | 8/13/18s | 调高度存活，18.296s 停车时轮子站位越界 |
| +2m/s 切原地转向、再后退、再转向 | 8/13/18s | 此种子跑满24s，但首次切换 pitch 达21.4° |

0.27m 高度平移末秒速度约 +1.888 / −1.934m/s，启动 T90 约1.12 / 0.928s。
3–8s 平移段腿长差 RMS 为5.24 / 8.33mm；统计包含加速段。
在 `motion` 协议中 +1 → +2m/s 后的4s内未满足稳定保持0.25s的T90判据。
前进平移轮饱和比例约3.3%，不能将整体跟踪迟缓简单归因于持续满力矩。
后退停车时轮饱和比例约63.5%，制动阶段仍可能受力矩能力限制。

按键切换只更新命令；环境触发终止后，回放才自动 reset。
联合跟踪终止条件包含倾角、车身触地、轮子站位、低高度和横向速度。
这里的复现没有“左右腿不等长就直接 reset”的终止规则。
旧的5/6键还会切入零速度调高度模式，因此高速按键同时包含急停。
现在5/6键保留当前平移/转向命令；7仍显式停车。已运行的回放进程需重启后生效。

## 固定修改与取舍

- **切换密度和种类。** v13 的普通运动命令有50%机会保持12–24s，
  24s回合可能只有少数切换。v14长命令概率改为20%、时长6–10s，
  普通运动保持3–5s，停车保持4–8s。此前纯平移命令满足幅度>0.1m/s时，
  重采样有20%显式停车、40%直接反向、15%保持速度切换到另一侧高度端点、
  10%切原地转向，其余15%沿用原随机混合。直接反向中50%保持原幅值，
  另50%在0.8–1.2m/s均匀采样新幅值，正反方向规则相同且保留原高度。
  这强化±1m/s附近的换向，同时保留高速同幅值反向。所有比例从第一批固定启用。
  这些概率以重采样时的命令为条件，并不保证当时实际速度已经达到目标；
  全速度带、后退占比和35%运动初态继续保留。长稳态覆盖减少是明确取舍，需另做24s固定命令验收。
- **实际腿长。** hip站点到wheel站点的真实距离差死区2→1mm，尺度4→3mm，
  权重0.8→1.2；5mm差值成本从0.45变为2.0，roll连续误差权重0.4→0.6。
  转弯也使用软约束，不强制关节角相同或锁定双腿；需检查是否影响转向和抗扰。
- **全误差速度跟踪。** Huber连续误差尺度0.30→0.15m/s，权重保持1.4。
  ±0.05m/s附近成本曲率为v13的4倍；大误差线性斜率9.33→18.67。
  每个控制周期持续累计误差成本，鼓励更早制动、过零、反向加速并消除持续欠速。
  初稿v14只将小误差曲率加倍、大误差斜率保持原值，未加强换向全过程；
  本次在完整v14训练开始前修订，旧的`joint_tracking_pos_v14_smoke`不用于续训。
  大误差成本增加可能影响存活与姿态，因此必须联合验收；不把增大权重当作效果证明。
  原速度正奖励及姿态、漂移、平滑和饱和成本继续保留。

这些修改针对观察到的问题，但分布覆盖和奖励的相对作用尚未经消融确认。
倾角和轮子站位终止边界、执行器参数、动作/观测接口及物理过程保持原值。
仍为纯PPO的两轮独立速度与四关节360°周期位置，无命令爬坡、动作滤波或平衡辅助。
v13阶段配置和奖励定义保留，旧模型回放不使用v14奖励。

## ±1m/s直接换向基线与验收口径

`logs/joint_tracking_pos_v13_reverse10_eval.json` 包含两个种子、正向起步和后退起步的24s协议，
名义动力学、0.27m高度，均跑满回合。种子20000的主要结果：

| 切换 | 过零时间 | T90 | 进入并保持±0.05m/s至少0.25s | 末秒速度 |
|---|---|---|---|---|
| +1 → −1m/s（8s） | 0.464s | 0.888s | 1.040s | −0.980m/s |
| −1 → +1m/s（14s） | 0.376s | 0.920s | 6s片段内未达到 | +0.929m/s |

+1→−1切换0.5s后的实际速度只有−0.082m/s，1s后为−0.921m/s。
反方向在2s后为+0.932m/s，反映出稳态欠速，而不是一直未完成过零。
T90误差带是实际初速度到目标速度阶跃的10%，±1换向约为±0.2m/s，
因此不能用T90代替“已经准确达到目标速度”。脚本现在分别输出过零、
反向加速至T90、绝对±0.05m/s稳定时间、速度绝对误差积分及0.25/0.5/1/2s速度。
轮饱和比例是整个6s片段的均值（约0.6%/1.75%），不能据此排除瞬时饱和。

## 续训

```bash
bash scripts/train_joint_quiet_v23.sh \
  --stage joint_tracking_v14 --version joint_tracking_pos_v14 \
  --checkpoint checkpoints/joint_tracking_pos_v13/checkpoint \
  --reset-vecnormalize --preserve-action-std \
  --learning-rate 0.00003 --updates 4000 \
  2>&1 | tee logs/joint_tracking_pos_v14.log
```

短训练 `joint_tracking_pos_v14_speed_smoke` 用于检查修订后的续训流程，
不能据此认定换向时间、腿长或速度跟踪已改善。
修订后82项测试、10次更新（40960步）续训检查、shell语法和`git diff --check`通过。
2026-11-03已从v13最终模型启动4000次更新正式续训，输出为`joint_tracking_pos_v14`，
训练日志为`logs/joint_tracking_pos_v14.log`。本次启动的任务在训练成功退出后，
顺序执行下方两组验收，分别输出`*_switch_eval.json`及`*_speed2_eval.json`，
对应运行日志为同名`.log`。训练及验收尚未完成，不能宣称速度跟踪已改善。

## 训练后验收

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v14/checkpoint \
  --profiles reverse10 reverse10_backfirst stop2_forward stop2_reverse motion height2_forward height2_reverse turn2 \
  --episodes 5 --output logs/joint_tracking_pos_v14_switch_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v14/checkpoint --episodes-per-case 5 \
  --duration 24 --heights .22 .27 .32 --speed 2 --cases forward reverse \
  --output logs/joint_tracking_pos_v14_speed2_eval.json
```

切换报告补充腿长差峰值、轮子错位/偏移峰值、轮/关节饱和比例与奖励分项均值。
比较生存、过零时间、T90、±0.05m/s稳定时间、实际速度偏差、腿长差、pitch/roll、高度和停车位移；
未存活或未达到稳态的片段不能用均值替代成功验收。
