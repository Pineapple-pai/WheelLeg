# 2026-11-04：速度优先级校准与训练入口

本次沿用五组简化奖励，不继续叠加腿长、关节目标相等或连杆范围成本。
评估优先看存活、实际速度和几何，训练reward不作为跨定义选优依据。

## 已有结果与本次单因素试验

工作区已有两轮筛选，详见`optimization_20261104.md`。
简化支撑倾角＋同轴组，以及提高失败成本的候选，仍有欠速。
旧`joint_tracking_pos_v17_candidate`只有未完成运行的配置/日志，没有可恢复的模型文件；
本次不覆盖该目录，也不将它算作完成训练。

本次从同一v16最终checkpoint出发，两个组均为训练seed0、
4环境×512步×100更新，共204800步；学习率固定3e−5，均重置归一化统计，
保留相同探索标准差及所有命令采样和物理设置。
计划、源码和源checkpoint哈希存于`logs/reward_priority_20261104_pilot/manifest.json`。

| 因素 | tight_survival对照 | responsive候选 |
|---|---|---|
| stage | joint_tracking_simple_tight_survival | joint_tracking_simple_responsive |
| 速度成本 | `(1/6)ρ(ev/0.05)` | `(1/3)ρ(ev/0.05)` |
| 失败成本 | −20/次 | −20/次 |
| 其余项及采样 | 完全相同 | 完全相同 |

`ρ`近零为平方、大误差为线性；仅速度成本加倍，未改目标速度或限制其变化。
五组为任务跟踪、车身姿态、动作变化、失败事件及真实轮轴同轴成本。
车身姿态使用与整机支撑倾角的误差，并保留roll和适用时的航向要求。
主动转向不锁定世界yaw；所有六个动作继续独立。

此次候选在0.1m/s误差时速度成本为1/s，对照为0.5/s，
但不能仅用这一个成本与原v16包含多项速度奖励的总效果比较。
高速大误差成本斜率也加倍，可能损害恢复；必须在实际换向和停车协议验证。

## 检查与边界

已有真实CLI续训回归确保请求的seed、rollout、batch、学习率schedule、gamma和GAE
在重建模型前传入，不被源checkpoint的设置覆盖。控制参数和实测预算需要一致。
本轮增加单因素检查：除schema与vx_weight外，responsive与tight_survival参数完全一致，
且环境实际只产生上述五组奖励。

筛选使用seed31000的六个切换协议及固定±2m/s，属于探索性短训。
随后在未用于此次筛选的34000–34002种子上与v16重测。
固定3个评估种子不能代表训练方差，单个训练seed的短训也不能证明长期最优。
正式长训尚未启动。

## 追加的失败预算对照

首轮100更新中，responsive的±1m/s末秒速度约+0.938/−0.952m/s，
优于tight_survival对照的+0.903/−0.879m/s，但高车身后退切站立重新失败。
因此追加`joint_tracking_simple_responsive_survival`，速度权重、全部其他系数均不变，
只把失败事件成本从20提高到30；仍为五组奖励。
它同样从v16、seed0、4环境×512步×100更新开始，
记录于`logs/reward_recovery_20261104_pilot/`，不覆盖先前试验。
新增一次失败成本试验不代表已经解决恢复问题，必须看实际协议结果。

## 实测结果与未通过的候选

提高失败成本至30的候选在初筛中仍未通过motion和高位后退停车，
前进停车也在11.576s失败，因此不将它作为推荐升级。

速度优先候选从100更新继续200更新（seed1、4环境、不重置同阶段VecNormalize），
模型保存在`checkpoints/joint_tracking_pos_v17_refine/`。它只是候选，不替代v16。
34000–34002种子、名义动力学、0.27m、三回合复测：

| 指标 | v16 | 速度优先候选 |
|---|---:|---:|
| +1m/s末秒实际速度 | 约+0.884 | +0.968m/s |
| −1m/s末秒实际速度 | 约−0.899 | −0.954m/s |
| +1m/s末秒轴线偏移 | 约6.4mm | 2.5mm |
| −1m/s末秒轴线偏移 | 约5.3mm | 5.8mm |
| 前进停车24s存活 | 3/3 | 3/3 |
| 后退停车24s存活 | 3/3 | 0/3 |
| 高位后退停车24s存活 | 0/3 | 0/3 |
| 高速直接换向24s存活 | 0/3 | 0/3 |

候选直接reset给±2m/s的固定命令测试，存活为2/3、1/3，低于v16的3/3、3/3。
其存活轨迹速度更接近目标，不能用这些条件均值掩盖失败回合。
前进加速首秒车身pitch仍约−3.63°，整机支撑倾角约+5.61°，反倾问题没有解决。
因此没有全面通过的候选；不能声称本次已修复全部问题，也不推荐将短训权重直接替换v16。

原始结果：`logs/joint_tracking_pos_v17_refine_validation_34000.json`、
`joint_tracking_pos_v17_refine_held2_34000.json`、`joint_tracking_pos_v17_refine_posture_34000.json`。

## 下一轮训练配方：保持五组奖励，增加困难状态覆盖

新stage为`joint_tracking_simple_recovery`，输出版本名使用`joint_tracking_pos_v17`。
使用速度优先profile，但从稳定v16重新开始，不从已出现存活退步的短训候选继续。
这是一份待正式训练验证的配方，不是已经通过验收的模型。

只改变固定的训练采样分布，未加入控制辅助、命令爬坡或新的奖励分项：

- 平移速度幅值1.5–2m/s的采样占比从30%提高到45%（以发生平移采样为条件）。
- 高度新采样额外有40%在0.30–0.32m均匀抽取，其余保留0.22–0.32m全范围。
  reset时新高度抽样落在高段的概率约52%；不等于52%的训练时间在高段。
- 先前命令为纯平移且幅值≥1.3m/s、高度≥0.30m时，
  显式停车/反向/高度切换/转向概率为40%/35%/10%/5%，余10%为原随机混合。
- 先前为高速纯平移且高度较低时，相应概率为20%/50%/15%/5%，余10%为原随机混合。
- 高速显式反向保持原幅值与高度，避免其中一半被替换成0.8–1.2m/s而稀释完整高速换向。
  低速情况保留原来的中速覆盖。显式事件之外的随机混合仍可能产生额外停车或反向。
- 长命令概率从20%降至10%，保留长稳态命令；普通运动3–5s、停车4–8s保持。

这些规则以命令为条件，并不保证当时实际速度已达目标。
每批训练从一开始使用同一规则，不按训练进度改变难度。
执行器、六路独立动作、观测维度、域随机化、推力及终止边界不变。
高车身与高速分布加重也可能影响一般运动，完整评估必须检查低速和转向，不能预先认定更好。

## 正式训练命令

```bash
bash scripts/train_joint_quiet_v23.sh \
  --stage joint_tracking_simple_recovery \
  --version joint_tracking_pos_v17 \
  --checkpoint checkpoints/joint_tracking_pos_v16/checkpoint \
  --reset-vecnormalize --preserve-action-std \
  --learning-rate 0.00003 --updates 4000 --seed 0 \
  2>&1 | tee logs/joint_tracking_pos_v17.log
```

正式4000更新未启动。先做小运行检查，不能以检查通过证明上述失败已经解决。
107项测试通过，包括新的高位停车/全幅换向采样比例、上段高度及高速带覆盖、
相同动作下物理轨迹一致、五组奖励边界和旧阶段回归。
`joint_tracking_pos_v17_recovery_smoke`从v16完成5次更新（4环境×128步，共2560步），
正常保存模型、归一化量及契约；shell语法与`git diff --check`通过。
旧reward对照manifest保留哈希，源码更新后需要新实验`--name`，不覆盖旧实验记录。

## 训练后验收

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v17/checkpoint \
  --profiles reverse10 reverse10_backfirst motion stop2_forward stop2_reverse height2_reverse turn \
  --episodes 5 --seed 36000 --output logs/joint_tracking_pos_v17_transitions.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v17/checkpoint --episodes-per-case 5 --duration 24 \
  --heights .22 .27 .32 --speed 2 --cases stand forward reverse yaw_left yaw_right \
  --seed 36000 --output logs/joint_tracking_pos_v17_held2.json

conda run --no-capture-output -n sim python scripts/audit_joint_tracking_posture.py \
  checkpoints/joint_tracking_pos_v17/checkpoint --seed 36000 \
  --profiles reverse10 reverse10_backfirst settled_forward2 settled_reverse2 \
  --output logs/joint_tracking_pos_v17_posture.json
```

验收优先看各协议存活；同时看实际速度、严格速度带稳定时间、车身反倾、
共同摆腿幅度与真实轮轴偏移，不能只看训练reward或筛选存活轨迹。
