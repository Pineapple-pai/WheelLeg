# 2026-11-04：续训修复、简化短训筛选与尺度校准

本轮先做实测筛选，不把新增版本或测试通过当成策略问题已经解决。
没有改变电机参数、动作自由度、终止边界或加入手写控制辅助。

## 续训加载修复

在checkpoint重建模型之前显式传入请求的seed、n_steps、batch_size、
learning_rate、gamma、GAE lambda与entropy参数。旧加载路径会恢复源模型保存的设置，
仅修改learning_rate属性也没有更新内部schedule。
这会破坏多种子和不同rollout预算的受控比较。

新增真实CLI回归：源seed7/rollout8/LR1e−4，实际续训输出必须为
seed23/rollout16/LR3e−5、2次更新共32步，而不是只检查打印的参数。
v15/v16已完成正式训练的seed0/rollout512/LR3e−5一致，不据此否定既有模型实测。

## 第一轮筛选：共同v15起点

5组均从同一v15最终模型续训，训练seed0，8环境×512步×50更新=204800步。
评估seed31000，六个关键24s切换协议及0.27m固定±2m/s。
这是单训练种子短训，不能替代正式多种子实验，也不能据此否定某组长训后的潜力。
原始manifest、每组模型和报告均保存在`logs/reward_screen_20261104_pilot/`。

以下为reverse10中+1与−1片段末秒速度和轴线偏移，双方片段均跑完：

| 方案 | +1实际速度 | −1实际速度 | 前进/后退轴线偏移 |
|---|---:|---:|---:|
| v15继续训练对照 | +0.934 | −0.936 | 14.5/7.9mm |
| 简化，零pitch，无同轴 | +0.931 | −0.897 | 14.1/28.3mm |
| 简化，零pitch，有同轴 | +0.888 | −0.927 | 5.1/4.6mm |
| 简化，支撑参考，无同轴 | +0.915 | −0.951 | 11.2/15.0mm |
| 简化，支撑参考，有同轴 | +0.930 | −0.946 | 1.8/6.6mm |

速度单位m/s。五组均只跑满六个协议中的两个±1m/s换向协议；
motion、stop2_forward、stop2_reverse及height2_reverse均未跑满。
支撑参考＋同轴组在stop2_forward中能完成加速，但切零后仍于10.336s轮位终止；
不能把它描述成停车已通过。

对照组仅继续训练也改变了指标，因此不能把某项变化全部归因于新增reward。
单纯删除奖励项没有立即解决问题；同轴项对这一短训样本有作用，但依旧可能伴随欠速。

## 第二轮：共同v16起点，分离速度尺度和失败成本

第一轮的简化任务尺度对0.1m/s误差只扣约0.111/s，
而10mm轴线偏移扣0.64/s，存在接受速度偏差以换取几何成本降低的空间。
这是成本定义的量级审计，不是唯一根因证明。

三个组从同一v16最终模型开始，4环境×512步×50更新=102400步。
仍为训练seed0、评估seed31000；与第一轮的共同起点及预算不同，不混成同一个因果比较。
第二轮目录为`logs/reward_calibration_20261104_pilot/`。

| 实验臂 | stage | vx尺度/权重 | 失败事件成本 |
|---|---|---|---:|
| support_axle | joint_tracking_simple_support_axle | 0.30m/s / 1 | −5 |
| tight_axle | joint_tracking_simple_tight_axle | 0.05m/s / 1/6 | −5 |
| tight_survival | joint_tracking_simple_tight_survival | 0.05m/s / 1/6 | −20 |

所有组保留相同的五个分组：任务、姿态、动作平滑、失败、同轴。
没有增加腿长、关节目标相等或其他几何项。

速度成本`w*ρ(e/s)`的大误差斜率为`2w/s`。
新旧都为6.667，因此不会仅因校准而将高速阶跃的惩罚斜率放大六倍；
零附近曲率提高6倍，0.1m/s误差成本从0.111/s变为0.5/s。
失败成本单独设第三组验证：−20相当于5s最大正奖励率4/s的预算，
目的是检查快速失败是否比恢复更有吸引力，而不是预先断言此权重最优。

103项测试通过，包括大误差斜率不变、小误差曲率六倍以及仅切换失败成本时其他项不变。
第二轮三个组均完成50次更新。初筛seed31000中，仅提高失败成本的组
跑满height2_reverse；另外两组仍在18s切站立后轮位终止。

## 独立初验与候选续训

初筛后将tight_survival候选与原v16在未用于筛选的32000–32002种子上重测：

| 协议 | 原v16 | tight_survival候选 |
|---|---:|---:|
| reverse10、reverse10_backfirst | 各3/3 | 各3/3 |
| stop2_forward、stop2_reverse | 各3/3 | 各3/3 |
| height2_reverse（0.32m后退切站立） | 0/3 | 3/3 |
| motion（包含+2→−2m/s） | 0/3 | 0/3 |

原始报告：`logs/reward_calibration_20261104_pilot/v16_validation_32000.json`、
`tight_survival_validation_32000.json`。这是有限种子、名义动力学结果，不代表总体成功率。
仍存在速度欠跟踪和高速换向失败，不能宣称完整修复。

候选已继续训练300次更新，输出`checkpoints/joint_tracking_pos_v17_candidate/`，
stage为`joint_tracking_simple_tight_survival`，并没有新增一个堆叠reward的v17阶段。
同阶段续训复用候选VecNormalize，不重新清空奖励统计。训练结束后自动执行
三个种子的关键切换、固定±2m/s与车身/质心/轮轴审计。
日志为`logs/joint_tracking_pos_v17_candidate.log`，此时该续训尚未结束。

后续检查：该运行在20/300次更新后已不在运行，目录没有保存的模型ZIP，
不能视为完成或直接用作续训来源。后续独立试验与有效模型路径见
`optimization_20261104_priority.md`；原目录和日志保留。

```bash
bash scripts/train_joint_quiet_v23.sh \
  --stage joint_tracking_simple_tight_survival --version joint_tracking_pos_v17_candidate \
  --checkpoint checkpoints/reward_calibration_20261104_pilot_tight_survival_seed0/checkpoint \
  --preserve-action-std --learning-rate .00003 --updates 300 \
  --num-envs 8 --save-interval 100 --log-interval 10
```
