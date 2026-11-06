# 新版PPO起步诊断与采样对照

本实验从 `checkpoints/joint_tracking_pos_v16_model20261104/checkpoint` 的6000轮状态继续。
不修改物理、奖励、电机映射或动作接口。结果报告位于
`diagnostics/ppo_launch_sampling/report.md`，完整冻结计划为该目录的 `experiment_config.json`。

## 三组采样

| 参数 | original | low_band | launch_focus |
|---|---|---|---|
| 命令模式 | 原v16 | 原v16 | 每回合先站立3秒，再保持起步目标21秒 |
| 非零vx | 原±2范围及其采样 | 原非零vx压缩到绝对值0.25–0.5 | ±0.25/±0.5，四目标等概率 |
| 高度/偏航命令 | 原分布 | 原分布 | 0.27m / 0 |
| 移动起步 | 原概率0.35 | 原概率0.35 | 0 |
| 其余Reset分布 | 原分布 | 原分布 | 原分布 |
| 域随机化/推力 | 原配置 | 原配置 | 原配置 |

low_band是速度命令范围干预；launch_focus是一套集中的任务/初态采样，不能把它的效果唯一归因于速度上限。
每组均由PPO自由输出全部动作，没有姿态引导、控制器残差或参考平滑。

## 训练与评估

每组配对训练种子71/72/73，共9次短训。每次100轮×4环境×256 rollout步=102400步。
统一batch256、学习率1e-4、5epoch、entropy coefficient0.001、固定学习率。
Actor/Critic/Adam、归一化统计及探索标准差均从同一个源检查点继承，保存初态指纹。
这一短训使用较小的共同rollout配置；对照结论适用于三组短训间的采样差异，不能把它当作原6000轮训练的精确延长。

固定评估：种子64000–64002，0.27m、原阶跃、关闭移动起步/推力/域随机化，保留观测噪声。
每回合24秒，3秒下发目标。独立报告姿态、5%跟踪及综合通过率；缺失稳态不填零。
原始诊断还包含种子64003/64004，比较时另提供配对种子指标。

冻结的升速门槛是：四个低速任务各3/3综合通过，在至少两个训练种子上成立，且所有训练种子均保留站立3/3；
满足后才进入±1m/s，再根据实际通过情况进入±2m/s。未过门槛不自动扩大训练速度范围。

## 命令

在 `wheelleg-train` 环境中：

```bash
# 仅核对/生成冻结计划，不训练
python scripts/run_ppo_launch_sampling.py
# 执行配对短训与低速评估；已完成的运行核对哈希后跳过，不覆盖源checkpoint
python scripts/run_ppo_launch_sampling.py --execute --workers 2
# 九组结果齐全后生成表格、曲线与升速门槛核对
python scripts/analyze_ppo_launch_sampling.py
```

底层训练器新增显式开关 `--command-sampling original|low_band|launch_focus`，默认仍为original。
独立训练必须使用新的 `--version`，并通过 `--preserve-action-std` 保留源探索尺度。
各组中间与最终契约均保存采样定义和代码哈希。

不根据单个训练种子的好结果替换正式候选；源6000轮文件持续保留。
