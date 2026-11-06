# 站立和平移的速度精度续训

**本轮1000轮已经完成：[最新复盘与后续验收工具](precision_review_20261105.md)。下文为历史配方，不再是未测推荐；后续配对实验尚未选出可替代当前权重的整体升级。**

本轮5000轮训练的[复盘与配对优化结果](../diagnostics/startup_5000_review/report.md)：零命令站立保留，非零速度综合验收仍未通过，部分起步能力退化。已测1000/3000/5000保存点中，1000轮起步与姿态覆盖更稳，用它作为下一轮起点。

新增joint_tracking_startup_precision仅将现有速度误差惩罚尺度从0.15收紧到0.05m/s，对零命令和平移均生效。其余奖励、动作、电机和物理保持一致。三训练种子各100轮配对短训显示，±0.1/±0.25误差下降，姿态36/36，站立9/9；但5%综合验收仍未通过，±0.5起步也未解决。它是下一轮候选，不是已合格运动策略。

默认保持startup_balanced：25%整回合零速，±0.1/±0.25/±0.5各12.5%；前3秒零速，随后保持同一目标至24秒。高度0.27m、零yaw，无移动起步/推力，保留温和随机化和噪声。另试的startup_low导致+0.25姿态退化，未采用为默认。

完整命令：

```bash
cd /home/p/下载/WheelLeg
SOURCE=checkpoints/joint_stand_translation_20261105_161432/checkpoint_iter_1000 \
VERSION="joint_precision_$(date +%Y%m%d_%H%M%S)" \
SEED=0 bash scripts/train_joint_precision.sh \
  --updates 1000 --rollout-steps 256 --batch-size 256 --num-envs 4 \
  --save-interval 250 --log-interval 1
```

固定lr1e-4、entropy0.001，继承Actor/Critic/Adam和六维探索尺度。首次切换奖励时重置奖励归一化；以后SOURCE指向同精度阶段保存点时，自动保留其归一化统计。脚本不会覆盖已有目录。

1000轮相当于1024000个新转移，本次实际训练约10分18秒，另加评估。每250轮保存，训练结束自动评估35回合。已完成效果复盘，不自动进入±1/±2或长期训练。

中途可单独评估：

```bash
conda run --no-capture-output -n wheelleg-train python scripts/eval_new_model_training.py \
  checkpoints/<版本名>/checkpoint_iter_250 \
  --output diagnostics/<版本名>/evaluation_250 \
  --launch-speeds .1 .25 .5 --seeds 65300 65301 65302 65303 65304
```

评估目录必须不存在。分别查看零命令速度/站位漂移、非零速度精度、姿态和起步存活，不能只看平均回合长度或奖励。若250/500轮已出现站立或低速姿态退化，不应仅为跑满1000轮继续追加预算。
