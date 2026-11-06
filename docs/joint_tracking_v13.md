# v13：高速停车、后退跟踪与实际支撑几何

当前默认阶段为 [v14](joint_tracking_v14.md)，本文保留 v13 历史设置。

## v12 复测

最终 v12（2500 次更新），名义动力学、保留观测噪声、高度 0.27m，种子 20000/20001，
固定命令最多 24s，稳态统计排除首秒：

| 命令 | 实际速度 | 跑满24s | 实际腿长差 RMS | 轮饱和率 |
|---|---|---|---|---|
| +2m/s | +2.014m/s | 2/2 | 7.55mm | 0.59% |
| −2m/s | −1.720m/s | 2/2 | 1.83mm | 0.32% |

后退持续欠速，前进也存在实际支撑不对称；平均值不能覆盖所有高度及换向过程。
轮子饱和率较低，因此尚没有证据表明持续欠速源于执行器一直达到上限。
固定 reverse17 协议种子 20000 在 22s 从 +1.7m/s 切零命令后，23.136s 倾倒；
pitch 达 36.6°，停车锚点位移达 0.705m。另一个 motion 协议中，+2→−2m/s 换向
T90 为 2.36s，末秒实际 −1.724m/s，左腿连杆前伸峰值 0.156m。

报告：`logs/joint_tracking_pos_v12_speed2_diagnosis.json`、
`logs/joint_tracking_pos_v12_switch_diagnosis.json`。
有限种子用来复现问题，不代表总体成功率或已经确认唯一根因。

## 固定修改

1. **停车奖励冲突。** 旧设置在切零命令时固定捕获当前位置，制动产生位移后，
   位置 Huber 成本不断增大，同时位置质量因子将零速度正奖励压低至接近零。
   v13 保留同一个固定锚点、1cm 死区和5cm尺度，位置成本改为
   `4*tanh(old_cost/4)`，绝对成本不超过4；近锚点仍近似原成本。
   零速度和 yaw rate 的正奖励不再乘位置质量因子，固定位置/航向成本继续独立生效。
   这允许策略先学习制动和平衡，同时仍因漂移扣分。远距离位置成本梯度减弱是明确的取舍，
   训练后必须同时检查停车漂移，不能只看是否倒地。
2. **明确高速停车覆盖。** 纯平移且命令幅度大于0.1m/s时，重采样固定有25%概率
   切为零速度、35%概率同幅值直接反向，其余40%使用原命令混合；随机混合也可能采到站立。
   这些显式停车/反向保持原高度，无命令爬坡。保留 v12 全速度范围、运动初态、
   高速后退占比和长命令覆盖，从第一批开始使用。
3. **跟踪与实际几何。** 速度连续误差成本权重1.0→1.4，粗速度奖励尺度0.7→0.5m/s，
   正反方向使用同一公式。实际腿长差权重0.6→0.8（2mm死区、4mm尺度不变）；
   实际连杆前后伸出成本权重0.4→0.6，前伸阈值仍为0.115m。
   pitch、roll、前后错位、高度、漂移、平滑、饱和及抗扰目标继续固定启用。

所有位置和姿态约束均为软成本；不强制左右关节角相等，不锁定关节。
保持两轮独立速度加四关节360°周期位置动作。电机增益和额定限制不变。
无教师、残差、课程、退火、动作滤波、手工刹车或平衡外环；终端日志不增加行数。

## 续训

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh \
  --stage joint_tracking_v13 --version joint_tracking_pos_v13 \
  --checkpoint checkpoints/joint_tracking_pos_v12/checkpoint \
  --reset-vecnormalize --preserve-action-std \
  --learning-rate 0.00003 --updates 4000 \
  2>&1 | tee logs/joint_tracking_pos_v13.log
```

启动脚本默认阶段已改为v13。继承v12策略和探索标准差，重新统计归一化量，学习率固定。
75项测试及10次更新运行检查通过；相同动作下 v12/v13 物理轨迹一致、checkpoint契约兼容。
短训练仅验证运行，尚不能证明实际高速停车、速度跟踪或腿长一致性已达标。

## 训练后验收

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v13/checkpoint --episodes-per-case 5 \
  --duration 24 --heights .27 --speed 2 --cases forward reverse \
  --output logs/joint_tracking_pos_v13_speed2_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v13/checkpoint \
  --profiles stop2_forward stop2_reverse reverse17 motion --episodes 3 \
  --output logs/joint_tracking_pos_v13_braking_eval.json
```

停车协议分别在3s给±2m/s、10s给零命令，观察余下14s。
同时比较生存率、停车T90和制动位移、后退实际速度、换向前伸量、实际腿长差、
pitch/roll、高度、动作变化和饱和率，避免单独优化训练 reward 或 episode length。
