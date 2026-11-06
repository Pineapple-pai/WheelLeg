# v16：整机支撑倾角、真实轮轴与关节目标协调

## v15最终模型复测

2026-11-03，最终4000轮`joint_tracking_pos_v15/checkpoint`，名义动力学、无推力、
保留观测噪声、高度0.27m、近静止初态、种子20000。报告为：

- `logs/joint_tracking_pos_v15_switch_diagnosis.json`：换向、停车再起步和高速切换。
- `logs/joint_tracking_pos_v15_pitch_axle_audit.json`：车身、整机质心、实际加速度与关节目标。
- `logs/joint_tracking_pos_v15_high_speed_axle_audit.json`：先站立3s，再切±2m/s。

这些是有限复现样本，不是总体成功率或唯一根因的证明。

### 车身确实出现反向倾斜，但整机质心没有反向

模型为X向前、Y向左、Z向上。正pitch将车身+Z转向+X，表示车身上方向前倾。
以轮心中点降低一个轮半径作为支撑几何参考，将整机质心到该参考的向量投影至
航向与世界竖直方向构成的平面，测量整机支撑倾角。这个参考不是力传感器测得的压力中心。
加速度由整机质心速度差分并投影到水平航向计算，未用命令正负代替实际加速度。

`reverse10`各切换首秒的均值：

| 切换 | 实际质心加速度 | 整机质心支撑倾角 | 车身pitch |
|---|---|---|---|
| 站立→+1m/s | +0.968m/s² | +5.48° | −3.29° |
| +1→−1m/s | −1.908m/s² | −10.82° | +6.40° |
| −1→+1m/s | +1.891m/s² | +10.34° | −5.28° |
| +1m/s→站立 | −1.121m/s² | −6.44° | +3.57° |

因此用户观察的“车身反着倾”有实测依据，不能归咎于显示符号。
但不能进一步断言整机质心或前后轮动作符号也反了：整机质心倾角与加速度方向相符。
当前策略通过较大的腿部共同前后摆，允许车身反向倾斜。
v15及此前的pitch奖励一直以车身零pitch为目标，协调姿态成本也以零pitch计算，
并没有明确奖励“加速时车身与整机支撑倾角协调”。这可能形成竞争目标，仍需训练/消融验证。
不把稳态行驶方向当作倾角方向：制动时加速度方向与原速度方向相反。

### 腿长相等不保证轮轴同轴，目标角和实际角均存在差异

两轮轴方向接近平行（夹角数值量级约10⁻⁶°），但两轴所在直线不同。
真实错轴距离取各轮心到另一侧轮轴直线的垂距最大值，包含前后及垂直偏移，
不会因为轮轴平行或腿长相等而自动归零。

`reverse10`中±1m/s片段，以及先站立3s再给−2m/s的末秒均值：

| 目标/实际速度 | 真实腿长差 | 真实轴线偏移 | L1−R1/L2−R2目标差 | 对应实际角差 |
|---|---|---|---|---|
| +1 / +0.96m/s | 3.72mm | 13.44mm | +3.95/+2.28° | +4.10/+2.63° |
| −1 / −0.93m/s | 0.99mm | 26.79mm | −6.31/−6.34° | −6.74/−7.13° |
| −2 / −1.79m/s | 0.19mm | 53.94mm | −12.65/−12.04° | −13.57/−13.64° |

目标角差按周期最短角差计算。MJCF对应关节的正轴方向一致，reset时两组轴点积均为1，
因此这里的L1/R1、L2/R2可以直接比较，未将镜像轴或±2π表示误认为目标不等。
目标差并非所有时候都应强制为零：驱动负载、反馈误差或主动转向可能需要不同目标。
实际轴线几何是主要约束，目标协调仅为较弱的软成本。

v15还没有解决切换摆腿：±1m/s换向轮位偏移仍可达约22.7cm。
`motion`中+2→−2m/s于11.272s触发轮位终止。
先站立3s后直接给+2m/s，于3.712s倾角终止；失败片段不能算作高速稳态能力。
直接reset即给±2m/s也分别在0.560/0.584s倾角终止，完整初态覆盖仍需验收。

## v16修改

1. **用物理支撑倾角误差替换固定零pitch误差。**
   将车身上方向与“轮轴支撑参考→整机质心”的向量投影到航向竖直平面，
   取二者的周期角差。原pitch对齐奖励、pitch连续误差成本和协调姿态成本改用该误差。
   稳态质心接近支撑正上方时仍鼓励车身接近竖直；加速时允许车身沿实际质心倾斜方向变化。
   不按速度命令设置正负pitch，不使用`atan(a/g)`作为教师角度，也不修改PPO动作。
   这是几何目标，不宣称任意瞬间倾角必须与加速度同号；完整动力学还受角动量和驱动作用影响。
2. **直接约束真实轴线偏移。**
   `wheel_axle_alignment = -HuberSquare(max(0, axle_offset−0.002)/0.010)`。
   2mm死区、10mm尺度、权重1，在所有模式固定启用。27mm偏移成本为−4，
   54mm偏移成本约−9.4。不是用更小的腿长差间接代替同轴性。
3. **较弱的关节目标协调。**
   仅站立/直行（yaw命令绝对值≤0.01）计算L1−R1、L2−R2的周期最短角差，
   2°死区、8°尺度、权重0.2，先计算每对成本再平均。
   主动转向时此成本为零，但真实轴线目标仍启用。六路动作继续独立，没有强制同目标。

保留v15各腿前后支撑位置成本，以及原速度误差、换向采样、腿长、roll、
高度、漂移、平滑、饱和和连杆范围目标。物理参数、执行器增益、观测维度和终止边界不变。
v15阶段和奖励保留，可继续复现旧模型。更强同轴目标及倾角目标可能影响跟踪、转向和抗扰，
需联合验收；权重或几何目标变化不能作为已改善的证据。

回放HUD新增：整机支撑倾角、车身与质心倾角误差、轴线偏移、轴夹角、腿长差、
四个关节目标及对应左右目标差。已打开的回放进程需要重启后才能显示。

## 验证与续训

92项测试通过，包括正pitch前倾符号、质心方向、空间变换不变性、
平行但不共线的轮轴、周期目标差、转向下目标独立性和v15相同动作下的物理轨迹回归。
从v15继承的10次更新（40960步）`joint_tracking_pos_v16_smoke`完成，正常写出模型及归一化量。
短训仍有倾角、轮位和触地终止，仅验证流程；完整v16尚未启动。

```bash
bash scripts/train_joint_quiet_v23.sh \
  --stage joint_tracking_v16 --version joint_tracking_pos_v16 \
  --checkpoint checkpoints/joint_tracking_pos_v15/checkpoint \
  --reset-vecnormalize --preserve-action-std \
  --learning-rate 0.00003 --updates 4000 \
  2>&1 | tee logs/joint_tracking_pos_v16.log
```

## 训练后验收

```bash
conda run --no-capture-output -n sim python scripts/audit_joint_tracking_posture.py \
  checkpoints/joint_tracking_pos_v16/checkpoint \
  --profiles reverse10 reverse10_backfirst settled_forward2 settled_reverse2 held_forward2 held_reverse2 \
  --output logs/joint_tracking_pos_v16_pitch_axle_audit.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v16/checkpoint \
  --profiles reverse10 reverse10_backfirst stop_go10 stop_go10_backfirst motion height2_forward height2_reverse \
  --episodes 5 --output logs/joint_tracking_pos_v16_switch_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v16/checkpoint --episodes-per-case 5 --duration 24 \
  --heights .22 .27 .32 --speed 2 --cases stand forward reverse yaw_left yaw_right \
  --output logs/joint_tracking_pos_v16_held_eval.json
```

倾角/轴线审计需更换种子重复；默认只有20000，不能当作统计成功率。
优先比较存活、轮轴偏移、车身与质心倾角误差、共同摆腿幅度/持续时间，
同时检查关节目标/实际角差、真实腿长、T90、±0.05m/s稳定时间和停车漂移。
