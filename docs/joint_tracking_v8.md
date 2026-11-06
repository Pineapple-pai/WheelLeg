# v8：±2m/s、0.22–0.32m 高度与切换响应

启动脚本现默认 v11，见 [v11 复现、修改与命令](joint_tracking_v11.md)。复现本页训练需
显式指定 `--stage joint_tracking_v8` 并设置独立 `VERSION`。

默认阶段 `joint_tracking_v8`，输出 `checkpoints/joint_tracking_pos_v8/`。
本次完成范围、动作接口、固定训练分布、奖励及评估修改；正式长训练尚未开始，
不能把可配置 ±2m/s 当成已经学会高速行驶。

## v7 问题复现与物理检查

v7 最终模型、名义动力学、种子 20000，高度始终命令 0.27m：
站立→+0.5m/s→−0.5m/s→左转→站立，每段 3 秒。
前进转后退阶段，最低高度约 0.2175m、pitch 峰值约 7.35°；左转段约 8.12°。
这确认了在没有降低高度命令时，切换也会造成约 5.25cm 降高。
[复现统计](../logs/joint_tracking_pos_v7_transition_bounds_audit.json)保留了各段结果。

过去 0.24–0.30m 是**车身相对地面的高度目标**，不是三维几何腿长。
本次按车身高度扩展为 0.22–0.32m，实际腿长仍从髋部到轮心几何计算，不能混为一个量。
离线闭链运动学检查找到两端均可双轮接地的构型，闭链残差小于 4e-12m；
直立、轮心前后对齐时，两端腿长分别约 0.165m 和 0.265m。
检查没有修改初始化姿态，也没有向训练提供解算动作或数据。
[几何检查](../logs/joint_tracking_v8_height_reachability.json)仅说明几何可达，不证明动态平衡和驱动余量。

## 高速动作接口

v1–v7 轮速动作缩放为 16rad/s，配合 0.055m 轮半径，纯滚动约 0.88m/s。
只扩大 vx 命令范围无法让这个动作接口达到 2m/s。

v8 两轮独立速度目标改为 ±48rad/s；四关节周期位置目标仍为 ±π 对应的最近等价位置，
允许连续跨圈。2m/s 约需要 36.36rad/s；按半轮距约 0.2103m、同时转向 1.5rad/s 估算，
外侧轮约需 42.1rad/s，动作范围保留一定调整空间。
电调速度上限 59.79rad/s、额定电流 10A、关节额定力矩 20N·m 等原有限幅均未增加。
这是指令容量检查，不能代替高速下的实际跟踪与饱和测试。

轮子初始归一化动作标准差设为 0.02，保持 `48*0.02 = 16*0.06 = 0.96rad/s` 的初始物理探索量级；
四关节初始标准差仍为 0.06。之后六个标准差均由 PPO 自由学习，不做退火。
学习率 1e-4、熵系数 0.001 保持固定。

动作缩放是接口的一部分，契约会拒绝把 v7 checkpoint 直接当 v8 续训或回放。
默认从头训练，旧模型和旧阶段的 16rad/s 动作含义保留。

## 固定训练覆盖

从第一个 reset 就启用全范围：vx ±2m/s、yaw ±1.5rad/s、车身高度 0.22–0.32m。
没有课程、阶段扩范围或退火。基础命令仍覆盖站立、纯平移、纯转向、组合四种模式。
平移幅值按固定的三个速度带等概率采样，再选正负方向：

* 0–0.5m/s
* 0.5–1.25m/s
* 1.25–2m/s

每次重采样时，固定 50% 只改运动目标、25% 只改高度、25% 两者都改。
只改高度时保留运动命令，可以训练行驶中升降；只改运动时保留高度目标，专门暴露切换降高。
初始命令同时采样所有量，不继承前一回合目标。

固定 40% 采用 12–24 秒长保持；其余运动命令保持 1–4 秒，零命令保持 2–6 秒。
回合仍最长 24 秒，既保留长时间恒速任务，也增加反向、转向、停止和高度切换覆盖。
命令直接阶跃，不添加斜坡器、动作滤波或姿态恢复控制器。

## 固定软代价

保留 v7 的 pitch、roll、实际腿长差、前后构型、支撑、平滑、饱和、零命令锚点等目标。
补充三项，全部从第一步固定启用，`H(x)` 为绝对值小于 1 时平方、大于 1 时线性增长的函数：

| 项目 | 定义 |
|---|---|
| 高度不足 | `−0.5 H(max(0, height_target-height_actual-0.003)/0.015)` |
| 密集 vx 误差 | `−0.15 H((vx_actual-vx_target)/0.30)` |
| pitch 角速度 | `−0.04 H(pitch_rate/1.0)` |

高度不足代价针对实际降高，不硬锁高度；原有双向高度跟踪仍保留。
密集速度误差在高速误差较大、指数跟踪收益很小时仍提供得分差异。
pitch 近零目标保持，角速度项另约束快速姿态变化；不规定恢复动作的角度或左右关节相等。
这些目标有物理上的竞争关系，正式训练后仍需同时验收高度、响应时间、pitch 和饱和，不能只压低其中一个值。

新增详细项只写入 TensorBoard：`Reward/tracking_deficit_cost_mean`、
`Tracking/height_shortfall_rms_m`、`Posture/pitch_rate_rms_rad_s`，不继续增加终端逐轮行数。
pitch 角速度代价计入原有控制代价分组；训练配置与契约记录实际生效奖励、范围和动作缩放。

## 验证状态

45 项测试通过，覆盖高速轮速容量、额定限幅保持、旧接口拒绝加载、首批全速度带、
独立运动/高度切换、额外软代价、响应时间必须持续满足误差带等。
完成从头 PPO 30 次更新、保存和固定/切换评估、短视频回放链路验证。
短跑模型仍会很快倒下，不是 v8 性能证明。

## 训练命令

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh 2>&1 | tee logs/joint_tracking_pos_v8.log
```

默认 6000 次更新、8 环境、rollout 512，共 24,576,000 个采样步。
这是扩展后的首次训练配置，不能预先保证 6000 轮足以掌握全部高速任务，后续按独立验收结果判断。
仍是纯 PPO 六维轮速/位置动作，无教师、残差、课程、退火或手写平衡辅助。

## 训练后验收

固定目标需要分别测低速、中速和高速，而不是只查看平均速度误差：

```bash
for speed in 0.3 1.0 2.0; do
  conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
    checkpoints/joint_tracking_pos_v8/checkpoint --speed "$speed" \
    --heights .22 .27 .32 --duration 24 --episodes-per-case 3 \
    --output "logs/joint_tracking_pos_v8_speed_${speed}_eval.json"
done

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v8/checkpoint --episodes 3 \
  --output logs/joint_tracking_pos_v8_transition_eval.json

conda run --no-capture-output -n sim python scripts/eval_joint_tracking_transitions.py \
  checkpoints/joint_tracking_pos_v8/checkpoint --episodes 3 \
  --domain-randomization --perturbations \
  --output logs/joint_tracking_pos_v8_transition_stress_eval.json
```

切换评估覆盖 0→1→2→−2m/s、行驶中左右转、停止、0.22↔0.32m 高度改变和行驶中升降。
每段记录响应时间、最低高度、最大高度不足、pitch 峰值/角速度峰值、末秒实际速度和停止锚点误差。
响应时间定义为进入初始到目标变化量 10% 或绝对容差的误差带（取较大者），并连续保持至少 0.25 秒；
vx 绝对容差 0.05m/s、yaw 0.10rad/s、高度 5mm。`null` 表示该段结束前未满足，不能算快速跟踪成功。
同时看末秒实际速度，避免一次经过目标值被误认为已经稳定。

固定评估不传 `--speed/--heights` 时，v8 默认使用 2m/s 和 .22/.27/.32m；旧阶段默认保持原配置。
有限种子通过后还需扩大验证，尤其是极限高度下的高速反向和受扰恢复。

## 终端日志

默认每轮省略 14 项次要或重复指标，保留 PPO 主要诊断、reward/length、跟踪误差、
漂移、输出变化、饱和、pitch/roll 和实际腿长/前后构型差。完整训练指标继续写入 TensorBoard，
不影响奖励、训练或保存。需要展开时使用 `bash scripts/train_joint_quiet_v23.sh --log-verbose`；
也可用 `--log-interval 10` 改为每 10 轮打印一次。

## 原生 MuJoCo 回放

仍保留原生左右侧栏、棋盘背景、顶部姿态/实际与期望速度、数字键盘模式与 `-/+` 调节。

顶部文字使用原生屏幕叠加，在鼠标点击、拖动和缩放时保持固定。
显示进程单独使用 `.cache/mujoco-native-ui-3.3.7` 中的 MuJoCo 3.3.7；
物理步进、执行器和策略推理仍使用 `sim` 环境原有的 MuJoCo 3.2.3。
显示进程只接收状态，不运行 `mj_step`；鼠标施加的扰动力返回物理进程。
新机器首次使用需安装显示依赖（不会升级 `sim` 环境中的包）：

```bash
conda run -n sim python -m pip install --no-deps \
  --target .cache/mujoco-native-ui-3.3.7 mujoco==3.3.7
```

```bash
bash scripts/replay_joint_tracking.sh --keyboard --nominal \
  --checkpoint checkpoints/joint_tracking_pos_v8/checkpoint

bash scripts/replay_joint_tracking.sh --nominal --case forward \
  --speed 2 --height .27 --duration 24 --once \
  --checkpoint checkpoints/joint_tracking_pos_v8/checkpoint
```

键盘模式仍默认从 0.3m/s 起，`+` 可逐步提高到 2m/s；`--speed 2` 可直接设置起始幅值。
`7` 为零速度站立，`R` 重置；`0` 不再绑定站立。
5/6 和站立/高度模式的 `-/+` 现可在 0.22–0.32m 范围调节。
旧 v7 模型不会因为回放参数或窗口显示改变就具备新范围能力。
