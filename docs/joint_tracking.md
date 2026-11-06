# 从零联合训练高度、vx、yaw

**当前启动脚本默认使用 v13，见 [v13 修改与命令](joint_tracking_v13.md)。下文保留 v2 历史配置与回放说明；其中未指定阶段的训练命令也会启动 v13。**

当前默认是姿态修正版 `joint_tracking_v2`，输出版本 `joint_tracking_pos_v2`。
旧 `joint_tracking` / `joint_tracking_pos_v1` 保留用于重现已发现的斜撑姿态问题。

训练入口：`bash scripts/train_joint_quiet_v23.sh`。这是独立的 `joint_tracking_v2` 模式，
无须站立 checkpoint。第一批环境就使用完整固定命令分布，训练过程中不扩范围、不收紧边界。

| 命令 | 范围 |
|---|---|
| 离地高度 | 0.24–0.30 m，所有模式均采样 |
| 机体系 vx | −0.5–0.5 m/s |
| yaw 角速度 | −1.5–1.5 rad/s |

停止、纯平移、纯旋转、平移加旋转各占采样概率 25%；每 250–624 个控制步
重新采样命令（约 2–5 秒）。零速度命令保留有效高度目标。零速度和运动使用同一套
速度跟踪奖励，没有站立阶段、教师、残差、命令前馈、指定动作方向奖励、辅助损失、
学习率退火或手写平衡外环。默认不做域随机化，观测保留现有传感器噪声。

六维 PPO 动作顺序：`[左轮速度, 右轮速度, 左腿1位置, 左腿2位置, 右腿1位置, 右腿2位置]`。
轮子各自缩放至 ±16 rad/s，以覆盖组合运动；关节缩放至 ±π rad 的周期绝对位置，
执行时选取距离当前角最近的等价整圈位置，跨 ±π 时可连续越界。
不限制累计转数，没有原来的 ±0.35 rad 偏置范围和 ±1.2 rad 软件角度惩罚。
actor 编码器角度也按周期折回 ±π；模型不设置主动铰链角度限位。
关节仍受原有闭链约束、腿长 tendon 和碰撞限制，不能据此声称机构中任意四个关节
都能独立无碰撞地转整圈。

关节驱动器执行位置 PD，轮驱动器执行速度环；这是执行 PPO 目标的电机模型。
PPO 不输出力矩，控制接口没有姿态修正叠加。
奖励包含高度、vx、yaw 跟踪、直立、侧向/竖直速度、角速度、周期动作变化、归一化执行力矩、
机身碰撞和倾倒惩罚。没有固定关节角、腿长、关节静止奖励，也不奖励无意义的摆腿。
高度变化需要腿部调节，运动和平衡需要轮腿共同控制；是否形成协同必须通过回放验证。
零速度跟踪约束速度，**不保证零命令始终保持起始坐标和绝对航向**。

默认 5000 次更新、8 环境、每环境 512 步，共 20,480,000 个采样步。
学习率固定 1e-4，初始动作标准差 0.08，此后由 PPO 学习标准差；熵系数固定 0.005。
每 100 次更新保存策略、归一化统计量和接口契约。日志记录速度/高度跟踪 RMSE；
该模式不再打印旧站立验收结论。新旧动作语义不同，契约会阻止直接加载旧站立策略。

```bash
cd /home/p/下载/WheelLeg
mkdir -p logs
bash scripts/train_joint_quiet_v23.sh > logs/joint_tracking_pos_v2.log 2>&1
```

另一个终端可以执行 `tail -f logs/joint_tracking_pos_v2.log` 查看进度。
CPU 负责小型 MLP 和 MuJoCo 并行环境，可用脚本末尾参数覆盖设置，例如 `--device cuda`。
换种子时同时换输出版本，避免覆盖：

```bash
VERSION=joint_tracking_pos_seed1 bash scripts/train_joint_quiet_v23.sh --seed 1
```

独立回放在三种高度下测试零命令、前后平移、左右旋转、组合运动，输出存活率、
跟踪误差和各关节/轮子速度 RMS。RMS 只能说明活动程度，不能单独证明因果协同。

```bash
conda run --no-capture-output -n sim python scripts/eval_joint_tracking.py \
  checkpoints/joint_tracking_pos_v2/checkpoint \
  --episodes-per-case 10 --output logs/joint_tracking_pos_v2_eval.json
```

验收应同时检查存活率、vx/yaw/高度误差、零命令漂移，以及关节是否能随高度和扰动调整。
配置和短跑通过不等于模型已经学会这些行为。

日志按 `Learning iteration 当前/总数` 分块对齐输出，每次 PPO 更新完成后打印该次
采样/学习耗时、吞吐率、损失、动作标准差、累计步数和 ETA。
`Tracking/tracking_*_error` 是本轮全部环境采样的平均绝对误差；
`Tracking/wheel_target_delta_rms` 是同一 episode 内相邻控制步轮速目标变化的 RMS（rad/s）。
平均回报/长度使用 PPO 的 episode 窗口，存活率使用最近 50 个完成的 episode，
没有完成回合时显示 n/a。`--log-verbose` 可追加状态与奖励分项。
当前 PPO 没有 leg_feedback 模块，不输出示例中该模块的专属指标。
已启动的 Python 进程继续使用启动时的代码，新日志格式在下次启动或续训时生效。

MuJoCo 原生窗口回放最终模型（循环切换零命令、前进、旋转、组合运动和高度变化）：

```bash
bash scripts/replay_joint_tracking.sh --case suite
```

单项回放可选 `--case stand`、`forward`、`reverse`、`yaw`、`mixed`、`height` 或
`random`（训练时的随机变命令）。`--once` 在选定回放结束后退出。
默认使用本机实物显示网格；其他机器可指定 `--mesh-dir`，或 `--no-mesh` 使用碰撞几何。
导出视频：

```bash
bash scripts/replay_joint_tracking.sh --case suite --video logs/joint_tracking_pos_v2_replay.mp4
```

独立评估还支持 `--ablation wheel_zero` 或 `--ablation joint_hold`，默认先让完整策略
运行两秒，再将轮速目标置零或将四关节目标保持在两秒时的实际关节角，比较后续表现。
这是诊断干预，不能当作经过单独训练的消融策略。


姿态修正版使用机体系下「轮心相对同侧髋部的前后位置」评估支撑几何，不固定关节角度。
左右轮前后错位超过 4 cm、轮子相对髋部前后偏置超过 8 cm 后开始软惩罚，
正向任务奖励也随几何异常连续衰减。前后错位超过 15 cm 或单轮偏置超过 20 cm，
持续 25 个控制步则终止；重置前 50 个控制步不累计此终止计数。
这些阈值是当前平地站立/速度跟踪任务的设计参数，定义在 `StageSpec`，不代表机械行程实测值。
执行器归一化力矩惩罚提高到旧版的 5 倍，以减少长期饱和；没有强制静止关节或增加教师。

验证记录：8 项行为检查和两轮真实 PPO 短跑通过。旧策略在新姿态条件下第 75 步
（0.6 秒）触发 `wheel_stance_limit`，当时左右轮前后错位 54.45 cm。
这只能证明异常姿态会被拒绝，新目标是否能训练出合格策略仍需从头训练验证。

```bash
bash scripts/train_joint_quiet_v23.sh > logs/joint_tracking_pos_v2.log 2>&1
```

新模型回放需显式指定对应 checkpoint：

```bash
bash scripts/replay_joint_tracking.sh --checkpoint checkpoints/joint_tracking_pos_v2/checkpoint --case suite
```


数字键盘交互回放（默认加载已训练的 v2 模型）：

```bash
bash scripts/replay_joint_tracking.sh --keyboard
```

点击 MuJoCo 场景获得键盘焦点。小键盘及主键盘数字均支持：

```text
1 前进        2 后退
3 原地左转    4 原地右转
5 降低1cm     6 升高1cm（进入站立调高度模式）
7 零速度站立  R 重置回合
- / +：调当前模式的速度大小、转速大小或高度
```

手动模式切换保留当前高度并即时改变命令，不重置姿态；高度范围 0.24–0.30m。
前进/倒退初始目标为 ±0.3m/s，旋转初始目标为 ±1rad/s。
`- / +` 在前进/后退模式每次调 0.05m/s，转向模式每次调 0.1rad/s，
站立/高度模式每次调 1cm；后退与右转时 `+` 表示加快，`-` 表示减慢。
主键盘 `=` 同样作为 `+`，支持小键盘加减键，数值限制在训练范围内。
每回合时长采用所属阶段的配置（v2 为 12 秒，v5–v7 为 24 秒），
结束后自动重置并保持选定模式；`--once` 可改为单回合结束后退出。
键盘模式用于原生窗口，与 `--video` 互斥。回调通过消息队列传递按键，
由仿真线程修改命令，策略及六维动作接口不变。
