# WheelLeg 文档

本仓库当前只保留轮腿训练项目。主要说明见根目录 `README.md`。

轮腿相关代码位置：

- `omni_drones/envs/twowheel/balance.py`
- `omni_drones/robots/twowheel.py`
- `omni_drones/robots/assets/twowheel_uz05/`
- `cfg/task/TwoWheel*.yaml`
- `scripts/train.py`
- `scripts/play.py`
- `scripts/diagnose_twowheel.py`
- `scripts/sweep_twowheel_stand.py`

最小训练命令：

```bash
conda activate sim
pip install -e .
python scripts/train.py task=TwoWheelBalance
```

USD 设置要点：

- 轮关节 `Lwhl/Rwhl` 不设角度限位，保持可连续旋转。
- 非轮 revolute 关节设有 `[-1.57, 1.57]` 宽限位。
- 任务配置显式覆盖腿部和被动关节 drive，并锁定被动关节。
