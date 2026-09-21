"""用训练脚本的 make_env 直接跑零动作，看是否复现训练日志里的 1.05°。"""
import sys
import numpy as np
sys.path.insert(0, '/home/p/下载/WheelLeg/scripts')
from scipy.spatial.transform import Rotation  # noqa: E402
from train_uz05 import make_env  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor  # noqa: E402

for ch, res in (("all", (0.0, 0.0, 0.0)), ("all", (0.05, 0.02, 0.05))):
    venv = VecMonitor(DummyVecEnv([
        make_env("stand", r, 0, 2, 0.0, 1.0, False, 1.0, 1.0, res, ch)
        for r in range(4)]))
    venv.reset()
    th = []
    for k in range(1000):
        _, _, dones, infos = venv.step(np.zeros((4, 6), dtype=np.float32))
        for i, info in enumerate(infos):
            th.append(float(info.get("pitch", 0.0)))
            if dones[i]:
                venv.reset()
    # 每环境的 info 是同步的，这里按环境分组统计
    arr = np.asarray(th)
    print(f"channels={ch} residual={res} n={arr.size} "
          f"pitch_rms={np.degrees(np.sqrt(np.mean(arr**2))):.3f}deg "
          f"pitch_max={np.degrees(np.abs(arr).max()):.3f}deg")
    venv.close()

# --- 直接统计 env 内部看到的状态（与 ProgressLogger 用的 info 同源）---
print("\n--- 带 VecNormalize 的训练包装 ---")
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402
venv2 = VecNormalize(VecMonitor(DummyVecEnv([
    make_env("stand", r, 0, 2, 0.0, 1.0, False, 1.0, 1.0, (0.0, 0.0, 0.0), "all")
    for r in range(4)])), norm_obs=False, norm_reward=True, clip_reward=10.0)
venv2.reset()
th2, acc2 = [], []
for k in range(1000):
    _, _, dones, infos = venv2.step(np.zeros((4, 6), dtype=np.float32))
    for i, info in enumerate(infos):
        th2.append(float(info.get("pitch", 0.0)))
        acc2.append(float(info.get("body_accel", 0.0)))
        if dones[i]:
            venv2.reset()
a2 = np.asarray(th2); b2 = np.asarray(acc2)
print(f"n={a2.size} pitch_rms={np.degrees(np.sqrt(np.mean(a2**2))):.3f}deg "
      f"body_accel_rms={np.sqrt(np.mean(b2**2)):.3f}")
venv2.close()
