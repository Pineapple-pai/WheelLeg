"""验证：训练环境构造与评估环境构造是否给出同一条轨迹。

背景：零残差（策略完全无影响力）的训练里，pitch RMS 也到 1.05°、存活率掉到
0.92，而 accept_stand.py 同样零动作只有 0.12°。必须定位差异。
"""
import sys
import numpy as np
sys.path.insert(0, '/home/p/下载/WheelLeg/scripts')
from scipy.spatial.transform import Rotation  # noqa: E402
from uz05.env import UZ05Env  # noqa: E402
from dataclasses import replace  # noqa: E402


def run(tag, **kw):
    env = UZ05Env(stage='stand', stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=1.0, **kw)
    env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=1200)
    env.reset(seed=0)
    th, xs = [], []
    info = {}
    for k in range(1200):
        _, _, term, trunc, info = env.step(np.zeros(6))
        q = env.sim.base_quat
        rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')
        th.append(float(rpy[1])); xs.append(float(info['station_error']))
        if term or trunc:
            break
    env.close()
    th = np.asarray(th); xs = np.asarray(xs)
    print(f"{tag:44s} n={len(th):4d} prms={np.degrees(np.sqrt(np.mean(th**2))):6.3f} "
          f"xpeak={100*np.abs(xs).max():6.2f} coord_mix={env.coord_mix} "
          f"res={env.coord_residual_scale} ch={env.coord_channels}")


run("默认 (residual 0.05/0.02/0.05)", coord_mix=1.0)
run("训练构造 residual=0", coord_mix=1.0, coord_residual_scale=(0.0, 0.0, 0.0))
run("训练构造 residual=0 + channels=all", coord_mix=1.0,
    coord_residual_scale=(0.0, 0.0, 0.0), coord_channels="all")
run("coord_mix=1.0 explicit", coord_mix=1.0, coord_residual_scale=(0.05, 0.02, 0.05))
