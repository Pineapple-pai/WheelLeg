"""纯几何腿长上限：关掉重力，让腿机构在无载下自由伸展。

区分两件事：
  * **几何上限**：五连杆本身能到的最大髋-轮距离（本脚本，gravity=0）
  * **负载上限**：关节 PD 有限刚度 + 支撑力矩下的实际可达（实测 0.2819 m）

用法::  python scripts/diag_leg_geo_limit.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uz05.env import UZ05Env

env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0, init_scale=0.0)
env.params.domain_randomization.enabled = False
env.reset(seed=0)
sim = env.sim
sim.model.opt.gravity[:] = 0.0        # 无载
base = np.asarray(env.params.robot.stand_joint_pos, dtype=np.float64)
# 站姿下 q2/q4 反号驱动；差模方向 = (q2 增大, q4 减小)
D = np.array([+1.0, -1.0])

def settle(diff: float, n: int = 4000) -> tuple[float, float]:
    q = base[[0, 1]].copy() + D * diff
    qq = np.array([q[0], q[1], q[0], q[1]])
    for _ in range(n):
        err = qq - sim.joint_positions()
        sim.data.ctrl[:4] = np.clip(200.0 * err - 6.0 * sim.joint_velocities(), -20, 20)
        sim.data.ctrl[4:] = 0.0
        for _ in range(4):
            sim.step()
    L = sim.leg_lengths()
    return float(L.mean()), float(L[0] - L[1])

print(f"{'diff(rad)':>10} {'leg':>9} {'L/R(mm)':>9} {'q2':>8} {'q4':>8}")
prev = None
for d in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2):
    sim.data.qpos[:3] = [0.0, 0.0, 0.30]
    sim.data.qpos[3:7] = [1, 0, 0, 0]
    sim.data.qvel[:] = 0.0
    sim.data.qpos[sim.hip_qpos_adr] = np.array([base[0]+d, base[1]-d, base[0]+d, base[1]-d])
    L, lr = settle(d)
    q = sim.joint_positions()
    print(f"{d:>10.2f} {L:>9.4f} {lr*1000:>+9.2f} {q[0]:>+8.4f} {q[1]:>+8.4f}")
# 收缩方向
print()
for d in (-0.2, -0.4, -0.6):
    sim.data.qpos[:3] = [0.0, 0.0, 0.30]
    sim.data.qpos[3:7] = [1, 0, 0, 0]
    sim.data.qvel[:] = 0.0
    sim.data.qpos[sim.hip_qpos_adr] = np.array([base[0]+d, base[1]-d, base[0]+d, base[1]-d])
    L, lr = settle(d)
    q = sim.joint_positions()
    print(f"{d:>10.2f} {L:>9.4f} {lr*1000:>+9.2f} {q[0]:>+8.4f} {q[1]:>+8.4f}")
env.close()
