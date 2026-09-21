"""验证五连杆腿运动学求解器（对照 MuJoCo 实测）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402
from uz05.leg_kinematics import (  # noqa: E402
    forward, inverse_kinematics, nominal_angles,
)

env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0, init_scale=0.0)
env.params.domain_randomization.enabled = False
env.reset(seed=0)
th = nominal_angles(env)
pts = forward(th)
print("nominal θ2,θ4,θ5,θ3,θ1,θ6 =", np.round(th, 5))
print("FK 轮心 (y,z) =", np.round(pts["F"], 5),
      "⇒ 腿长", round(float(np.linalg.norm(pts["F"])), 5))
print("实测腿长 =", np.round(env.sim.leg_lengths(), 5))
print("闭环误差 Q3-P4 =", np.round(pts["Q3"] - pts["P4"], 6),
      "  Q6-Q2 =", np.round(pts["Q6"] - pts["Q2"], 6))

print("\n=== IK：目标腿长扫描（x=0）===")
seeds = [th] + [np.random.default_rng(i).uniform(-1.6, 1.6, 6) for i in range(40)]
for L in (0.14, 0.16, 0.18, 0.184, 0.22, 0.26, 0.30, 0.34, 0.36):
    sols = inverse_kinematics(0.0, L, seeds=seeds)
    if not sols:
        print(f"  L={L:.3f}: 无解")
        continue
    print(f"  L={L:.3f}: {len(sols)} 解")
    for s in sols[:3]:
        t = s["thetas"]
        print(f"      θ2={t[0]:+.4f} θ4={t[1]:+.4f}")

print("\n=== 用 MuJoCo 校验一个 IK 解：锁关节 → 稳态腿长 ===")
sols = inverse_kinematics(0.0, 0.26, seeds=seeds)
if sols:
    t = sols[0]["thetas"]
    q = np.array([t[0], t[1], -t[0], -t[1]], dtype=np.float64)
    sim = env.sim
    sim.data.qpos[sim.hip_qpos_adr] = q
    sim.data.qvel[:] = 0.0
    sim.data.qpos[0] = 0.0
    sim.data.qpos[1] = 0.0
    sim.data.qpos[2] = 0.30
    sim.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for k in range(600):
        err = q - sim.joint_positions()
        torque = 100.0 * err - 4.0 * sim.joint_velocities()
        np.clip(torque, -20.0, 20.0, out=torque)
        sim.data.ctrl[:4] = torque
        sim.data.ctrl[4:] = 0.0
        for _ in range(4):
            sim.step()
    print(f"  目标 0.260 → 实测腿长 {sim.leg_lengths()} "
          f"(mean {sim.leg_lengths().mean():.4f})  base_z={sim.data.qpos[2]:.4f}")
else:
    print("  0.26 无解")
env.close()
