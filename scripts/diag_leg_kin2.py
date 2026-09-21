"""用 mj_step 让 connect 等式约束真正求解，再读腿长。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from uz05.env import UZ05Env

env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0, init_scale=0.0)
env.params.domain_randomization.enabled = False
env.reset(seed=0)
sim = env.sim
base = np.asarray(env.params.robot.stand_joint_pos, dtype=np.float64)
print("nominal leg len:", sim.leg_lengths())

def settle(q2, q4, n=400):
    """把主动关节锁到 (q2,q4)（加重 PD），跑 n 步让被动关节/约束收敛。"""
    q = base.copy(); q[0] = q2; q[1] = q4; q[2] = -q2; q[3] = -q4
    sim.data.qpos[sim.hip_qpos_adr] = q
    sim.data.qvel[:] = 0.0
    sim.data.qpos[0] = 0.0; sim.data.qpos[1] = 0.0
    sim.data.qpos[2] = 0.23825
    sim.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for _ in range(n):
        err = q - sim.joint_positions()
        sim.data.ctrl[:4] = 100.0 * err - 4.0 * sim.joint_velocities()
        sim.data.ctrl[4:] = 0.0
        for _ in range(4):
            sim.step()
    return sim.leg_lengths(), sim.data.qpos[2]

print("\n=== 对称驱动 delta 扫描（锁关节后读腿长）===")
for delta in (-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5):
    L, z = settle(base[0] + delta, base[1] + delta)
    print(f"  delta={delta:+.2f} legL={L[0]:.4f} legR={L[1]:.4f} base_z={z:.4f}")

print("\n=== 单 L2 扫描 ===")
for delta in (-0.5, -0.25, 0.0, 0.25, 0.5):
    L, z = settle(base[0] + delta, base[1])
    print(f"  dL2={delta:+.2f} legL={L[0]:.4f} legR={L[1]:.4f} base_z={z:.4f}")

print("\n=== 单 L4 扫描 ===")
for delta in (-0.5, -0.25, 0.0, 0.25, 0.5):
    L, z = settle(base[0], base[1] + delta)
    print(f"  dL4={delta:+.2f} legL={L[0]:.4f} legR={L[1]:.4f} base_z={z:.4f}")
env.close()
