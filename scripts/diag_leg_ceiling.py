"""腿长上限的成因诊断：是几何限位，还是关节行程/力矩限制？

对固定差模偏置扫一遍，记录稳态腿长 + 4 个主动关节角，判断：
  * 腿长在哪一点饱和
  * 饱和时关节角是否已经顶到 action 限幅（±0.35 rad 偏置）
  * 两条腿的上限是否一致
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataclasses import replace
from uz05.env import UZ05Env

print(f"{'d_trim':>8} {'legL':>8} {'legR':>8} {'q2':>8} {'q4':>8} {'off2':>8} {'off4':>8} "
      f"{'base_z':>8} {'I_rms':>6} {'n':>5}")
for d in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0):
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0, init_scale=0.0,
                  lock_stand_leg_actions=False, stand_leg_action_limit=1.0, coord_mix=1.0)
    env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=2600)
    env.reset(seed=0)
    env.coordinated.p.diff_trim = float(d)
    leg, qq, z, cur = [], [], [], []
    n = 0; info = {}
    for k in range(2600):
        _, _, term, trunc, info = env.step(np.zeros(6))
        n = k + 1
        if k >= 2000:
            leg.append(env.sim.leg_lengths().copy())
            qq.append(env.sim.joint_positions().copy())
            z.append(float(env.sim.data.qpos[2]))
            cur.append(float(info.get("wheel_current_left", 0.0)))
        if term or trunc: break
    env.close()
    if len(leg) < 40:
        print(f"{d:>+8.2f}  FELL n={n} term={info.get('termination_reason')}")
        continue
    L = np.mean(leg, axis=0); q = np.mean(qq, axis=0)
    base = np.asarray(env.params.robot.pd_neutral_joint_pos, dtype=np.float64)
    print(f"{d:>+8.2f} {L[0]:>8.4f} {L[1]:>8.4f} {q[0]:>+8.4f} {q[1]:>+8.4f} "
          f"{q[0]-base[0]:>+8.4f} {q[1]-base[1]:>+8.4f} {np.mean(z):>8.4f} "
          f"{np.sqrt(np.mean(np.square(cur))):>6.3f} {n:>5}")
