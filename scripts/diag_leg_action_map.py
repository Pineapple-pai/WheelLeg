"""沿**真实动作接口**测腿长：action[2..5] → 腿长。

与 `diag_leg_load_limit.py`（直接写关节目标）对照，定位到底是
"动作缩放/裁剪" 还是 "机构/负载" 在限制腿长。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(a_common: float, a_diff: float, steps: int = 2000, settle: int = 900,
        gravity: float = 9.81) -> dict:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=0.0)
    env.params.domain_randomization.enabled = False
    env.sim.model.opt.gravity[:] = [0.0, 0.0, -gravity]
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=0)
    a = np.zeros(6)
    a[2] = a_common      # 左腿共模
    a[3] = a_diff        # 左腿差模
    a[4] = a_common      # 右腿共模
    a[5] = a_diff        # 右腿差模
    leg, q, z = [], [], []
    n = 0
    info: dict = {}
    for k in range(steps):
        _, _, term, trunc, info = env.step(a)
        n = k + 1
        if k >= settle:
            L = env.sim.leg_lengths()
            leg.append(L.copy()); q.append(env.sim.joint_positions().copy())
            z.append(float(env.sim.data.qpos[2]))
        if term or trunc:
            break
    env.close()
    if len(leg) < 40:
        return {"valid": False, "n": n, "term": info.get("termination_reason")}
    L = np.mean(leg, axis=0); qq = np.mean(q, axis=0)
    return {"valid": True, "n": n, "leg": float(L.mean()),
            "lr_mm": float((L[0] - L[1]) * 1000), "q2": float(qq[0]),
            "q4": float(qq[1]), "z": float(np.mean(z))}


if __name__ == "__main__":
    print("gravity=9.81（带载）")
    print(f"{'a_common':>9} {'a_diff':>7} {'leg':>9} {'L/R(mm)':>9} {'q2':>8} {'q4':>8} {'z':>8}")
    for diff in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = run(0.0, diff)
        if not r["valid"]:
            print(f"{0.0:>9.2f} {diff:>7.2f} FELL n={r['n']} {r.get('term')}")
        else:
            print(f"{0.0:>9.2f} {diff:>7.2f} {r['leg']:>9.4f} {r['lr_mm']:>+9.2f} "
                  f"{r['q2']:>+8.4f} {r['q4']:>+8.4f} {r['z']:>8.4f}")
    print("\ngravity=0（无载）")
    for diff in (0.5, 1.0):
        r = run(0.0, diff, gravity=0.0)
        if not r["valid"]:
            print(f"{0.0:>9.2f} {diff:>7.2f} FELL n={r['n']} {r.get('term')}")
        else:
            print(f"{0.0:>9.2f} {diff:>7.2f} {r['leg']:>9.4f} {r['lr_mm']:>+9.2f} "
                  f"{r['q2']:>+8.4f} {r['q4']:>+8.4f} {r['z']:>8.4f}")
