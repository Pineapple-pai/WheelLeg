"""新差模缩放下的腿长可达范围（走真实动作接口 + 控制器全权）。

用法::  python scripts/diag_leg_range_check.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(target: float, steps: int = 3000, settle: int = 1800,
        lo: float = 0.12, hi: float = 0.35) -> dict:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=0)
    env.coordinated.enable_leg_length(lo, hi, ref=target)
    leg, z, pitch, drift, q = [], [], [], [], []
    n = 0
    info: dict = {}
    for k in range(steps):
        _, _, term, trunc, info = env.step(np.zeros(6))
        n = k + 1
        if k >= settle:
            L = env.sim.leg_lengths()
            leg.append(L.copy()); z.append(float(env.sim.data.qpos[2]))
            pitch.append(float(info.get("pitch", 0.0)))
            drift.append(float(info.get("station_error", 0.0)))
            q.append(env.sim.joint_positions().copy())
        if term or trunc:
            break
    env.close()
    if len(leg) < 40:
        return {"valid": False, "n": n, "term": info.get("termination_reason")}
    L = np.mean(leg, axis=0)
    return {"valid": True, "n": n, "leg": float(L.mean()),
            "lr_mm": float((L[0] - L[1]) * 1000), "z": float(np.mean(z)),
            "pitch_deg": float(np.degrees(np.max(np.abs(pitch)))),
            "drift_cm": float(100 * np.max(np.abs(drift))),
            "q2": float(np.mean([x[0] for x in q])),
            "q4": float(np.mean([x[1] for x in q]))}


if __name__ == "__main__":
    print(f"{'target':>7} {'leg':>9} {'err_mm':>8} {'L/R(mm)':>9} {'z':>8} "
          f"{'pitch_pk':>9} {'drift_pk':>9} {'q2':>8} {'q4':>8} {'n':>5}")
    for t in (0.150, 0.180, 0.195, 0.210, 0.240, 0.270, 0.285, 0.300, 0.310, 0.320):
        r = run(t)
        if not r["valid"]:
            print(f"{t:>7.3f} FELL n={r['n']} {r.get('term')}")
        else:
            print(f"{t:>7.3f} {r['leg']:>9.4f} {(r['leg'] - t) * 1000:>+8.1f} "
                  f"{r['lr_mm']:>+9.2f} {r['z']:>8.4f} {r['pitch_deg']:>9.3f} "
                  f"{r['drift_cm']:>9.3f} {r['q2']:>+8.4f} {r['q4']:>+8.4f} {r['n']:>5}")
