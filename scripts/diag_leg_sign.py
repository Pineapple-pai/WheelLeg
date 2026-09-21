"""腿长开环灵敏度：固定差模偏置 d → 稳态腿长（在平衡控制器工作下）。

目的只有一个：确定 d 的**符号**，以及大致增益。开环逆解在奇异位形附近不可靠，
所以这里只做"符号正确性 + 单调性"的标定。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(d: float, steps: int = 1400, settle: int = 700, seed: int = 0,
        height_diff: float = 0.0) -> dict:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=seed)
    env.coordinated.p.diff_trim = float(d)
    leg, pitch, drift, legl, legr = [], [], [], [], []
    info: dict = {}
    n = 0
    for k in range(steps):
        _, _, term, trunc, info = env.step(np.zeros(6))
        n = k + 1
        if k >= settle:
            L = env.sim.leg_lengths()
            leg.append(float(L.mean())); legl.append(float(L[0])); legr.append(float(L[1]))
            pitch.append(float(info.get("pitch", 0.0)))
            drift.append(float(info.get("station_error", 0.0)))
        if term or trunc:
            break
    env.close()
    if len(leg) < 40:
        return {"valid": False, "n": n, "term": info.get("termination_reason")}
    return {"valid": True, "n": n, "leg": float(np.mean(leg)),
            "leg_diff": float(np.mean(legl) - np.mean(legr)),
            "pitch_deg": float(np.degrees(np.mean(pitch))),
            "drift_cm": float(100 * np.abs(drift).max())}


if __name__ == "__main__":
    print(f"{'d_trim':>8} {'leg':>9} {'左右差':>9} {'pitch°':>8} {'drift_pk':>9} {'n':>5}")
    for d in (-0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5):
        r = run(d)
        if not r["valid"]:
            print(f"{d:>+8.2f} {'FELL':>9} {'':>9} {'':>8} {'':>9} {r['n']:>5} {r.get('term')}")
            continue
        print(f"{d:>+8.2f} {r['leg']:>9.4f} {r['leg_diff'] * 1000:>+8.1f}m {r['pitch_deg']:>+8.2f} "
              f"{r['drift_cm']:>9.3f} {r['n']:>5}")
