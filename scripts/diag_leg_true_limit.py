"""腿长真实上限：绕过动作限幅，直接看机构能伸多长。

为什么需要复查
--------------
`CoordinatedBalance.to_action` 把归一化动作裁剪到 ±1，而差模的缩放是 0.35 rad，
所以**动作通道最多只能给出 ±0.35 rad 的差模**（= 0.35 × 0.26 ≈ 91 mm 腿长）。
之前的 `diag_leg_ceiling.py` 用 `p.diff_trim` 时同样经过这条裁剪，
所以"d>0.40 腿长恒定"可能只是**动作饱和**，不是机构几何饱和。

这里直接写 `sim.data.ctrl`，把 4 个关节 PD 目标设成任意大，观察真正的机构上限。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import replace  # noqa: E402

from uz05.env import UZ05Env  # noqa: E402

POS = 0.35
SETTLE, SUB = 60, 4


def run(diff_rad: float, steps: int = 1800, settle: int = 900) -> dict:
    """差模直接给到关节（rad），不经过动作裁剪；共模由控制器负责。"""
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=0)
    env.coordinated.disable_leg_length()
    base = np.asarray(env.params.robot.pd_neutral_joint_pos, dtype=np.float64)
    leg, q2, q4, pitch, drift = [], [], [], [], []
    n = 0
    info: dict = {}
    for k in range(steps):
        _, _, term, trunc, info = env.step(np.zeros(6))
        # 叠加：控制器共模 + 指定差模（直接进 PD 目标，不裁剪）
        common = float(env.coordinated.last["leg_offset"])
        q_target = base.copy()
        q_target[0] += (common * 1.0 + diff_rad * 1.0)
        q_target[1] += (common * 1.0 - diff_rad * 1.0)
        q_target[2] += (common * 1.0 + diff_rad * 1.0)
        q_target[3] += (common * 1.0 - diff_rad * 1.0)
        torque = np.clip(100.0 * (q_target - env.sim.joint_positions())
                         - 4.0 * env.sim.joint_velocities(), -20.0, 20.0)
        env.sim.data.ctrl[:4] = torque
        n = k + 1
        if k >= settle:
            leg.append(float(np.mean(env.sim.leg_lengths())))
            q = env.sim.joint_positions()
            q2.append(float(q[0])); q4.append(float(q[1]))
            pitch.append(float(info.get("pitch", 0.0)))
            drift.append(float(info.get("station_error", 0.0)))
        if term or trunc:
            break
    env.close()
    if len(leg) < 40:
        return {"valid": False, "n": n, "term": info.get("termination_reason")}
    return {"valid": True, "n": n, "leg": float(np.mean(leg)),
            "q2": float(np.mean(q2)), "q4": float(np.mean(q4)),
            "pitch_deg": float(np.degrees(np.max(np.abs(pitch)))),
            "drift_cm": float(100 * np.max(np.abs(drift)))}


if __name__ == "__main__":
    print("共模由协同控制器给；差模直接写关节 PD 目标（绕过动作裁剪）")
    print(f"{'diff(rad)':>10} {'a_equiv':>8} {'leg':>9} {'base_z':>8} {'q2':>8} "
          f"{'q4':>8} {'Δ':>7} {'pitch°':>8} {'drift_pk':>9} {'n':>5}")
    for d in (0.0, 0.1, 0.2, 0.3, 0.35, 0.45, 0.6, 0.8, 1.0, 1.3):
        r = run(d)
        if not r["valid"]:
            print(f"{d:>10.2f} {d / POS:>8.2f} {'FELL':>9} n={r['n']} {r.get('term')}")
            continue
        print(f"{d:>10.2f} {d / POS:>8.2f} {r['leg']:>9.4f} {r['leg'] + 0.0541:>8.4f} "
              f"{r['q2']:>+8.4f} {r['q4']:>+8.4f} {r['q2'] - r['q4']:>+7.4f} "
              f"{r['pitch_deg']:>8.3f} {r['drift_cm']:>9.3f} {r['n']:>5}")
