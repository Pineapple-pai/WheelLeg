"""腿长工作空间标定：左右腿用**相同**关节值（实测唯一保持左右对称的约定）。

输出 (q2, q4) → 腿长 / 前后偏移 的二维表，并做局部线性化，供逆运动学使用。

用法::

    python scripts/diag_leg_ws.py --grid 17 --steps 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

OUT = Path(__file__).resolve().parent / "leg_ws.json"
SETTLE, SUBSTEPS = 50, 4


def hold(env: UZ05Env, q: np.ndarray, settle: int = SETTLE) -> None:
    sim = env.sim
    for _ in range(settle):
        err = q - sim.joint_positions()
        torque = np.clip(200.0 * err - 6.0 * sim.joint_velocities(), -20.0, 20.0)
        sim.data.ctrl[:4] = torque
        sim.data.ctrl[4:] = 0.0
        for _ in range(SUBSTEPS):
            sim.step()


def probe(env: UZ05Env, q2: float, q4: float) -> dict:
    q = np.array([q2, q4, q2, q4], dtype=np.float64)   # ★ 左右同值
    hold(env, q)
    L = env.sim.leg_lengths()
    rel = env.sim.body_frame(env.sim.data.site_xpos[env.sim.wheel_sites[0]]
                             - env.sim.data.site_xpos[env.sim.hip_sites[0]])
    return {"q2": float(q2), "q4": float(q4),
            "leg_left": float(L[0]), "leg_right": float(L[1]),
            "leg_mean": float(L.mean()), "leg_diff": float(L[0] - L[1]),
            "rel_x": float(rel[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=19)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    base = np.asarray(env.params.robot.stand_joint_pos, dtype=np.float64)
    lo2, hi2 = base[0] - 0.45, base[0] + 0.45
    lo4, hi4 = base[1] - 0.45, base[1] + 0.45
    q2s = np.linspace(lo2, hi2, a.grid)
    q4s = np.linspace(lo4, hi4, a.grid)
    table = []
    Lmat = np.zeros((a.grid, a.grid))
    for i, q2 in enumerate(q2s):
        for j, q4 in enumerate(q4s):
            d = probe(env, float(q2), float(q4))
            table.append(d)
            Lmat[i, j] = d["leg_mean"]
    print("腿长表 单位 mm（行=q2, 列=q4）")
    print("  q2\\q4 " + " ".join(f"{v:>6.2f}" for v in q4s))
    for i, q2 in enumerate(q2s):
        print(f" {q2:>+6.2f} " + " ".join(f"{1000 * v:>6.1f}" for v in Lmat[i]))
    valid = [d for d in table if abs(d["leg_diff"]) < 0.01]
    L = np.asarray([d["leg_mean"] for d in valid])
    print(f"\n左右差 <10 mm 的位形: 腿长 {L.min():.4f} .. {L.max():.4f} m "
          f"({len(valid)}/{len(table)})")
    # 在线性区里做最小二乘：ΔL ≈ [g2, g4]·[Δq2, Δq4]
    A = np.asarray([[d["q2"] - base[0], d["q4"] - base[1]] for d in valid])
    y = np.asarray([d["leg_mean"] for d in valid]) - 0.1841
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    print(f"线性增益: dL/dq2 = {coef[0]:+.4f} m/rad, dL/dq4 = {coef[1]:+.4f} m/rad")
    print(f"残差 RMS = {np.sqrt(np.mean((y - pred) ** 2)) * 1000:.2f} mm")
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"saved {OUT}")
    env.close()


if __name__ == "__main__":
    main()
