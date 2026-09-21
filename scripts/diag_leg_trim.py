"""腿长标定：恒定腿偏置 trim → 稳态腿长（在有平衡控制的前提下）。

这一步是腿长控制环的前提：先知道「关节偏置 → 腿长」的静态增益，才能写
高度/腿长环。为避免机器人摔倒，用协同平衡控制器稳住姿态，只在腿通道上叠加
一个**恒定 trim**（直接注入 data.ctrl，绕过控制器输出）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402
from uz05.spec import ACTION_SPEC  # noqa: E402

POS_SCALE = {n: s for n, _, s in ACTION_SPEC}["hip_position_offset"]
OUT = Path(__file__).resolve().parent / "leg_trim_curve.json"


def measure(trim_rad: float, steps: int, settle: int = 500) -> dict:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    action = np.zeros(6)
    leg_len, base_z, rel_x, pitch, sat = [], [], [], [], []
    n = 0
    for k in range(steps):
        _, _, term, trunc, info = env.step(action)
        # 在控制器下发之后叠加恒定腿 trim
        base_q = np.asarray(env.params.robot.pd_neutral_joint_pos, dtype=np.float64)
        trim = float(np.clip((info["coord_leg_offset"] + trim_rad) / POS_SCALE, -1, 1))
        env.sim.data.qfrc_applied  # noqa: B018  (保持可读性)
        ctrl = env.sim.data.ctrl
        # 重新按 PD 计算：等价于把腿动作改成 trim
        q_target = base_q + trim * POS_SCALE
        torque = (100.0 * (q_target - env.sim.joint_positions())
                  - 4.0 * env.sim.joint_velocities())
        np.clip(torque, -20.0, 20.0, out=torque)
        ctrl[:4] = torque
        n = k + 1
        if k >= settle:
            leg_len.append(float(np.mean(env.sim.leg_lengths())))
            base_z.append(float(env.sim.data.qpos[2]))
            rel = env.sim.body_frame(
                env.sim.data.xpos[env.sim.wheel_bodies[0]] - env.sim.data.qpos[:3])
            rel_x.append(float(rel[0]))
            pitch.append(float(info.get("pitch", 0.0)))
            sat.append(abs(info["coord_leg_offset"]) >= 0.3495)
        if term or trunc:
            break
    env.close()
    if len(leg_len) < 50:
        return {"n": n, "valid": False}
    return {
        "n": n, "valid": True,
        "leg_length": float(np.mean(leg_len)),
        "base_height": float(np.mean(base_z)),
        "rel_x": float(np.mean(rel_x)),
        "pitch_deg": float(np.degrees(np.mean(pitch))),
        "sat_frac": float(np.mean(sat)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1600)
    a = ap.parse_args()
    trims = [round(v, 3) for v in np.linspace(-0.35, 0.35, 15)]
    rows = []
    print(f"{'trim_rad':>9} {'leg_len':>9} {'base_z':>8} {'pitch°':>7} {'sat':>6} {'n':>5}")
    for t in trims:
        d = measure(t, a.steps)
        rows.append({"trim_rad": t, **d})
        if d.get("valid"):
            print(f"{t:>+9.3f} {d['leg_length']:>9.4f} {d['base_height']:>8.4f} "
                  f"{d['pitch_deg']:>+7.2f} {d['sat_frac']:>6.3f} {d['n']:>5}")
        else:
            print(f"{t:>+9.3f} {'--':>9} {'--':>8} {'--':>7} {'--':>6} {d.get('n', 0):>5} FELL")
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    valid = [r for r in rows if r.get("valid")]
    if len(valid) >= 3:
        O = np.asarray([r["trim_rad"] for r in valid])
        L = np.asarray([r["leg_length"] for r in valid])
        slope, intercept = np.polyfit(O, L, 1)
        print(f"\n可达腿长 {L.min():.4f} .. {L.max():.4f} m "
              f"(trim {O.min():+.3f}..{O.max():+.3f} rad)")
        print(f"静态增益 dL/dtrim = {slope:+.4f} m/rad "
              f"(腿长 {intercept:.4f} m @ trim=0)")
        resid = L - (slope * O + intercept)
        print(f"线性拟合残差 RMS = {np.sqrt(np.mean(resid ** 2)) * 1000:.2f} mm")
        print(f"saved {OUT}")


if __name__ == "__main__":
    main()
