"""主动关节指令 → 稳态腿长 的数值标定表。

UZ-05 每条腿是闭环五连杆（3 个被动关节 + 2 个 connect 约束），腿长由约束
动力学决定，不能解析求解。这里直接扫「归一化腿动作 a_leg ∈ [-1,1]」→
稳态 (腿长, 机体高度, 髋-轮前后偏移)，为腿长控制器标定增益。

用法::

    python scripts/diag_leg_curve.py            # 打印标定表并缓存到 json
    python scripts/diag_leg_curve.py --steps 4000
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
OUT = Path(__file__).resolve().parent / "leg_curve.json"


def measure(offset_rad: float, steps: int, settle: int = 250) -> dict:
    """恒定关节偏置 offset_rad，跑到稳态后统计。"""
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    # 只给腿通道恒定偏置，轮通道交给内置平衡辅助（避免直接摔倒）
    action = np.zeros(6)
    action[2:6] = float(np.clip(offset_rad / POS_SCALE, -1.0, 1.0))
    leg_len, base_z, rel_x, pitch = [], [], [], []
    for k in range(steps):
        _, _, term, trunc, info = env.step(action)
        if k >= settle:
            leg_len.append(float(np.mean(env.sim.leg_lengths())))
            base_z.append(float(env.sim.data.qpos[2]))
            rel = env.sim.body_frame(
                env.sim.data.site_xpos[env.sim.wheel_sites[0]]
                - env.sim.data.site_xpos[env.sim.hip_sites[0]])
            rel_x.append(float(rel[0]))
            pitch.append(float(info.get("pitch", 0.0)))
        if term or trunc:
            break
    env.close()
    if not leg_len:
        return {"n": 0, "valid": False}
    return {
        "n": len(leg_len),
        "valid": len(leg_len) > 50,
        "leg_length": float(np.mean(leg_len)),
        "base_height": float(np.mean(base_z)),
        "rel_x": float(np.mean(rel_x)),
        "pitch_deg": float(np.degrees(np.mean(pitch))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    offsets = [round(v, 3) for v in np.linspace(-0.35, 0.35, 15)]
    rows = []
    print(f"{'offset_rad':>10} {'a_leg':>7} {'leg_len':>9} {'base_z':>8} {'rel_x':>8} "
          f"{'pitch°':>7} {'n':>5} ok")
    for off in offsets:
        d = measure(off, a.steps)
        rows.append({"offset_rad": off, "a_leg": off / POS_SCALE, **d})
        if d.get("valid"):
            print(f"{off:>+10.3f} {off / POS_SCALE:>+7.3f} {d['leg_length']:>9.4f} "
                  f"{d['base_height']:>8.4f} {d['rel_x']:>+8.4f} {d['pitch_deg']:>+7.2f} "
                  f"{d['n']:>5} yes")
        else:
            print(f"{off:>+10.3f} {off / POS_SCALE:>+7.3f} {'--':>9} {'--':>8} {'--':>8} "
                  f"{'--':>7} {d.get('n', 0):>5} NO (fell)")
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    valid = [r for r in rows if r.get("valid")]
    if len(valid) >= 2:
        L = np.asarray([r["leg_length"] for r in valid])
        O = np.asarray([r["offset_rad"] for r in valid])
        slope = np.polyfit(O, L, 1)
        print(f"\n有效区间: offset {O.min():+.3f}..{O.max():+.3f} rad, "
              f"腿长 {L.min():.4f}..{L.max():.4f} m")
        print(f"线性拟合: leg_len ≈ {slope[0]:+.4f} * offset + {slope[1]:.4f} "
              f"(dL/doffset = {slope[0]:+.4f} m/rad)")
        print(f"saved {OUT}")


if __name__ == "__main__":
    main()
