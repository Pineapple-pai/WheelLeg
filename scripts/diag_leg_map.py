"""腿长标定（增量延拓法）。

手工推导五连杆 FK 容易在坐标系上出错；这里用**增量延拓**绕过解析推导：
从额定位形出发，每步只把两个主动关节移动一个很小的量，让被动关节与
connect 约束重新收敛，再读稳态腿长。这样始终停留在同一个机构分支上，
不会跳到畸变位形。

用法::

    python scripts/diag_leg_map.py                  # 生成标定表
    python scripts/diag_leg_map.py --plot           # 额外打印 ASCII 曲线
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

OUT = Path(__file__).resolve().parent / "leg_map.json"
SETTLE = 60          # 每个增量点让约束收敛的步数
SUBSTEPS = 4


def hold_joints(env: UZ05Env, q: np.ndarray, settle: int = SETTLE) -> None:
    """用强 PD 把 4 个主动关节锁到 q，跑 settle 个控制步让约束收敛。"""
    sim = env.sim
    for _ in range(settle):
        err = q - sim.joint_positions()
        torque = 200.0 * err - 6.0 * sim.joint_velocities()
        np.clip(torque, -20.0, 20.0, out=torque)
        sim.data.ctrl[:4] = torque
        sim.data.ctrl[4:] = 0.0
        for _ in range(SUBSTEPS):
            sim.step()


def probe(env: UZ05Env, q2: float, q4: float) -> dict:
    q = np.array([q2, q4, -q2, -q4], dtype=np.float64)
    hold_joints(env, q)
    lengths = env.sim.leg_lengths()
    rel = env.sim.body_frame(env.sim.data.site_xpos[env.sim.wheel_sites[0]]
                             - env.sim.data.site_xpos[env.sim.hip_sites[0]])
    return {"q2": float(q2), "q4": float(q4),
            "leg_left": float(lengths[0]), "leg_right": float(lengths[1]),
            "leg_mean": float(lengths.mean()),
            "leg_diff": float(lengths[0] - lengths[1]),
            "rel_x": float(rel[0]), "rel_z": float(rel[2])}


def walk(env: UZ05Env, base: np.ndarray, steps: int, dq: float,
         direction: tuple[float, float]) -> list[dict]:
    """从 base 沿 (dq2, dq4) 方向一步步走，每步一小段。"""
    out = []
    q2, q4 = float(base[0]), float(base[1])
    for _ in range(steps):
        q2 += dq * direction[0]
        q4 += dq * direction[1]
        out.append(probe(env, q2, q4))
    return out


def fresh_env() -> UZ05Env:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--dq", type=float, default=0.03)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    env = fresh_env()
    base = np.asarray(env.params.robot.stand_joint_pos, dtype=np.float64)
    print(f"nominal q2={base[0]:+.4f} q4={base[1]:+.4f} "
          f"leg={env.sim.leg_lengths().mean():.4f}")

    # 4 个方向：q2 ±、 q4 ±，以及两个"剪叉"方向
    dirs = {
        "q2+": (1.0, 0.0), "q2-": (-1.0, 0.0),
        "q4+": (0.0, 1.0), "q4-": (0.0, -1.0),
        "scissor_L(+,-)": (1.0, -1.0), "scissor_R(-,+)": (-1.0, 1.0),
        "same(+,+)": (1.0, 1.0), "same(-,-)": (-1.0, -1.0),
    }
    results = {}
    for name, d in dirs.items():
        env.reset(seed=0)
        hold_joints(env, base, settle=120)
        rows = walk(env, base, a.steps, a.dq, d)
        L = np.asarray([r["leg_mean"] for r in rows])
        dx = np.asarray([r["rel_x"] for r in rows])
        results[name] = rows
        print(f"\n-- {name} (dq={a.dq}) --")
        print(f"   腿长 {L.min():.4f} .. {L.max():.4f} m   前后偏移 {dx.min():+.4f} .. {dx.max():+.4f}")
        for i in range(0, len(rows), max(1, len(rows) // 6)):
            r = rows[i]
            print(f"   q2={r['q2']:+.3f} q4={r['q4']:+.3f} legL={r['leg_left']:.4f} "
                  f"legR={r['leg_right']:.4f} diff={r['leg_diff']:+.4f} "
                  f"rel_x={r['rel_x']:+.4f}")
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved {OUT}")

    print("\n=== 汇总：哪个方向能改腿长、左右是否对称 ===")
    for name, rows in results.items():
        L = np.asarray([r["leg_mean"] for r in rows])
        diff = np.abs(np.asarray([r["leg_diff"] for r in rows]))
        print(f"  {name:16s} ΔL={L.max() - L.min():+.4f} m  "
              f"max|左右差|={diff.max() * 1000:6.1f} mm  "
              f"L@末={L[-1]:.4f}")
    env.close()


if __name__ == "__main__":
    main()
