"""五连杆腿的正运动学标定：扫 (q2, q4) → 腿长 / 前后偏移 / 左右对称性。

UZ-05 每条腿是 2-DOF 闭环五连杆（2 个主动关节 + 3 个被动关节 + 2 个 connect
约束）。腿长**不能只靠一个关节**改变：单动 L2 会把轮子沿髋部圆弧摆动，
腿长基本不变（实测 trim 扫描腿长恒定 0.1842 m）。真正的伸缩需要 L2 / L4
反向协同（"剪叉"式）。

本脚本对每个 (q2, q4) 位形用带约束的静力求解得到稳态腿长，输出：
  * 可达腿长范围
  * dL/dq2、dL/dq4 的局部增益（用于写逆运动学）
  * 保持左右腿长一致的对称解
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

OUT = Path(__file__).resolve().parent / "leg_fk_table.json"


def solve_configuration(env: UZ05Env, q2: float, q4: float,
                        settle: int = 260, free_base: bool = True) -> dict:
    """把主动关节锁到 (q2,q4)（左右镜像），让被动关节收敛，返回稳态量。"""
    sim = env.sim
    q = np.array([q2, q4, -q2, -q4], dtype=np.float64)
    sim.data.qpos[sim.hip_qpos_adr] = q
    sim.data.qvel[:] = 0.0
    if free_base:
        sim.data.qpos[0] = 0.0
        sim.data.qpos[1] = 0.0
        sim.data.qpos[2] = 0.23825
        sim.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for _ in range(settle):
        err = q - sim.joint_positions()
        torque = 100.0 * err - 4.0 * sim.joint_velocities()
        np.clip(torque, -20.0, 20.0, out=torque)
        sim.data.ctrl[:4] = torque
        sim.data.ctrl[4:] = 0.0
        for _ in range(4):
            sim.step()
    lengths = sim.leg_lengths()
    rel = sim.body_frame(sim.data.site_xpos[sim.wheel_sites[0]]
                         - sim.data.site_xpos[sim.hip_sites[0]])
    return {
        "q2": float(q2), "q4": float(q4),
        "leg_left": float(lengths[0]), "leg_right": float(lengths[1]),
        "leg_mean": float(lengths.mean()),
        "leg_diff": float(lengths[0] - lengths[1]),
        "rel_x": float(rel[0]),
        "base_z": float(sim.data.qpos[2]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--settle", type=int, default=220)
    a = ap.parse_args()
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    base = np.asarray(env.params.robot.stand_joint_pos, dtype=np.float64)
    print(f"nominal q2={base[0]:+.4f} q4={base[1]:+.4f} "
          f"leg={env.sim.leg_lengths().mean():.4f}")

    q2s = np.linspace(base[0] - 0.7, base[0] + 1.4, a.grid)
    q4s = np.linspace(base[1] - 1.6, base[1] + 0.7, a.grid)
    rows = np.zeros((a.grid, a.grid))
    table = []
    for i, q2 in enumerate(q2s):
        for j, q4 in enumerate(q4s):
            d = solve_configuration(env, float(q2), float(q4), a.settle)
            rows[i, j] = d["leg_mean"]
            table.append(d)
    print(f"\n腿长表 (行=q2, 列=q4)，单位 mm：")
    header = "  q2\\q4 " + " ".join(f"{v:>6.3f}" for v in q4s)
    print(header)
    for i, q2 in enumerate(q2s):
        print(f" {q2:>+6.3f} " + " ".join(f"{1000 * v:>6.1f}" for v in rows[i]))
    valid = [d for d in table if d["leg_mean"] > 0.05]
    L = np.asarray([d["leg_mean"] for d in valid])
    print(f"\n可达腿长: {L.min():.4f} .. {L.max():.4f} m")
    symmetric = [d for d in valid if abs(d["leg_diff"]) < 0.02]
    if symmetric:
        Ls = np.asarray([d["leg_mean"] for d in symmetric])
        print(f"左右差 <20 mm 的位形: 腿长 {Ls.min():.4f} .. {Ls.max():.4f} m "
              f"({len(symmetric)}/{len(valid)})")
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"saved {OUT}")
    env.close()


if __name__ == "__main__":
    main()
