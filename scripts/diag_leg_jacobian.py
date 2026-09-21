"""直接测量腿通道的 2x2 耦合矩阵（在平衡控制器工作时）。

腿通道语义（由 `CoordinatedBalance.to_action` + `ActuatorBank` 共同定义）::

    q2 = neutral2 + (c + d)·0.35
    q4 = neutral4 + (c − d)·0.35

在协同控制器**工作**的前提下，给 (c, d) 各加一个恒定小偏置，测稳态的
腿长与支撑点前后偏移，用中心差分得到雅可比::

    [ΔL]   [ J_Lc  J_Ld ] [Δc]
    [Δx] = [ J_xc  J_xd ] [Δd]

这是写腿长环 + 解耦所需的**真实**增益（解析推导在奇异位形附近不可靠）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

OUT = Path(__file__).resolve().parent / "leg_jacobian.json"


def measure(env: UZ05Env, c: float, d: float, steps: int, settle: int) -> dict:
    """恒定 (c,d) 前馈 trim 下的稳态量（trim 直接进控制器，闭环完整）。"""
    env.coordinated.p.common_trim = float(c)
    env.coordinated.p.diff_trim = float(d)
    leg, relx, pitch, drift, legl, legr = [], [], [], [], [], []
    info: dict = {}
    n = 0
    for k in range(steps):
        _, _, term, trunc, info = env.step(np.zeros(6))
        n = k + 1
        if k >= settle:
            L = env.sim.leg_lengths()
            rel = env.sim.body_frame(
                env.sim.data.site_xpos[env.sim.wheel_sites[0]]
                - env.sim.data.site_xpos[env.sim.hip_sites[0]])
            leg.append(float(L.mean())); legl.append(float(L[0])); legr.append(float(L[1]))
            relx.append(float(rel[0]))
            pitch.append(float(info.get("pitch", 0.0)))
            drift.append(float(info.get("station_error", 0.0)))
        if term or trunc:
            break
    if len(leg) < 40:
        return {"valid": False, "n": n, "term": info.get("termination_reason")}
    return {
        "valid": True, "n": n,
        "leg": float(np.mean(leg)), "leg_diff": float(np.mean(legl) - np.mean(legr)),
        "rel_x": float(np.mean(relx)),
        "pitch_deg": float(np.degrees(np.mean(pitch))),
        "drift_cm": float(100 * np.abs(drift).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--settle", type=int, default=600)
    ap.add_argument("--h", type=float, default=0.15, help="差分步长（归一化动作）")
    a = ap.parse_args()

    def fresh():
        env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                      init_scale=0.0, lock_stand_leg_actions=False,
                      stand_leg_action_limit=1.0, coord_mix=1.0)
        env.params.domain_randomization.enabled = False
        env.stage = replace(env.stage, episode_steps=a.steps + 50)
        env.reset(seed=0)
        return env

    print("=== 中心点 (c=0,d=0) ===")
    env = fresh()
    base = measure(env, 0.0, 0.0, a.steps, a.settle)
    print("  base:", base)
    env.close()

    print(f"\n=== 差分（h={a.h}）===")
    table = {"base": base, "h": a.h}
    for name, (c, d) in (("c+", (a.h, 0.0)), ("c-", (-a.h, 0.0)),
                         ("d+", (0.0, a.h)), ("d-", (0.0, -a.h))):
        env = fresh()
        r = measure(env, c, d, a.steps, a.settle)
        env.close()
        table[name] = r
        if r.get("valid"):
            print(f"  {name}: leg={r['leg']:.4f} rel_x={r['rel_x']:+.4f} "
                  f"pitch={r['pitch_deg']:+.2f}° drift={r['drift_cm']:.2f}cm")
        else:
            print(f"  {name}: FELL n={r['n']} term={r.get('term')}")

    if all(table[k].get("valid") for k in ("c+", "c-", "d+", "d-")):
        JLc = (table["c+"]["leg"] - table["c-"]["leg"]) / (2 * a.h)
        JLd = (table["d+"]["leg"] - table["d-"]["leg"]) / (2 * a.h)
        Jxc = (table["c+"]["rel_x"] - table["c-"]["rel_x"]) / (2 * a.h)
        Jxd = (table["d+"]["rel_x"] - table["d-"]["rel_x"]) / (2 * a.h)
        J = np.array([[JLc, JLd], [Jxc, Jxd]])
        table["J"] = J.tolist()
        table["det"] = float(np.linalg.det(J))
        print(f"\n雅可比 J = [[dL/dc, dL/dd], [dx/dc, dx/dd]]")
        print(f"  dL/dc = {JLc:+.4f} m/动作单位   dL/dd = {JLd:+.4f}")
        print(f"  dx/dc = {Jxc:+.4f} m/动作单位   dx/dd = {Jxd:+.4f}")
        print(f"  det(J) = {np.linalg.det(J):+.6f}")
        if abs(np.linalg.det(J)) > 1e-6:
            Jinv = np.linalg.inv(J)
            print(f"  J^-1 = [[{Jinv[0,0]:+.4f}, {Jinv[0,1]:+.4f}],"
                  f" [{Jinv[1,0]:+.4f}, {Jinv[1,1]:+.4f}]]")
            # 解读：想只伸长腿、不动支撑点，需要 (c,d) 的比例
            print(f"  纯伸长方向 (Δx=0): Δc:Δd = {Jinv[0,0]:+.4f} : {Jinv[1,0]:+.4f}")
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
