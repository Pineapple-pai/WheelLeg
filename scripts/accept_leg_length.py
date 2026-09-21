"""腿长变化验收：在保持平衡、不产生漂移的前提下跟踪 0.15~0.27 m 内部腿长。

验收口径（对应目标"加入腿长变化训练，范围 0.15-0.35 左右，同时不能失去平衡
能力和造成漂移"）：

* 腿长跟踪误差
* 全程漂移峰值 / 末段平均漂移
* pitch RMS / 峰值
* 存活（跑满整集）
* 左右腿长一致性

代码内部腿长加轮半径约 55 mm 才是车底离地高度。

用法::

    python scripts/accept_leg_length.py
    python scripts/accept_leg_length.py --targets 0.15 0.20 0.25 0.27 --steps 3000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(leg_ref: float | None, steps: int, seed: int = 0, dr: bool = False,
        settle: int = 2000, lo: float = 0.15, hi: float = 0.270) -> dict:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=seed)
    if leg_ref is not None:
        env.coordinated.enable_leg_length(lo, hi)
        env.set_leg_length_command(leg_ref)
    leg, legl, legr, z, pitch, drift, cur = [], [], [], [], [], [], []
    info: dict = {}
    n = 0
    for k in range(steps):
        _, _, term, trunc, info = env.step(np.zeros(6))
        n = k + 1
        if k >= settle:
            L = env.sim.leg_lengths()
            leg.append(float(L.mean())); legl.append(float(L[0])); legr.append(float(L[1]))
            z.append(float(env.sim.data.qpos[2]))
            pitch.append(float(info.get("pitch", 0.0)))
            drift.append(float(info.get("station_error", 0.0)))
            cur.append(float(info.get("wheel_current_left", 0.0)))
        if term or trunc:
            break
    env.close()
    if len(leg) < 40:
        return {"valid": False, "n": n, "term": info.get("termination_reason")}
    leg = np.asarray(leg); pitch = np.asarray(pitch); drift = np.asarray(drift)
    return {
        "valid": True, "n": n, "term": info.get("termination_reason"),
        "leg_ref": leg_ref, "leg_mean": float(leg.mean()),
        "leg_err_mm": float((leg.mean() - leg_ref) * 1000) if leg_ref is not None else 0.0,
        "leg_ripple_mm": float(leg.std() * 1000),
        "leg_lr_mm": float((np.mean(legl) - np.mean(legr)) * 1000),
        "base_z": float(np.mean(z)),
        "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(pitch ** 2)))),
        "pitch_peak_deg": float(np.degrees(np.abs(pitch).max())),
        "drift_peak_cm": float(100 * np.abs(drift).max()),
        "drift_tail_cm": float(100 * np.abs(drift[-200:]).mean()),
        "drift_final_cm": float(100 * drift[-1]),
        "current_rms_a": float(np.sqrt(np.mean(np.asarray(cur) ** 2))),
    }


def show(tag: str, d: dict) -> None:
    if not d["valid"]:
        print(f"[{tag}] FELL n={d['n']} term={d.get('term')}")
        return
    ref = d["leg_ref"]
    ref_txt = f"{ref:.4f}" if ref is not None else "--"
    err_txt = f"{d['leg_err_mm']:+.1f}" if ref is not None else "--"
    print(f"[{tag}] leg={d['leg_mean']:.4f} (ref {ref_txt}) "
          f"err={err_txt}mm ripple={d['leg_ripple_mm']:.2f}mm "
          f"L/R={d['leg_lr_mm']:+.2f}mm base_z={d['base_z']:.4f}")
    print(f"[{tag}] pitch_rms={d['pitch_rms_deg']:.3f}° "
          f"pitch_peak={d['pitch_peak_deg']:.3f}° "
          f"drift_peak={d['drift_peak_cm']:.3f}cm drift_tail={d['drift_tail_cm']:.3f}cm "
          f"drift_final={d['drift_final_cm']:+.3f}cm I_rms={d['current_rms_a']:.3f}A "
          f"n={d['n']}({d['term']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--settle", type=int, default=2000)
    ap.add_argument("--targets", type=float, nargs="*",
                    default=[0.150, 0.180, 0.210, 0.240, 0.270])
    a = ap.parse_args()

    print("=== 腿长环关闭（回归检查：必须与修复前一致）===")
    show("nominal", run(None, a.steps, settle=a.settle))
    show("nominal+DR", run(None, a.steps, settle=a.settle, dr=True))

    print(f"\n=== 腿长跟踪（内部腿长范围 0.15~0.27 m，{a.steps} 步）===")
    ok = 0
    for t in a.targets:
        d = run(t, a.steps, settle=a.settle)
        show(f"L={t:.3f}", d)
        if (d["valid"] and d["n"] >= a.steps
                and abs(d["leg_err_mm"]) <= 15.0
                and d["drift_tail_cm"] < 5.0):
            ok += 1
    print(f"\n通过 {ok}/{len(a.targets)}（判据：跑满整集 + "
          "|腿长误差| <= 15 mm + 末段漂移 < 5 cm）")


if __name__ == "__main__":
    main()
