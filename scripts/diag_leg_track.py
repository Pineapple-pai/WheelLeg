"""验证腿长控制环：能命令的腿长范围、跟踪精度、对平衡/漂移的影响。

用法::

    python scripts/diag_leg_track.py --steps 1500
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(leg_ref: float, steps: int, seed: int = 0, dr: bool = False,
        settle: int = 400) -> dict:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=seed)
    # Keep the diagnostic in the same range as training.  Set the reference
    # through the public environment API so the current measured pose is not
    # silently replaced by a nominal-height command.
    env.coordinated.enable_leg_length(0.15, 0.27)
    env.set_leg_length_command(leg_ref)
    leg, z, pitch, drift, cur = [], [], [], [], []
    n = 0
    info: dict = {}
    for k in range(steps):
        _, _, term, trunc, info = env.step(np.zeros(6))
        n = k + 1
        if k >= settle:
            leg.append(float(np.mean(env.sim.leg_lengths())))
            z.append(float(env.sim.data.qpos[2]))
            pitch.append(float(info.get("pitch", 0.0)))
            drift.append(float(info.get("station_error", 0.0)))
            cur.append(float(info.get("wheel_current_left", 0.0)))
        if term or trunc:
            break
    env.close()
    if len(leg) < 50:
        return {"n": n, "valid": False, "term": info.get("termination_reason")}
    leg = np.asarray(leg); z = np.asarray(z); pitch = np.asarray(pitch)
    drift = np.asarray(drift)
    return {
        "n": n, "valid": True, "term": info.get("termination_reason"),
        "leg_ref": leg_ref,
        "leg_mean": float(leg.mean()), "leg_err_mm": float((leg.mean() - leg_ref) * 1000),
        "leg_std_mm": float(leg.std() * 1000),
        "base_z": float(z.mean()),
        "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(pitch ** 2)))),
        "drift_peak_cm": float(100 * np.abs(drift).max()),
        "drift_tail_cm": float(100 * np.abs(drift[-200:]).mean()),
        "current_rms_a": float(np.sqrt(np.mean(np.asarray(cur) ** 2))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--settle", type=int, default=400)
    a = ap.parse_args()
    refs = [0.15, 0.16, 0.18, 0.1841, 0.20, 0.22, 0.25, 0.27]
    print(f"{'L_ref':>7} {'L_actual':>9} {'err_mm':>8} {'std_mm':>7} {'base_z':>8} "
          f"{'pitch°':>7} {'drift_pk':>9} {'drift_tl':>9} {'I_rms':>6} {'n':>5}")
    for r in refs:
        d = run(r, a.steps, settle=a.settle)
        if not d["valid"]:
            print(f"{r:>7.3f} {'FELL':>9} n={d['n']} term={d.get('term')}")
            continue
        print(f"{r:>7.3f} {d['leg_mean']:>9.4f} {d['leg_err_mm']:>+8.1f} "
              f"{d['leg_std_mm']:>7.2f} {d['base_z']:>8.4f} {d['pitch_rms_deg']:>7.3f} "
              f"{d['drift_peak_cm']:>9.3f} {d['drift_tail_cm']:>9.3f} "
              f"{d['current_rms_a']:>6.2f} {d['n']:>5}")


if __name__ == "__main__":
    main()
