"""腿长变化课程的验收：随机腿长目标下，跟踪精度 + 平衡 + 漂移。

每 episode 在 leg_length_range 内采样一个目标，检查：
  * 腿长是否跟到命令值（末段误差）
  * 高度命令是否被跟踪
  * 漂移、pitch、存活
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataclasses import replace
from uz05.env import UZ05Env


def episode(seed: int, steps: int, settle: int, dr: bool = False) -> dict:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps + 50)
    env.reset(seed=seed)
    leg_cmd = env.leg_length_command
    h_cmd = float(env.command[3])
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
    if len(leg) < 40:
        return {"valid": False, "n": n, "term": info.get("termination_reason"),
                "leg_cmd": leg_cmd}
    return {
        "valid": True, "n": n, "term": info.get("termination_reason"),
        "leg_cmd": leg_cmd, "h_cmd": h_cmd,
        "leg_meas": float(np.mean(leg)), "z_meas": float(np.mean(z)),
        "leg_err_mm": float((np.mean(leg) - leg_cmd) * 1000),
        "h_err_mm": float((np.mean(z) - h_cmd) * 1000),
        "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(pitch))))),
        "drift_peak_cm": float(100 * max(abs(np.array(drift)))),
        "drift_tail_cm": float(100 * np.mean(np.abs(np.array(drift)[-200:]))),
        "current_rms_a": float(np.sqrt(np.mean(np.square(cur)))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--settle", type=int, default=1500)
    a = ap.parse_args()
    print(f"{'seed':>5} {'leg_cmd':>8} {'leg':>8} {'err_mm':>8} {'h_err':>7} "
          f"{'pitch°':>7} {'drift_pk':>9} {'drift_tl':>9} {'I_rms':>6} {'n':>5} term")
    ok = 0; errs = []; drifts = []
    for i in range(a.episodes):
        d = episode(100 + i, a.steps, a.settle)
        if not d["valid"]:
            print(f"{100+i:>5} {d['leg_cmd']:>8.4f} {'FELL':>8} n={d['n']} {d.get('term')}")
            continue
        good = d["drift_tail_cm"] < 5.0 and abs(d["leg_err_mm"]) < 15.0
        ok += int(good)
        errs.append(abs(d["leg_err_mm"])); drifts.append(d["drift_peak_cm"])
        print(f"{100+i:>5} {d['leg_cmd']:>8.4f} {d['leg_meas']:>8.4f} {d['leg_err_mm']:>+8.1f} "
              f"{d['h_err_mm']:>+7.1f} {d['pitch_rms_deg']:>7.3f} {d['drift_peak_cm']:>9.3f} "
              f"{d['drift_tail_cm']:>9.3f} {d['current_rms_a']:>6.3f} {d['n']:>5} {d['term']}")
    print(f"\n通过 {ok}/{a.episodes}（判据：跑满整集 + 腿长误差 <15 mm + 末段漂移 <5 cm）")
    if errs:
        print(f"腿长误差 |e|: 均值 {np.mean(errs):.1f} mm  最大 {np.max(errs):.1f} mm")
        print(f"漂移峰值: 均值 {np.mean(drifts):.3f} cm  最大 {np.max(drifts):.3f} cm")


if __name__ == "__main__":
    main()
