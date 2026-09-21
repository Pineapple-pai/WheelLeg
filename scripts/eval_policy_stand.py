"""评估策略 checkpoint 在零指令站定下的表现（可叠加/不叠加协同控制器）。

用法::

    # 纯策略（协同关闭）
    python scripts/eval_policy_stand.py --checkpoint checkpoints/<v>/checkpoint --mix 0

    # 策略残差（协同全权，默认残差预算）
    python scripts/eval_policy_stand.py --checkpoint checkpoints/<v>/checkpoint --mix 1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stable_baselines3 import PPO  # noqa: E402

from train_uz05 import AsymmetricActorCriticPolicy  # noqa: E402
from uz05.env import UZ05Env  # noqa: E402


def rollout(model, steps=1500, seed=0, init_scale=1.0, dr=False, mix=0.0,
            residual=(0.05, 0.02, 0.05), stand_level=2):
    env = UZ05Env(stage="stand", stand_level=stand_level, assist_scale=0.0, seed=seed,
                  init_scale=init_scale, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=mix,
                  coord_residual_scale=residual)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    obs, _ = env.reset(seed=seed)
    th, xs, ys, I, n = [], [], [], [], 0
    info: dict = {}
    for k in range(steps):
        action = np.zeros(6) if model is None else model.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action)
        q = env.sim.base_quat
        rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
        th.append(float(rpy[1]))
        xs.append(float(env.sim.data.qpos[0] - env.nominal_xy[0]))
        ys.append(float(env.sim.data.qpos[1]))
        I.append(float(info["wheel_current_left"]))
        n = k + 1
        if term or trunc:
            break
    env.close()
    th = np.asarray(th); xs = np.asarray(xs); I = np.asarray(I); ys = np.asarray(ys)
    tail = slice(max(0, n - 200), n)
    return {
        "n": n, "survived": n >= steps, "term": info.get("termination_reason", "?"),
        "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(th ** 2)))),
        "pitch_ptp_deg": float(np.degrees(th.max() - th.min())),
        "pitch_rate_rms": float(np.sqrt(np.mean(info.get("pitch_rate", 0.0) ** 2))),
        "drift_peak_cm": float(100 * np.abs(xs).max()),
        "drift_tail_cm": float(100 * np.mean(np.abs(xs[tail]))),
        "drift_final_cm": float(100 * xs[n - 1]),
        "yaw_peak_cm": float(100 * np.abs(ys - ys[0]).max()),
        "current_rms_A": float(np.sqrt(np.mean(I ** 2))),
        "current_dI_A": float(np.sqrt(np.mean(np.diff(I) ** 2))) if n > 2 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="", help="留空 = 纯协同控制器（无策略）")
    ap.add_argument("--mix", type=float, default=0.0, help="协同控制器混合系数")
    ap.add_argument("--residual-scale", type=float, nargs=3, default=(0.05, 0.02, 0.05))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--stand-level", type=int, default=2)
    a = ap.parse_args()

    model = None
    if a.checkpoint:
        model = PPO.load(a.checkpoint, custom_objects={
            "policy_class": AsymmetricActorCriticPolicy}, device="cpu")
        print(f"checkpoint: {a.checkpoint}")
    print(f"mix={a.mix} residual_scale={tuple(a.residual_scale)} steps={a.steps}")
    for dr in (False, True):
        for ini in (0.0, 1.0):
            agg = [rollout(model, steps=a.steps, seed=ep, init_scale=ini, dr=dr,
                           mix=a.mix, residual=tuple(a.residual_scale),
                           stand_level=a.stand_level)
                   for ep in range(a.episodes)]
            keys = ("pitch_rms_deg", "pitch_ptp_deg", "drift_peak_cm", "drift_tail_cm",
                    "drift_final_cm", "yaw_peak_cm", "current_rms_A", "current_dI_A")
            m = {k: float(np.mean([d[k] for d in agg])) for k in keys}
            surv = float(np.mean([d["survived"] for d in agg]))
            print(f"dr={int(dr)} init={ini:.1f} n={agg[0]['n']:5d} surv={surv:.2f} "
                  f"term={agg[0]['term']:>14s} " +
                  " ".join(f"{k.replace('_deg','').replace('_cm','').replace('_A','')}={m[k]:.3f}"
                           for k in keys))


if __name__ == "__main__":
    main()
