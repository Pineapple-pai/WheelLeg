"""诊断：站立可行性 / 辅助控制器 / checkpoint 行为，逐项打印。

用法::

    conda run --no-capture-output -n sim python -u scripts/diag_stand.py \
        [--checkpoint PATH] [--assist 0.0] [--stand-level 2] [--episodes 5] [--trace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(env: UZ05Env, policy, episodes: int, seed0: int, trace: bool = False):
    rows = []
    trace_rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        info: dict = {}
        step = 0
        while not done:
            if policy is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action = policy(obs)
            obs, rew, term, trunc, info = env.step(action)
            step += 1
            if trace and (step % 25 == 0 or step <= 3):
                trace_rows.append(
                    dict(step=step, pitch=info["pitch"], roll=info["roll"],
                         height=info["base_height"], vx=info["body_vx"],
                         x=info["station_error"], wheelL=info["wheel_current_left"],
                         wheelR=info["wheel_current_right"],
                         legL=info["leg_length_left"], legR=info["leg_length_right"],
                         bal=info["balance_torque"], rew=rew)
                )
            done = term or trunc
        rows.append(dict(survived=bool(trunc and not term), steps=step,
                         reason=info["termination_reason"],
                         tail=info["station_tail_mean"], peak=info["station_max_abs"],
                         height=info["base_height"], pitch=info["pitch"],
                         legL=info["leg_length_left"], legR=info["leg_length_right"],
                         bal=info["balance_torque"]))
    return rows, trace_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--assist", type=float, default=0.0)
    ap.add_argument("--stand-level", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--policy-current-scale", type=float, default=None)
    args = ap.parse_args()

    env = UZ05Env(stage="stand", stand_level=args.stand_level, assist_scale=args.assist)
    if args.no_noise:
        env.params.noise.enabled = False
    if args.policy_current_scale is not None:
        env.params.wheel.policy_current_scale_a = args.policy_current_scale

    policy = None
    if args.checkpoint:
        from stable_baselines3 import PPO
        from train_uz05 import AsymmetricActorCriticPolicy
        model = PPO.load(args.checkpoint, custom_objects={
            "policy_class": AsymmetricActorCriticPolicy}, device="cpu")

        def policy(obs):  # noqa: E306
            return model.predict(obs, deterministic=True)[0]

    rows, trace_rows = run(env, policy, args.episodes, args.seed, args.trace)
    print(f"stand_level={env.stand_level.name} assist={args.assist} "
          f"action_limit={env.action_limit} policy_cur_scale="
          f"{env.params.wheel.policy_current_scale_a} "
          f"station_hard={env.params.station_hard_limit} tilt_limit={env.tilt_limit} "
          f"target_height={env.command_target[3]}")
    survive = np.mean([r["survived"] for r in rows])
    print(f"survive_rate: {survive:.2f}")
    for r in rows:
        print(f"  survived={int(r['survived'])} steps={r['steps']:4d} "
              f"reason={r['reason']:<40s} tail_cm={100*r['tail']:7.2f} "
              f"peak_cm={100*r['peak']:7.2f} h={r['height']:.3f} "
              f"pitch={r['pitch']:+.3f} legL={r['legL']:.3f} legR={r['legR']:.3f} "
              f"bal={r['bal']:+.3f}")
    if trace_rows:
        print("\nstep  pitch    roll     height   vx      x_err   curL    curR    "
              "legL    legR    bal     rew")
        for t in trace_rows:
            print(f"{t['step']:5d} {t['pitch']:+.4f} {t['roll']:+.4f} {t['height']:.4f} "
                  f"{t['vx']:+.3f} {t['x']:+.4f} {t['wheelL']:+.2f} {t['wheelR']:+.2f} "
                  f"{t['legL']:.4f} {t['legR']:.4f} {t['bal']:+.3f} {t['rew']:+.3f}")
    env.close()


if __name__ == "__main__":
    main()
