"""逐帧跟踪腿长环：确认解耦后的共模/差模与实际腿长。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def main() -> None:
    leg_ref = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=400)
    env.reset(seed=0)
    env.coordinated.enable_leg_length(0.15, 0.31, ref=leg_ref)
    print(f"L_ref={leg_ref} nominal={env.params.robot.nominal_leg_length} "
          f"diff_limit={env.coordinated.p.height_diff_limit:.3f}")
    print("  k  leg_meas  leg_ref  common_a  diff_a   q2      q4    base_z  pitch°  term")
    for k in range(120):
        _, _, term, trunc, info = env.step(np.zeros(6))
        q = env.sim.joint_positions()
        if k < 12 or k % 12 == 0 or term or trunc:
            print(f"{k:4d} {np.mean(env.sim.leg_lengths()):8.4f} "
                  f"{env.coordinated.leg_length_ref:8.4f} "
                  f"{env.coordinated.last['leg_offset']:+8.4f} "
                  f"{env.coordinated.last['height_diff']:+8.4f} "
                  f"{q[0]:+7.3f} {q[1]:+7.3f} {env.sim.data.qpos[2]:7.4f} "
                  f"{np.degrees(info.get('pitch', 0.0)):+7.2f}  {info['termination_reason']}")
        if term or trunc:
            break
    env.close()


if __name__ == "__main__":
    main()
