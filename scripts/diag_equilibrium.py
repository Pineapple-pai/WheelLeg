"""找到"负载下的平衡稳态"，用于重置与标定。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def main() -> None:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=1.0)
    env.params.noise.enabled = False
    env.params.wheel.policy_current_scale_a = 8.0
    env.params.station_hard_limit = 10.0
    env.tilt_limit = 1.0
    env.params.terminate_on_drift = 10.0
    env.params.terminate_lateral_vel = 10.0
    hip = []
    pitch = []
    print(f"reset_height={env.params.robot.reset_height} "
          f"nominal_stand_height={env.params.robot.nominal_stand_height} "
          f"nominal_leg_length={env.params.robot.nominal_leg_length}")
    for ep in range(3):
        env.reset(seed=200 + ep)
        a = np.zeros(6)
        info = {}
        for n in range(1500):
            _, _, term, trunc, info = env.step(a)
            if n > 900:
                hip.append([info[f"hip_pos_{i}"] for i in range(4)])
                pitch.append(info["pitch"])
            if term:
                print(f"  ep{ep} terminated at {n}: {info['termination_reason']}")
                break
        print(f"ep{ep} final: h={info['base_height']:.5f} "
              f"legL={info['leg_length_left']:.5f} legR={info['leg_length_right']:.5f} "
              f"hip={[round(info[f'hip_pos_{i}'],4) for i in range(4)]} "
              f"wheel_vel={info['wheel_vel_left']:+.3f},{info['wheel_vel_right']:+.3f} "
              f"pitch={info['pitch']:+.4f}")
    hip = np.asarray(hip)
    print(f"\nsteady hip_pos mean={hip.mean(axis=0).round(5).tolist()} "
          f"std={hip.std(axis=0).round(5).tolist()}")
    print(f"steady pitch mean={np.mean(pitch):+.4f} std={np.std(pitch):.4f}")
    env.close()


if __name__ == "__main__":
    main()
