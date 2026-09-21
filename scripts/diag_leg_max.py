"""腿长可达上限（负载下）：命令很长的目标，看实际能到多少、是否还稳。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataclasses import replace
from uz05.env import UZ05Env

for target in (0.26, 0.28, 0.30, 0.32, 0.35):
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0, init_scale=0.0,
                  lock_stand_leg_actions=False, stand_leg_action_limit=1.0, coord_mix=1.0)
    env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=5000)
    env.reset(seed=0)
    env.coordinated.enable_leg_length(0.15, 0.35, ref=target)
    leg, pitch, drift, n = [], [], [], 0
    info = {}
    for k in range(5000):
        _, _, term, trunc, info = env.step(np.zeros(6))
        n = k + 1
        if k >= 4000:
            leg.append(float(np.mean(env.sim.leg_lengths())))
            pitch.append(float(info.get("pitch", 0.0)))
            drift.append(float(info.get("station_error", 0.0)))
        if term or trunc: break
    env.close()
    if len(leg) < 40:
        print(f"target={target:.2f} FELL n={n} term={info.get('termination_reason')}")
    else:
        print(f"target={target:.2f} leg={np.mean(leg):.4f} base_z={env_z if False else 0:.4f} "
              f"pitch_rms={np.degrees(np.sqrt(np.mean(np.square(pitch)))):.3f}° "
              f"drift_pk={100*max(abs(np.array(drift))):.3f}cm n={n} sat={env.coordinated.height_diff:+.3f}")
