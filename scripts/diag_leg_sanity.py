"""健全性检查：腿长环跑起来时，控制器是否真的在工作？"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataclasses import replace
from uz05.env import UZ05Env

env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0, init_scale=0.0,
              lock_stand_leg_actions=False, stand_leg_action_limit=1.0, coord_mix=1.0)
env.params.domain_randomization.enabled = False
env.stage = replace(env.stage, episode_steps=2500)
env.reset(seed=0)
env.coordinated.enable_leg_length(0.15, 0.26, ref=0.24)
print("  k  leg_len  base_z   pitch°  drift_cm  I_l    I_r   ctrl_q2 ctrl_q4  contact")
for k in range(2500):
    _, _, term, trunc, info = env.step(np.zeros(6))
    if k % 250 == 0 or term or trunc:
        q = env.sim.joint_positions()
        print(f"{k:4d} {np.mean(env.sim.leg_lengths()):7.4f} {env.sim.data.qpos[2]:7.4f} "
              f"{np.degrees(info['pitch']):+7.3f} {100*info['station_error']:+8.3f} "
              f"{info['wheel_current_left']:+6.3f} {info['wheel_current_right']:+6.3f} "
              f"{q[0]:+7.4f} {q[1]:+7.4f} {info['wheel_force_left']:6.1f}")
    if term or trunc:
        print("TERM", info['termination_reason'], "at", k); break
print("ctrl wheel_vel", env.sim.wheel_velocities(), "qvel[0]", env.sim.data.qvel[0])
print("ncon", env.sim.data.ncon)
env.close()
