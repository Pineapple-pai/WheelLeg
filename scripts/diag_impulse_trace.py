"""冲量响应的逐帧轨迹：定位腿偏置饱和发生在"刹车"还是"回弹"阶段。"""
import sys
import numpy as np
sys.path.insert(0, '/home/p/下载/WheelLeg/scripts')
from scipy.spatial.transform import Rotation  # noqa: E402
from uz05.env import UZ05Env  # noqa: E402
from uz05.balance import CoordinatedBalanceParams as P  # noqa: E402
from dataclasses import replace  # noqa: E402

p = P(kl_p=1.0, kl_v=1.0, kl_x=2.0, kl_i=4.0, kw_p=14., kw_d=14., kw_x=15.)
env = UZ05Env(stage='stand', stand_level=2, assist_scale=0.0, seed=1, init_scale=1.0,
              lock_stand_leg_actions=False, stand_leg_action_limit=1.0, coord_mix=1.0,
              coord_params=p, extra_init_tilt=-0.030, extra_init_vel=0.15)
env.params.domain_randomization.enabled = False
env.stage = replace(env.stage, episode_steps=1500)
env.reset(seed=1)
print("  k  pitch_deg  pitchrate   x_cm     vx    leg_off    I_A   u_kp   u_kv   u_kx   u_ki")
ki = 0.0
for k in range(90):
    o, r, t, tr, info = env.step(np.zeros(6))
    ki = env.coordinated.integral
    q = env.sim.base_quat
    rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')
    th = float(rpy[1]); w = float(env.sim.base_ang_vel_world[1])
    vx = info['body_vx']; xe = info['station_error']
    u_kp = -1.0 * th; u_kv = 1.0 * vx; u_kx = 2.0 * xe; u_ki = 4.0 * ki
    if k % 5 == 0 or k > 74:
        print(f"{k:4d} {np.degrees(th):+9.3f} {w:+9.3f} {100*xe:+8.3f} {vx:+8.3f} "
              f"{info['coord_leg_offset']:+9.4f} {info['wheel_current_left']:+7.3f} "
              f"{u_kp:+7.4f}{u_kv:+7.4f}{u_kx:+7.4f}{u_ki:+7.4f}")
    if t or tr:
        print("TERM", info['termination_reason'], "at", k)
        break
env.close()
