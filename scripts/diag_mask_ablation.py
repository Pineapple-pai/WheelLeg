"""通道掩码消融：只看腿 / 只看轮，能否撑住零指令站定？

用来确认"腿 + 轮协同"的必要性：任何单独一半都不够，两半合起来才达标。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(channels: str, steps: int = 1500, seed: int = 0, dr: bool = True) -> dict:
    env = UZ05Env(stage='stand', stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=1.0, coord_channels=channels)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=seed)
    th, xs, I, u, n = [], [], [], [], 0
    info: dict = {}
    for k in range(steps):
        _, _, term, trunc, info = env.step(np.zeros(6))
        q = env.sim.base_quat
        rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz')
        th.append(float(rpy[1]))
        xs.append(float(info['station_error']))
        I.append(float(info['wheel_current_left']))
        u.append(float(info['coord_leg_offset']))
        n = k + 1
        if term or trunc:
            break
    env.close()
    th = np.asarray(th); xs = np.asarray(xs)
    I = np.asarray(I) if I else np.zeros(1)
    u = np.asarray(u) if u else np.zeros(1)
    return dict(n=n, surv=n >= steps, term=info.get('termination_reason', '?'),
                prms=float(np.degrees(np.sqrt(np.mean(th ** 2)))),
                xpeak=float(100 * np.abs(xs).max()),
                Irms=float(np.sqrt(np.mean(I ** 2))),
                sat=float(np.mean(np.abs(u) > 0.34)))


if __name__ == "__main__":
    print("channels  dr |     n surv  prms  xpeak  Irms  sat term")
    for ch in ('all', 'legs', 'wheels', 'none'):
        for dr in (False, True):
            d = run(ch, dr=dr)
            print(f"{ch:8s} {int(dr):2d} | {d['n']:5d} {int(d['surv'])} {d['prms']:6.3f} "
                  f"{d['xpeak']:6.2f} {d['Irms']:5.2f} {d['sat']:.3f} {d['term']}")
