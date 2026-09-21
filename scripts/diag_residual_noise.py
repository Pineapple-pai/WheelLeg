"""量化"策略探索噪声"对站定的破坏，定位 PPO 学不会自主协同的原因。

对每个噪声水平重复跑：策略动作 = 常量偏置 + 零均值高斯噪声，
记录存活步数 / pitch RMS / 漂移。同时把**协同控制器自身的闭环**当作参照
（它的输出是滤波后的，因此对同样的白噪声不敏感）。

用法::
    python scripts/diag_residual_noise.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402


def run(bias: float, sigma: float, residual=(0.10, 0.05, 0.10), steps=2000,
        seed=0, dr=True, init_scale=0.0, filtered=False, alpha=0.35):
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=init_scale, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0,
                  coord_residual_scale=residual)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 12345)
    th, xs, n = [], [], 0
    info: dict = {}
    prev = np.zeros(6)
    for k in range(steps):
        noise = rng.normal(0.0, sigma, 6) if sigma > 0 else np.zeros(6)
        raw = np.full(6, bias) + noise
        if filtered:
            prev = prev + alpha * (raw - prev)
            action = prev
        else:
            action = raw
        obs, _, term, trunc, info = env.step(np.clip(action, -1, 1))
        q = env.sim.base_quat
        rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
        th.append(float(rpy[1]))
        xs.append(float(env.sim.data.qpos[0] - env.nominal_xy[0]))
        n = k + 1
        if term or trunc:
            break
    env.close()
    th = np.asarray(th); xs = np.asarray(xs)
    return {"n": n, "survived": n >= steps, "term": info.get("termination_reason", "?"),
            "prms": float(np.degrees(np.sqrt(np.mean(th ** 2)))),
            "xpeak": float(100 * np.abs(xs).max())}


def main() -> None:
    print("=== 未滤波的探索噪声（= PPO 实际下发的动作）===")
    print(" bias  sigma |  n surv  prms   xpeak term")
    for bias in (0.0, 0.05):
        for sigma in (0.0, 0.01, 0.03, 0.08, 0.15, 0.25):
            d = run(bias, sigma)
            print(f"{bias:5.2f} {sigma:6.3f} | {d['n']:4d} {int(d['survived'])} "
                  f"{d['prms']:6.3f} {d['xpeak']:6.2f} {d['term']}")

    print("\n=== 同样噪声，但经一阶低通（模拟部署滤波 / 平滑策略）===")
    print(" bias  sigma |  n surv  prms   xpeak term")
    for bias in (0.0, 0.05):
        for sigma in (0.03, 0.08, 0.15, 0.25):
            d = run(bias, sigma, filtered=True)
            print(f"{bias:5.2f} {sigma:6.3f} | {d['n']:4d} {int(d['survived'])} "
                  f"{d['prms']:6.3f} {d['xpeak']:6.2f} {d['term']}")

    print("\n=== 无域随机化对照（sigma=0.15）===")
    for flt in (False, True):
        d = run(0.0, 0.15, dr=False, filtered=flt)
        print(f"filtered={int(flt)} n={d['n']} surv={int(d['survived'])} "
              f"prms={d['prms']:.3f} xpeak={d['xpeak']:.2f} {d['term']}")


if __name__ == "__main__":
    main()
