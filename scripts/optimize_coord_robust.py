"""协同平衡控制器的**鲁棒性**优化（网格种子 + (1+1)-ES）。

优化目标（压力目标，比验收更苛刻）：
  * 初始倾角 ±3°（课程只到 ±1.7°）、初始倾角速率 ±0.15 rad/s
  * 整机前向/后向速度冲量 ±0.15 m/s
  * 5 个域随机化采样（含最苛刻的 seed 0）± 开关域随机化
  * 腿部偏置饱和比例（不能靠长期顶限位换稳定）

成本 = 失败惩罚 + 漂移 + pitch + 饱和 + 电流颤振。

用法::
    python scripts/optimize_coord_robust.py grid --steps 900
    python scripts/optimize_coord_robust.py refine --iters 4000
    python scripts/optimize_coord_robust.py verify
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.balance import CoordinatedBalanceParams  # noqa: E402
from uz05.env import UZ05Env  # noqa: E402

DT = 0.008
OUT = Path(__file__).resolve().parent / "coord_robust_best.json"

# 压力用例：(seed, 是否域随机化, 初始倾角 rad, 倾角速率 rad/s, 速度冲量 m/s)
STRESS = [
    # ★ 实测最容易失败的用例：seed 0 标称机器 + 无域随机化（腿部位置环饱和后
    #   进入极限环 → station_limit）。必须放在压力集里。
    (0, False, 0.000, 0.00, 0.00),
    (0, False, 0.050, 0.05, 0.00),
    (0, True, 0.050, 0.05, 0.00),
    (0, True, -0.050, -0.05, 0.00),
    (1, True, 0.030, 0.00, 0.15),
    (1, True, -0.030, 0.00, -0.15),
    (2, False, 0.000, 0.00, 0.20),
    (2, False, 0.000, 0.00, -0.20),
    (0, True, 0.000, 0.00, 0.25),
    (0, True, 0.000, 0.00, -0.25),
    (3, True, 0.040, 0.10, 0.10),
    (4, True, -0.040, -0.10, -0.10),
]

KEYS = ("kl_p", "kl_d", "kl_v", "kl_x", "kl_i",
        "kw_p", "kw_d", "kw_v", "kw_x", "kw_i", "ky_p", "filter_alpha")
BOUNDS = {
    "kl_p": (0.2, 1.6), "kl_d": (0.0, 0.30), "kl_v": (0.0, 5.0),
    "kl_x": (2.0, 30.0), "kl_i": (0.0, 40.0),
    "kw_p": (4.0, 24.0), "kw_d": (2.0, 24.0), "kw_v": (0.0, 10.0),
    "kw_x": (0.0, 30.0), "kw_i": (0.0, 20.0), "ky_p": (0.0, 14.0),
    "filter_alpha": (0.10, 1.0),
}


def rollout(p: CoordinatedBalanceParams, case, steps: int):
    seed, dr, tilt, tilt_rate, vel = case
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=1.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0, coord_params=p,
                  extra_init_tilt=tilt, extra_init_tilt_rate=tilt_rate,
                  extra_init_vel=vel)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.stage = replace(env.stage, episode_steps=steps)
    env.reset(seed=seed)
    th, xs, I, u, n = [], [], [], [], 0
    info: dict = {}
    for k in range(steps):
        _, _, term, trunc, info = env.step(np.zeros(6))
        q = env.sim.base_quat
        rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
        th.append(float(rpy[1]))
        xs.append(float(env.sim.data.qpos[0] - env.nominal_xy[0]))
        I.append(float(info["wheel_current_left"]))
        u.append(float(info["coord_leg_offset"]))
        n = k + 1
        if term or trunc:
            break
    env.close()
    th = np.asarray(th); xs = np.asarray(xs); I = np.asarray(I); u = np.asarray(u)
    tail = slice(max(0, n - 200), n)
    return {
        "n": n, "survived": n >= steps, "term": info.get("termination_reason", "?"),
        "prms": float(np.degrees(np.sqrt(np.mean(th ** 2)))),
        "ptp": float(np.degrees(th.max() - th.min())),
        "xpeak": float(100 * np.abs(xs).max()),
        "xtail": float(100 * np.mean(np.abs(xs[tail]))),
        "dI": float(np.sqrt(np.mean(np.diff(I) ** 2))) if n > 2 else 0.0,
        "Irms": float(np.sqrt(np.mean(I ** 2))),
        "sat": float(np.mean(np.abs(u) > 0.98 * p.leg_offset_limit)),
    }


def evaluate(p: CoordinatedBalanceParams, steps: int, cases=STRESS):
    tot = 0.0
    worst = []
    for case in cases:
        d = rollout(p, case, steps)
        c = 0.0
        if not d["survived"]:
            c += 60.0 + 60.0 * (steps - d["n"]) / steps
        c += 4.0 * d["prms"] + 0.5 * d["ptp"]
        c += 6.0 * d["xpeak"] + 3.0 * d["xtail"]
        c += 0.6 * d["dI"] + 30.0 * d["sat"]
        tot += c
        worst.append((c, case, d))
    worst.sort(key=lambda x: -x[0])
    return tot / len(cases), worst


def to_params(v: np.ndarray) -> CoordinatedBalanceParams:
    kw = {k: float(np.clip(v[i], *BOUNDS[k])) for i, k in enumerate(KEYS)}
    return CoordinatedBalanceParams(**kw)


def to_vec(p: CoordinatedBalanceParams) -> np.ndarray:
    return np.asarray([getattr(p, k) for k in KEYS])


def show(tag: str, cost: float, worst, cases=STRESS) -> None:
    print(f"[{tag}] cost={cost:.3f}")
    for c, case, d in worst:
        seed, dr, tilt, tr, vel = case
        print(f"   c={c:7.2f} seed={seed} dr={int(dr)} tilt={np.degrees(tilt):+5.2f}deg "
              f"tr={tr:+.2f} v={vel:+.2f} | n={d['n']:4d} surv={int(d['survived'])} "
              f"prms={d['prms']:6.3f} xpeak={d['xpeak']:6.2f} dI={d['dI']:6.3f} "
              f"sat={d['sat']:.3f} {d['term']}")


def cmd_grid(a) -> None:
    t0 = time.time()
    rows = []
    # ★ 实测关键：腿的位置环太硬（kl_x=8, kl_i=20）会在速度误差稍大时
    #   比例饱和到 ±0.35 rad，随后进入极限环 → station_limit。软化到
    #   kl_x=2, kl_i=4, kl_v=1 后，同样的 0.2 m/s 冲量可以稳住且不饱和。
    grid = {
        "kl_x": [1.0, 2.0, 3.5],
        "kl_i": [2.0, 4.0, 8.0],
        "kl_v": [0.5, 1.0, 2.0],
        "kw_x": [10.0, 15.0, 24.0],
        "kw_d": [14.0, 20.0],
    }
    keys = list(grid)
    base = CoordinatedBalanceParams()
    for combo in itertools.product(*[grid[k] for k in keys]):
        kw = {k: getattr(base, k) for k in KEYS}
        kw.update(dict(zip(keys, combo)))
        p = CoordinatedBalanceParams(**kw)
        cost, worst = evaluate(p, a.steps)
        rows.append((cost, p, worst))
    rows.sort(key=lambda r: r[0])
    print(f"# grid {len(rows)} 组，{time.time() - t0:.0f}s")
    for cost, p, worst in rows[:8]:
        show("grid", cost, worst[:3])
    OUT.write_text(json.dumps({k: getattr(rows[0][1], k) for k in KEYS}, indent=2))
    print("saved", OUT)


def cmd_refine(a) -> None:
    rng = np.random.default_rng(a.seed)
    keys = list(KEYS)
    lo = np.asarray([BOUNDS[k][0] for k in keys])
    hi = np.asarray([BOUNDS[k][1] for k in keys])
    start = json.loads(OUT.read_text()) if OUT.exists() else None
    v = (np.asarray([start[k] for k in keys]) if start
         else np.asarray([getattr(CoordinatedBalanceParams(), k) for k in keys]))
    cost, worst = evaluate(to_params(v), a.steps)
    show("start", cost, worst[:3])
    sigma = 0.10 * (hi - lo)
    best = cost
    for i in range(a.iters):
        cand = np.clip(v + rng.normal(0, 1, v.size) * sigma, lo, hi)
        c, w = evaluate(to_params(cand), a.steps)
        if c < best:
            v, best, worst = cand, c, w
            sigma *= 1.05
            print(f"[{i:5d}] NEW cost={c:.3f} worst={w[0][2]['n']} "
                  f"prms={w[0][2]['prms']:.3f} xpeak={w[0][2]['xpeak']:.2f}", flush=True)
        else:
            sigma = np.maximum(sigma * 0.995, 1e-4 * (hi - lo))
        if (i + 1) % 500 == 0:
            print(f"[{i + 1}/{a.iters}] cost={best:.3f}", flush=True)
            OUT.write_text(json.dumps(
                {k: float(v[j]) for j, k in enumerate(keys)}, indent=2))
    params = to_params(v)
    OUT.write_text(json.dumps({k: getattr(params, k) for k in KEYS}, indent=2))
    print("[done]", json.dumps({k: round(getattr(params, k), 4) for k in KEYS}, indent=2))
    show("done", best, worst[:4])


def cmd_verify(a) -> None:
    from uz05.balance import CoordinatedBalanceParams as P
    params = P(**json.loads(OUT.read_text())) if OUT.exists() else P()
    print("params:", json.dumps({k: round(getattr(params, k), 4) for k in KEYS}))
    cost, worst = evaluate(params, a.steps)
    show("stress", cost, worst)
    print("--- 标准验收条件（应为全通过）---")
    ok = 0
    for seed in (0, 1, 2, 3, 4):
        for dr in (False, True):
            for tilt in (0.0, 0.017, 0.030):
                d = rollout(params, (seed, dr, tilt, 0.0, 0.0), a.steps)
                ok += int(d["survived"])
                print(f"  seed={seed} dr={int(dr)} tilt={np.degrees(tilt):+5.2f}deg "
                      f"n={d['n']:4d} surv={int(d['survived'])} prms={d['prms']:6.3f} "
                      f"xpeak={d['xpeak']:6.2f} {d['term']}")
    print(f"通过 {ok}/30")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("grid", "refine", "verify"))
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    {"grid": cmd_grid, "refine": cmd_refine, "verify": cmd_verify}[a.mode](a)
