"""腿 + 轮协同平衡控制器（零指令站定）：结构化调参。

实测符号（关键，之前全部搞反）::

    正轮电流 I → 地面反力向前 → 机体角速度 dθ/dt 为正（向后仰）
    正腿偏置 u → 轮心相对机体前移 → CoP 前移 → dθ/dt 为正（向后仰）
    正轮电流 I → 机体速度 dvx/dt 为负（向前加速）
    正腿偏置 u → dvx/dt 为正（向后加速）

控制律（两路都以负号反馈 pitch / pitch_rate）::

    I = -kw_p*θ - kw_d*ω        + kw_v*vx + kw_x*x + kw_i*∫x      （轮：高频 pitch 修正）
    u = -kl_p*θ - kl_d*ω + s_l*(kl_v*vx + kl_x*x + kl_i*∫x)      （腿：低频姿态 + 位置）

用法::

    python scripts/tune_coord3.py grid
    python scripts/tune_coord3.py refine --iters 30000
    python scripts/tune_coord3.py verify --params-file scripts/coord_best.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402
from uz05.spec import ACTION_SPEC  # noqa: E402

DT = 0.008
_S = {n: s for n, _, s in ACTION_SPEC}
POS = _S["hip_position_offset"]
WHL = _S["wheel_common"]

KEYS = ("kw_p", "kw_d", "kw_v", "kw_x", "kw_i", "s_l", "kl_p", "kl_d", "kl_v", "kl_x", "kl_i")
BOUNDS = {
    "kw_p": (2.0, 60.0),
    "kw_d": (0.0, 12.0),
    "kw_v": (-20.0, 20.0),
    "kw_x": (-60.0, 60.0),
    "kw_i": (-30.0, 30.0),
    "s_l": (-1.0, 1.0),
    "kl_p": (0.0, 1.0),
    "kl_d": (0.0, 0.4),
    "kl_v": (0.0, 2.0),
    "kl_x": (0.0, 6.0),
    "kl_i": (0.0, 6.0),
}
OUT = Path(__file__).resolve().parent / "coord_best.json"


def fresh_env(seed=0, init_scale=0.0, dr=False):
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=init_scale, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.reset(seed=seed)
    return env


def rollout(p, steps=1000, seed=0, init_scale=0.0, dr=False):
    env = fresh_env(seed=seed, init_scale=init_scale, dr=dr)
    x0 = float(env.sim.data.qpos[0])
    xi = 0.0
    pt = np.zeros(steps)
    xs = np.zeros(steps)
    ws = np.zeros(steps)
    us = np.zeros(steps)
    Is = np.zeros(steps)
    n = 0
    info = {}
    for k in range(steps):
        sim = env.sim
        q = sim.base_quat
        rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
        th = float(rpy[1])
        w = float(sim.base_ang_vel_world[1])
        vx = float(sim.body_frame(sim.base_lin_vel_world)[0])
        xe = float(sim.data.qpos[0] - x0)
        xi = float(np.clip(xi + xe * DT, -0.3, 0.3))
        I = -p["kw_p"] * th - p["kw_d"] * w + p["kw_v"] * vx + p["kw_x"] * xe + p["kw_i"] * xi
        u = (-p["kl_p"] * th - p["kl_d"] * w
             + p["s_l"] * (p["kl_v"] * vx + p["kl_x"] * xe + p["kl_i"] * xi))
        a = np.zeros(6)
        a[0] = np.clip(I / WHL, -1.0, 1.0)
        a[2:] = np.clip(u / POS, -1.0, 1.0)
        _, _, term, trunc, info = env.step(a)
        pt[k] = th
        xs[k] = xe
        ws[k] = w
        us[k] = u
        Is[k] = a[0] * WHL
        n = k + 1
        if term or trunc:
            break
    env.close()
    tail = slice(max(0, n - 200), n)
    return {
        "n": n, "survived": n >= steps, "term": info.get("termination_reason", "?"),
        "ptail": float(np.degrees(np.mean(np.abs(pt[:n][tail])))),
        "prms": float(np.degrees(np.sqrt(np.mean(pt[:n] ** 2)))),
        "ptp": float(np.degrees(pt[:n].max() - pt[:n].min())),
        "wtail": float(np.mean(np.abs(ws[:n][tail]))),
        "xtail": float(100.0 * np.mean(np.abs(xs[:n][tail]))),
        "xpeak": float(100.0 * np.abs(xs[:n]).max()),
        "xfin": float(100.0 * xs[n - 1]),
        "uleg": float(np.sqrt(np.mean(us[:n] ** 2))),
        "Irms": float(np.sqrt(np.mean(Is[:n] ** 2))),
    }


def cost(d, steps):
    c = 0.0
    if not d["survived"]:
        c += 30.0 + 40.0 * (steps - d["n"]) / steps
    c += 4.0 * d["ptail"] + 2.0 * d["prms"] + 0.5 * d["ptp"]
    c += 10.0 * d["xtail"] + 2.0 * d["xpeak"]
    c += 2.0 * d["uleg"] + 0.05 * d["Irms"] + 5.0 * d["wtail"]
    return c


def cmd_grid(args):
    t0 = time.time()
    rows = []
    grid = {
        "kw_p": [6.0, 12.0, 20.0, 30.0],
        "kw_d": [1.0, 3.0, 6.0],
        "kl_p": [0.1, 0.25, 0.45],
        "kl_d": [0.0, 0.06],
        "s_l": [1.0],
    }
    keys = list(grid)
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = {k: 0.0 for k in KEYS}
        p.update(dict(zip(keys, combo)))
        d = rollout(p, steps=args.steps)
        rows.append((cost(d, args.steps), p, d))
    rows.sort(key=lambda r: r[0])
    print(f"# grid {len(rows)} 组，{time.time() - t0:.1f}s")
    for c, p, d in rows[:15]:
        print(f"cost={c:7.2f} n={d['n']:4d} surv={int(d['survived'])} ptail={d['ptail']:6.2f} "
              f"ptp={d['ptp']:6.2f} xtail={d['xtail']:6.2f} xpeak={d['xpeak']:6.2f} "
              f"Irms={d['Irms']:5.2f} | " + " ".join(f"{k}={p[k]:g}" for k in keys))
    if rows and rows[0][2]["survived"]:
        OUT.write_text(json.dumps(rows[0][1], indent=2))
        print("saved", OUT)


def _seed_vec(rng, keys, lo, hi):
    v = np.zeros(len(keys))
    for i, k in enumerate(keys):
        if k == "s_l":
            v[i] = rng.choice([-1.0, 1.0])
        elif k in ("kw_x", "kw_i", "kw_v"):
            v[i] = 0.0
        elif k in ("kl_v", "kl_x", "kl_i"):
            v[i] = 0.0
        else:
            v[i] = rng.uniform(lo[i], hi[i])
    return v


def cmd_refine(args):
    rng = np.random.default_rng(args.seed)
    keys = list(KEYS)
    lo = np.asarray([BOUNDS[k][0] for k in keys])
    hi = np.asarray([BOUNDS[k][1] for k in keys])
    start = json.loads(OUT.read_text()) if OUT.exists() else {k: 0.0 for k in keys}
    best_v = np.asarray([start.get(k, 0.0) for k in keys])
    best_d = rollout(dict(zip(keys, best_v)), steps=args.steps)
    best_c = cost(best_d, args.steps)
    print(f"[start] cost={best_c:.2f} {best_d}", flush=True)

    # 阶段 A：只调 pitch 环 + 腿姿态（位置环关闭）
    phase_a = ("kw_p", "kw_d", "kl_p", "kl_d", "s_l")
    idx_a = [keys.index(k) for k in phase_a]
    sigma = np.zeros(len(keys))
    for i, k in enumerate(keys):
        span = hi[i] - lo[i]
        sigma[i] = 0.10 * span if k in phase_a else 0.0
    for i in range(args.iters):
        cand = np.clip(best_v + rng.normal(0, 1, len(keys)) * sigma, lo, hi)
        d = rollout(dict(zip(keys, cand)), steps=args.steps)
        c = cost(d, args.steps)
        if c < best_c:
            best_v, best_c, best_d = cand, c, d
            print(f"[A {i:6d}] NEW cost={c:8.2f} n={d['n']:4d} surv={int(d['survived'])} "
                  f"ptail={d['ptail']:6.2f} xtail={d['xtail']:6.2f}", flush=True)
        else:
            sigma[idx_a] = np.maximum(sigma[idx_a] * 0.999, 1e-4 * (hi[idx_a] - lo[idx_a]))
        if (i + 1) % 2000 == 0:
            print(f"[A {i + 1}/{args.iters}] cost={best_c:.2f} {best_d}", flush=True)
            OUT.write_text(json.dumps(dict(zip(keys, best_v)), indent=2))

    # 阶段 B：加入位置环（kw_v/kw_x/kw_i 与 kl_v/kl_x/kl_i）
    phase_b = ("kw_v", "kw_x", "kw_i", "kl_v", "kl_x", "kl_i")
    idx_b = [keys.index(k) for k in phase_b]
    for i, k in enumerate(keys):
        if k in phase_b:
            sigma[i] = 0.06 * (hi[i] - lo[i])
    for i in range(args.iters):
        cand = np.clip(best_v + rng.normal(0, 1, len(keys)) * sigma, lo, hi)
        d = rollout(dict(zip(keys, cand)), steps=args.steps)
        c = cost(d, args.steps)
        if c < best_c:
            best_v, best_c, best_d = cand, c, d
            print(f"[B {i:6d}] NEW cost={c:8.2f} n={d['n']:4d} surv={int(d['survived'])} "
                  f"ptail={d['ptail']:6.2f} xtail={d['xtail']:6.2f} xfin={d['xfin']:+.2f}", flush=True)
        else:
            sigma[idx_b] = np.maximum(sigma[idx_b] * 0.999, 1e-4 * (hi[idx_b] - lo[idx_b]))
        if (i + 1) % 2000 == 0:
            print(f"[B {i + 1}/{args.iters}] cost={best_c:.2f} {best_d}", flush=True)
            OUT.write_text(json.dumps(dict(zip(keys, best_v)), indent=2))
    params = dict(zip(keys, best_v))
    OUT.write_text(json.dumps(params, indent=2))
    print("[done]", json.dumps(params, indent=2))
    print("[done]", best_d)


def cmd_verify(args):
    params = json.loads(Path(args.params_file).read_text())
    print("params:", json.dumps(params))
    for init in (0.0, 0.3, 1.0):
        for seed in (0, 1):
            d = rollout(params, steps=args.steps, seed=seed, init_scale=init)
            print(f"init={init:.1f} seed={seed} cost={cost(d, args.steps):7.2f} " +
                  "  ".join(f"{k}={v}" for k, v in d.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("grid", "refine", "verify"))
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--params-file", default=str(OUT))
    a = ap.parse_args()
    {"grid": cmd_grid, "refine": cmd_refine, "verify": cmd_verify}[a.mode](a)
