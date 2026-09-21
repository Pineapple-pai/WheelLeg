"""协同平衡控制器：局部精修 + 鲁棒性验证（UZ-05 零指令站定）。

实测符号（全部经 MuJoCo 验证）::

    正轮电流 I    → ω̇ > 0（后仰），v̇x < 0（前加速）
    正腿偏置 u    → ω̇ < 0（前倾），v̇x > 0（后加速）
    ⇒ 姿态反馈两者都取负；位置反馈两者都取正（腿是低频主力，轮做高频阻尼）。

用法::
    python scripts/coord_refine.py refine --iters 40000
    python scripts/coord_refine.py verify
"""
from __future__ import annotations

import argparse
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
DIF = _S["wheel_differential"]
OUT = Path(__file__).resolve().parent / "coord_gains.json"

KEYS = ("kl_p", "kl_d", "kl_v", "kl_x", "kl_i",
        "kw_p", "kw_d", "kw_v", "kw_x", "kw_i", "ky_p")
BOUNDS = {
    "kl_p": (0.0, 1.2), "kl_d": (0.0, 0.4), "kl_v": (0.0, 3.0),
    "kl_x": (0.0, 20.0), "kl_i": (0.0, 20.0),
    "kw_p": (0.0, 30.0), "kw_d": (0.0, 25.0), "kw_v": (0.0, 15.0),
    "kw_x": (-20.0, 20.0), "kw_i": (0.0, 15.0), "ky_p": (0.0, 10.0),
}


def fresh(seed=0, init_scale=0.0, dr=False, statlim=0.10):
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=init_scale, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0)
    if not dr:
        env.params.domain_randomization.enabled = False
    env.params.station_hard_limit = statlim
    env.reset(seed=seed)
    return env


def rollout(p, steps=1200, seed=0, init_scale=0.0, dr=False, statlim=0.10):
    env = fresh(seed=seed, init_scale=init_scale, dr=dr, statlim=statlim)
    x0 = float(env.sim.data.qpos[0])
    y0 = float(env.sim.data.qpos[1])
    xi = 0.0
    th = np.zeros(steps); xs = np.zeros(steps); ws = np.zeros(steps)
    us = np.zeros(steps); Is = np.zeros(steps); ys = np.zeros(steps)
    n = 0; info = {}
    for k in range(steps):
        sim = env.sim
        q = sim.base_quat
        rpy = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")
        thv = float(rpy[1]); w = float(sim.base_ang_vel_world[1])
        av = sim.body_frame(sim.base_ang_vel_world)
        vx = float(sim.body_frame(sim.base_lin_vel_world)[0])
        xe = float(sim.data.qpos[0] - x0)
        xi = float(np.clip(xi + xe * DT, -0.3, 0.3))
        u = -p["kl_p"] * thv - p["kl_d"] * w + p["kl_v"] * vx + p["kl_x"] * xe + p["kl_i"] * xi
        I = -p["kw_p"] * thv - p["kw_d"] * w + p["kw_v"] * vx + p["kw_x"] * xe + p["kw_i"] * xi
        Id = -p["ky_p"] * float(av[2])
        a = np.zeros(6)
        a[0] = np.clip(I / WHL, -1, 1)
        a[1] = np.clip(Id / DIF, -1, 1)
        a[2:] = np.clip(u / POS, -1, 1)
        _, _, t, tr, info = env.step(a)
        th[k] = thv; xs[k] = xe; ws[k] = w; us[k] = u
        Is[k] = a[0] * WHL; ys[k] = float(sim.data.qpos[1] - y0)
        n = k + 1
        if t or tr:
            break
    env.close()
    tail = slice(max(0, n - 200), n)
    sig = th[:n] - th[:n].mean()
    win = np.hanning(n)
    sp = np.abs(np.fft.rfft(sig * win)) ** 2
    fr = np.fft.rfftfreq(n, DT); sp[0] = 0
    kk = int(np.argmax(sp))
    return dict(
        n=n, surv=n >= steps, term=info.get("termination_reason", "?"),
        prms=float(np.degrees(np.sqrt(np.mean(th[:n] ** 2)))),
        ptp=float(np.degrees(th[:n].max() - th[:n].min())),
        ptail=float(np.degrees(np.mean(np.abs(th[:n][tail])))),
        vrms=float(np.sqrt(np.mean(ws[:n] ** 2))),
        dom=float(fr[kk]),
        xtail=float(100 * np.mean(np.abs(xs[:n][tail]))),
        xpeak=float(100 * np.abs(xs[:n]).max()),
        xfin=float(100 * xs[n - 1]),
        ypeak=float(100 * np.abs(ys[:n]).max()),
        uleg=float(np.sqrt(np.mean(us[:n] ** 2))),
        Irms=float(np.sqrt(np.mean(Is[:n] ** 2))),
    )


def cost(d, steps):
    c = 0.0
    if not d["surv"]:
        c += 40.0 + 60.0 * (steps - d["n"]) / steps
    c += 6.0 * d["ptail"] + 3.0 * d["prms"] + 1.0 * d["ptp"]
    c += 12.0 * d["xtail"] + 3.0 * d["xpeak"] + 2.0 * d["ypeak"]
    c += 1.0 * d["uleg"] + 0.05 * d["Irms"] + 4.0 * d["vrms"]
    return c


def evaluate(p, steps, seeds=(0,), inits=(0.0,)):
    tot = 0.0
    det = {}
    for ini in inits:
        for sd in seeds:
            d = rollout(p, steps=steps, seed=sd, init_scale=ini)
            c = cost(d, steps)
            tot += c
            det[(ini, sd)] = (c, d)
    return tot / (len(seeds) * len(inits)), det


def cmd_refine(args):
    rng = np.random.default_rng(args.seed)
    keys = list(KEYS)
    lo = np.asarray([BOUNDS[k][0] for k in keys])
    hi = np.asarray([BOUNDS[k][1] for k in keys])
    start = json.loads(OUT.read_text()) if OUT.exists() else {}
    v = np.asarray([start.get(k, 0.5 * (BOUNDS[k][0] + BOUNDS[k][1])) for k in keys])
    steps = args.steps
    seeds = (0,) if args.fast else (0, 1, 2)
    inits = (0.0,) if args.fast else (0.0, 0.5)
    c, det = evaluate(dict(zip(keys, v)), steps, seeds, inits)
    print(f"[start] cost={c:.2f}", flush=True)
    for key, (cc, dd) in det.items():
        print(f"   {key} cost={cc:8.2f} n={dd['n']} surv={int(dd['surv'])} ptail={dd['ptail']:6.2f} "
              f"xtail={dd['xtail']:6.2f} xpeak={dd['xpeak']:6.2f}", flush=True)
    sigma = 0.10 * (hi - lo)
    for i in range(args.iters):
        cand = np.clip(v + rng.normal(0, 1, v.size) * sigma, lo, hi)
        cc, cdet = evaluate(dict(zip(keys, cand)), steps, seeds, inits)
        if cc < c:
            v, c, det = cand, cc, cdet
            sigma *= 1.04
        else:
            sigma = np.maximum(sigma * 0.995, 1e-4 * (hi - lo))
        if (i + 1) % 200 == 0:
            k0 = list(det)[0]
            d0 = det[k0][1]
            print(f"[{i + 1:6d}] cost={c:8.2f} ptail={d0['ptail']:6.2f} ptp={d0['ptp']:6.2f} "
                  f"xtail={d0['xtail']:6.2f} xpeak={d0['xpeak']:6.2f} n={d0['n']}", flush=True)
            OUT.write_text(json.dumps(dict(zip(keys, v)), indent=2))
    params = dict(zip(keys, v))
    OUT.write_text(json.dumps(params, indent=2))
    print("[done]", json.dumps(params, indent=2))
    print("[done]", det[list(det)[0]][1])


def cmd_verify(args):
    params = json.loads(Path(args.params_file).read_text())
    print("params:", json.dumps(params))
    print("--- 无域随机化 ---")
    for ini in (0.0, 0.5, 1.0):
        for sd in (0, 1, 2):
            d = rollout(params, steps=args.steps, seed=sd, init_scale=ini)
            print(f"init={ini:.1f} seed={sd} cost={cost(d, args.steps):8.2f} " +
                  "  ".join(f"{k}={v}" for k, v in d.items()))
    print("--- 完整域随机化 ---")
    for ini in (0.0, 1.0):
        for sd in (0, 1, 2):
            d = rollout(params, steps=args.steps, seed=sd, init_scale=ini, dr=True)
            print(f"init={ini:.1f} seed={sd} cost={cost(d, args.steps):8.2f} " +
                  "  ".join(f"{k}={v}" for k, v in d.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("refine", "verify"))
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--params-file", default=str(OUT))
    a = ap.parse_args()
    OUT.write_text(json.dumps({
        "kl_p": 0.25, "kl_d": 0.0, "kl_v": 0.5, "kl_x": 3.0, "kl_i": 0.0,
        "kw_p": 8.0, "kw_d": 12.0, "kw_v": 3.0, "kw_x": 0.0, "kw_i": 0.0, "ky_p": 3.0,
    }, indent=2)) if a.mode == "refine" and not OUT.exists() else None
    {"refine": cmd_refine, "verify": cmd_verify}[a.mode](a)
