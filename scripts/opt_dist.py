"""扰动恢复专项优化：只调与"抗冲量"相关的增益。

背景：软化腿位置环后，标准条件 116/120 通过，但"初始前向冲量 + 后倾"的角点
仍会在 ±10 cm 处触发 station_limit —— 位置环要求的腿偏置吃满 ±0.35 rad 行程，
之后无法再把 CoP 移回质心前方。这里专门优化这一族工况。
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_coord_robust import KEYS, BOUNDS, rollout, to_params, to_vec
from uz05.balance import CoordinatedBalanceParams as P

# 以"软化位置环"的实测最优点为起点
BASE = dict(kl_p=1.0, kl_d=0.0, kl_v=1.0, kl_x=2.0, kl_i=4.0,
            kw_p=14.0, kw_d=14.0, kw_v=0.0, kw_x=15.0, kw_i=0.0,
            ky_p=10.0, filter_alpha=0.35)

CASES = [
    # 标准（必须全过）
    (0, False, 0.000, 0.00, 0.00),
    (0, False, 0.030, 0.00, 0.00),
    (2, False, 0.000, 0.00, 0.00),
    (3, True, 0.000, 0.00, 0.00),
    # 冲量族（当前失败点）
    (1, False, -0.030, 0.00, 0.15),
    (1, True, -0.030, 0.00, 0.15),
    (2, False, 0.000, 0.00, 0.20),
    (2, False, 0.000, 0.00, -0.20),
    (0, True, 0.000, 0.00, 0.25),
    (0, True, 0.000, 0.00, -0.25),
    # 倾角族
    (0, True, 0.050, 0.05, 0.00),
    (0, True, -0.050, -0.05, 0.00),
    (4, True, -0.040, -0.10, -0.10),
    (3, True, 0.040, 0.10, 0.10),
]
TUNE = ("kl_v", "kl_x", "kl_i", "kw_v", "kw_x", "kl_p", "kw_d", "kw_p")
TUNE_BOUNDS = {
    "kl_p": (0.4, 2.0), "kl_v": (0.0, 4.0), "kl_x": (0.0, 8.0), "kl_i": (0.0, 16.0),
    "kw_p": (8.0, 24.0), "kw_d": (6.0, 24.0), "kw_v": (0.0, 12.0), "kw_x": (0.0, 30.0),
}


def evaluate(p, steps, cases=CASES):
    tot, rows = 0.0, []
    for case in cases:
        d = rollout(p, case, steps)
        c = 0.0
        if not d["survived"]:
            c += 80.0 + 80.0 * (steps - d["n"]) / steps
        c += 4.0 * d["prms"] + 0.5 * d["ptp"] + 6.0 * d["xpeak"] + 3.0 * d["xtail"]
        c += 0.6 * d["dI"] + 30.0 * d["sat"]
        tot += c
        rows.append((c, case, d))
    rows.sort(key=lambda x: -x[0])
    return tot / len(cases), rows


def fmt(p):
    return {k: round(getattr(p, k), 4) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    v = to_vec(P(**BASE))
    keys = list(KEYS)
    lo = np.asarray([BOUNDS[k][0] for k in keys])
    hi = np.asarray([BOUNDS[k][1] for k in keys])
    for k in TUNE:
        i = keys.index(k)
        lo[i], hi[i] = TUNE_BOUNDS[k]
    idx = [keys.index(k) for k in TUNE]
    cost, rows = evaluate(to_params(v), a.steps)
    print(f"[start] cost={cost:.3f}", flush=True)
    for c, case, d in rows[:4]:
        print(f"   c={c:7.2f} {case} n={d['n']} surv={int(d['survived'])} "
              f"prms={d['prms']:6.3f} xpeak={d['xpeak']:6.2f} sat={d['sat']:.3f} {d['term']}",
              flush=True)
    sigma = np.zeros(len(keys))
    for i in idx:
        sigma[i] = 0.15 * (hi[i] - lo[i])
    best, best_rows = cost, rows
    for it in range(a.iters):
        cand = np.clip(v + rng.normal(0, 1, v.size) * sigma, lo, hi)
        c, r = evaluate(to_params(cand), a.steps)
        if c < best:
            v, best, best_rows = cand, c, r
            sigma[idx] = np.minimum(sigma[idx] * 1.06, 0.25 * (hi[idx] - lo[idx]))
            print(f"[{it:5d}] NEW cost={c:.3f} worst={r[0][2]['n']} "
                  f"surv={int(r[0][2]['survived'])} xpeak={r[0][2]['xpeak']:.2f}", flush=True)
        else:
            sigma[idx] = np.maximum(sigma[idx] * 0.995, 1e-4 * (hi[idx] - lo[idx]))
        if (it + 1) % 500 == 0:
            print(f"[{it + 1}/{a.iters}] cost={best:.3f}", flush=True)
            Path("coord_dist_best.json").write_text(
                json.dumps(fmt(to_params(v)), indent=2))
    p = to_params(v)
    Path("coord_dist_best.json").write_text(json.dumps(fmt(p), indent=2))
    print("[done]", json.dumps(fmt(p), indent=2), flush=True)
    for c, case, d in best_rows:
        print(f"   c={c:7.2f} {case} n={d['n']} surv={int(d['survived'])} "
              f"prms={d['prms']:6.3f} xpeak={d['xpeak']:6.2f} dI={d['dI']:.3f} "
              f"sat={d['sat']:.3f} {d['term']}", flush=True)


if __name__ == "__main__":
    main()
