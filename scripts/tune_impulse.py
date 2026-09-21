"""冲量恢复专项调参：在"标准条件 + 冲量角点"上做结构化搜索。

结论先行（供后续参考）：腿偏置 ±0.35 rad ⇒ 轮心前后行程 ±8.3 cm。0.15 m/s 冲量
的"纯阻尼"停止距离只有 ~1 cm，但姿态耦合会把回弹速度放大到 0.33 m/s，回弹
超出 station_limit。因此这里搜的是"回弹阶段"的增益组合。

用法::
    python scripts/tune_impulse.py --iters 300 --steps 900
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_coord_robust import rollout  # noqa: E402
from uz05.balance import CoordinatedBalanceParams as P  # noqa: E402

# 一半标准条件（必须全过）+ 一半冲量角点
CASES = [
    (0, False, 0.000, 0.00, 0.00),
    (2, False, 0.000, 0.00, 0.00),
    (3, True, 0.030, 0.00, 0.00),
    (0, True, -0.050, -0.05, 0.00),
    (1, True, -0.030, 0.00, 0.15),
    (2, False, 0.000, 0.00, 0.20),
    (0, True, 0.000, 0.00, 0.25),
    (4, True, -0.040, -0.10, -0.10),
]
from uz05.balance import CoordinatedBalanceParams as _P
# 以平衡控制器 dataclass 的**真实默认值**为种子，避免脚本与默认值不一致
BASE = {k: getattr(_P(), k) for k in (
    "kl_p", "kl_d", "kl_v", "kl_x", "kl_i", "kw_p", "kw_d", "kw_v", "kw_x",
    "kw_i", "ky_p", "filter_alpha", "integral_leg_limit", "integral_wheel_limit")}
TUNE = ("kl_p", "kl_d", "kl_v", "kl_x", "kl_i", "kw_v", "kw_x", "kw_d", "kw_p",
        "filter_alpha")
BOUNDS = {
    "kl_p": (0.0, 2.5), "kl_d": (0.0, 0.25), "kl_v": (0.0, 4.0),
    "kl_x": (0.0, 12.0), "kl_i": (0.0, 12.0),
    "kw_p": (6.0, 22.0), "kw_d": (6.0, 22.0), "kw_v": (0.0, 6.0),
    "kw_x": (0.0, 30.0), "filter_alpha": (0.15, 0.9),
}
OUT = Path(__file__).resolve().parent / "coord_impulse_best.json"


def evaluate(kw, steps):
    p = P(**{**BASE, **kw})
    tot, rows = 0.0, []
    for case in CASES:
        d = rollout(p, case, steps)
        c = 0.0
        if not d["survived"]:
            c += 100.0 + 100.0 * (steps - d["n"]) / steps
        c += 4.0 * d["prms"] + 6.0 * d["xpeak"] + 3.0 * d["xtail"] + 0.6 * d["dI"]
        tot += c
        rows.append((c, case, d))
    rows.sort(key=lambda x: -x[0])
    return tot / len(CASES), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    v = np.asarray([BASE[k] for k in TUNE])
    lo = np.asarray([BOUNDS[k][0] for k in TUNE])
    hi = np.asarray([BOUNDS[k][1] for k in TUNE])
    cost, rows = evaluate(dict(zip(TUNE, v)), a.steps)
    print(f"[start] cost={cost:.2f} pass={sum(int(r[2]['survived']) for r in rows)}/{len(CASES)}",
          flush=True)
    for c, case, d in rows[:3]:
        print(f"   c={c:7.2f} {case} n={d['n']} surv={int(d['survived'])} "
              f"prms={d['prms']:6.3f} xpeak={d['xpeak']:6.2f}", flush=True)
    sigma = 0.20 * (hi - lo)
    best, best_rows = cost, rows
    t0 = time.time()
    for it in range(a.iters):
        cand = np.clip(v + rng.normal(0, 1, v.size) * sigma, lo, hi)
        c, r = evaluate(dict(zip(TUNE, cand)), a.steps)
        if c < best:
            v, best, best_rows = cand, c, r
            sigma = np.minimum(sigma * 1.08, 0.3 * (hi - lo))
            print(f"[{it:4d}] NEW cost={c:.2f} pass={sum(int(x[2]['survived']) for x in r)}"
                  f"/{len(CASES)} worst_n={r[0][2]['n']}", flush=True)
            OUT.write_text(json.dumps({**BASE, **dict(zip(TUNE, v.tolist()))}, indent=2))
        else:
            sigma = np.maximum(sigma * 0.995, 1e-4 * (hi - lo))
        if (it + 1) % 50 == 0:
            print(f"[{it + 1}/{a.iters}] cost={best:.2f} ({time.time() - t0:.0f}s)", flush=True)
    kw = {**BASE, **dict(zip(TUNE, v.tolist()))}
    OUT.write_text(json.dumps(kw, indent=2))
    print("[done]", json.dumps({k: round(x, 4) for k, x in kw.items()}, indent=2))
    for c, case, d in best_rows:
        print(f"   c={c:7.2f} {case} n={d['n']} surv={int(d['survived'])} "
              f"prms={d['prms']:6.3f} xpeak={d['xpeak']:6.2f} dI={d['dI']:.3f} {d['term']}")


if __name__ == "__main__":
    main()
