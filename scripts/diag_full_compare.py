"""在完整工况集（标准 120 + 困难 10）上对比协同控制器配置。

用法::
    python scripts/diag_full_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_coord_robust import rollout  # noqa: E402
from uz05.balance import CoordinatedBalanceParams as P  # noqa: E402

# 标准集（课程分布内的零指令站定）：5 seed × 开关域随机化 × 3 初始扰动档。
# 冲量单独放到 HARD 集，避免把"抗冲击"和"站定精度"两个指标混在一起。
STD = [(s, dr, t, 0.0, 0.0) for s in (0, 1, 2, 3, 4) for dr in (False, True)
       for t in (0.0, 0.017, 0.030, -0.030)]
HARD = [
    # 站定 + 冲量（课程分布之外的压力测试）
    (1, True, -0.030, 0.0, 0.15), (1, False, -0.030, 0.0, 0.15),
    (0, True, -0.030, 0.0, 0.15), (0, False, -0.030, 0.0, 0.15),
    (2, False, 0.0, 0.0, 0.20), (2, False, 0.0, 0.0, -0.20),
    (0, True, 0.0, 0.0, 0.25), (0, True, 0.0, 0.0, -0.25),
    (0, True, -0.050, -0.05, 0.0), (0, False, -0.050, -0.05, 0.0),
    (4, True, -0.040, -0.10, -0.10), (3, True, 0.040, 0.10, 0.10),
    (1, True, 0.030, 0.0, 0.15), (1, True, -0.030, 0.0, -0.15),
]
D_LEG_P = 0.6   # 保证"腿也参与姿态"的 kl_p 下限
from uz05.balance import CoordinatedBalanceParams as _P
CFGS = {"★ 当前默认": {k: getattr(_P(), k) for k in (
    "kl_p", "kl_d", "kl_v", "kl_x", "kl_i", "kw_p", "kw_d", "kw_v", "kw_x",
    "kw_i", "ky_p", "filter_alpha")}}

if __name__ == "__main__":
    for tag, kw in CFGS.items():
        p = P(**kw)
        for name, cases in (("STD", STD), ("HARD", HARD)):
            ok = 0
            sx = sp = 0.0
            fails = []
            for case in cases:
                d = rollout(p, case, 1200)
                ok += int(d["survived"])
                sx += d["xpeak"]
                sp += d["prms"]
                if not d["survived"]:
                    fails.append((case, d["n"], round(d["prms"], 2), round(d["xpeak"], 2)))
            n = len(cases)
            print(f"{tag:24s} {name:5s} pass={ok:3d}/{n:3d} ({100 * ok / n:5.1f}%) "
                  f"xpeak_mean={sx / n:5.2f}cm prms_mean={sp / n:5.3f}deg", flush=True)
            if fails and len(fails) <= 14:
                print(f"    fails: {fails}", flush=True)
