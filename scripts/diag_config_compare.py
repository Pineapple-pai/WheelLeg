"""针对"抗冲量"角点的快速配置对比（腿位置环软化程度的影响）。"""
import sys
import numpy as np

sys.path.insert(0, '/home/p/下载/WheelLeg/scripts')
from optimize_coord_robust import rollout  # noqa: E402
from uz05.balance import CoordinatedBalanceParams as P  # noqa: E402

CASES = [
    (1, True, -0.030, 0, 0.15), (1, False, -0.030, 0, 0.15),
    (2, False, 0, 0, 0.20), (2, False, 0, 0, -0.20),
    (0, True, 0, 0, 0.25), (0, True, 0, 0, -0.25),
    (0, True, -0.050, -0.05, 0.0), (0, False, 0, 0, 0.0),
    (0, False, 0.030, 0, 0.0),
]
BASE = dict(kl_p=1.0, kl_v=1.0, kl_x=2.0, kl_i=4.0, kw_p=14., kw_d=14., kw_x=15.)
CFGS = {
    "kl_p=1.0 (cur)": dict(),
    "kl_p=1.4": dict(kl_p=1.4),
    "kl_p=2.0": dict(kl_p=2.0),
    "kl_p=2.8": dict(kl_p=2.8),
    "kl_p=4.0": dict(kl_p=4.0),
    "kl_p=1.4,kl_d=0.06": dict(kl_p=1.4, kl_d=0.06),
    "kl_p=2.0,kl_d=0.10": dict(kl_p=2.0, kl_d=0.10),
    "kl_p=2.8,kl_d=0.14": dict(kl_p=2.8, kl_d=0.14),
    "kl_p=1.4,kl_x=4": dict(kl_p=1.4, kl_x=4.0),
    "kl_p=2.0,kl_x=6": dict(kl_p=2.0, kl_x=6.0),
}

if __name__ == "__main__":
    print(f"{'config':18s} " + " ".join(f"c{i}" for i in range(len(CASES))) +
          "   pass  sumxpeak")
    for tag, over in CFGS.items():
        kw = dict(BASE)
        kw.update(over)
        p = P(**kw)
        res = []
        ok = 0
        sx = 0.0
        for case in CASES:
            d = rollout(p, case, 1200)
            res.append(d)
            ok += int(d["survived"])
            sx += d["xpeak"]
        cells = " ".join(f"{'P' if d['survived'] else 'F'}{d['prms']:4.1f}" for d in res)
        print(f"{tag:18s} {cells}   {ok}/{len(CASES)}  {sx:7.1f}")
