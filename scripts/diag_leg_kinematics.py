"""腿机构运动学：主动关节角 → 腿长（对称位形）。

UZ-05 每条腿是闭环五连杆：2 个主动关节（L2 / L4，都绕 x 轴）+ 3 个被动关节
+ 2 个 connect 等式约束。只有**对称驱动**（L2 与 L4 镜像）才能保持轮平面，
因此先扫 (q2, q4) 二维平面，找出腿长的可达范围与最优驱动方向。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)


def leg_length_for(env, q2: float, q4: float) -> tuple[float, float, float]:
    """返回 (左腿长, 右腿长, 髋-轮相对机体的 x 偏移)。"""
    sim = env.sim
    q = np.asarray(env.params.robot.stand_joint_pos, dtype=np.float64).copy()
    # hip_qpos_adr 顺序 L2, L4, R2, R4；左右同号驱动
    q[0] = q2
    q[1] = q4
    q[2] = -q2
    q[3] = -q4
    sim.data.qpos[sim.hip_qpos_adr] = q
    sim.forward()
    lengths = sim.leg_lengths()
    rel = sim.data.site_xpos[sim.wheel_sites[0]] - sim.data.site_xpos[sim.hip_sites[0]]
    rel_body = sim.body_frame(rel)
    return float(lengths[0]), float(lengths[1]), float(rel_body[0])


def main() -> None:
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=0,
                  init_scale=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    base = np.asarray(env.params.robot.stand_joint_pos, dtype=np.float64)
    print("nominal stand_joint_pos:", base)
    print("nominal leg length:", env.sim.leg_lengths())

    print("\n=== 单关节扫描（另一关节保持 nominal）===")
    for idx, name in ((0, "q2(L2)"), (1, "q4(L4)")):
        print(f"-- {name} --")
        for delta in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
            q = base.copy()
            q[idx] += delta
            q[2 + idx] = -q[idx]
            env.sim.data.qpos[env.sim.hip_qpos_adr] = q
            env.sim.forward()
            L = env.sim.leg_lengths()
            rel = env.sim.body_frame(
                env.sim.data.site_xpos[env.sim.wheel_sites[0]]
                - env.sim.data.site_xpos[env.sim.hip_sites[0]])
            print(f"  delta={delta:+.2f} legL={L[0]:.4f} legR={L[1]:.4f} "
                  f"rel_x={rel[0]:+.4f} rel_z={rel[2]:+.4f}")

    print("\n=== 对称驱动扫描（q2 与 q4 同向、等量偏置）===")
    for delta in (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8):
        L, R, relx = leg_length_for(env, base[0] + delta, base[1] + delta)
        print(f"  delta={delta:+.2f} legL={L:.4f} legR={R:.4f} rel_x={relx:+.4f}")

    print("\n=== 反向驱动扫描（q2 与 q4 反号、等量）===")
    for delta in (-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8):
        L, R, relx = leg_length_for(env, base[0] + delta, base[1] - delta)
        print(f"  delta={delta:+.2f} legL={L:.4f} legR={R:.4f} rel_x={relx:+.4f}")

    print("\n=== 粗网格：找可达腿长范围 ===")
    best = {}
    for q2 in np.linspace(-1.2, 0.6, 19):
        for q4 in np.linspace(-0.6, 1.2, 19):
            L, R, relx = leg_length_for(env, float(q2), float(q4))
            key = round(L, 3)
            best.setdefault(key, (q2, q4, relx))
    keys = sorted(best)
    print(f"可达腿长 {keys[0]:.3f} .. {keys[-1]:.3f} m（{len(keys)} 个采样点）")
    for k in (keys[0], keys[len(keys) // 4], keys[len(keys) // 2],
              keys[3 * len(keys) // 4], keys[-1]):
        q2, q4, relx = best[k]
        print(f"  L={k:.3f}  q2={q2:+.3f} q4={q4:+.3f}  rel_x={relx:+.4f}")
    env.close()


if __name__ == "__main__":
    main()
