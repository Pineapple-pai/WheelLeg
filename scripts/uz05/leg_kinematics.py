"""UZ-05 五连杆腿：精确运动学（平面 y-z，绕 x 轴）。

机构（左右几何相同，仅 y 取反）
--------------------------------
链条::

    hip(0,0)
      ├── L2  --(0,-0.087263,-0.036268)--> L5
      │        └── L3 --(0,+0.087263,+0.071002)--> P3   (被动)
      │        └── L1 --(0,-0.038083,+0.052675)--> L6
      │                 └── wheel --(0,+0.232,-0.210458)--> F  (轮心)
      └── L4  --(0,+0.087263,-0.036268)--> P4          (被动，接在底盘上)

两个 connect 闭环::

    P3 ≡ P4                       (site link3_to_link4 与 link4_to_link3)
    Q6 ≡ Q2                       (site link6_to_link2 与 link2_to_link6)

未知角 θ5, θ3, θ1, θ6（被动）+ 输入 θ2, θ4 ⇒ 6 个约束、6 个未知 ⇒ 唯一解。
目标位形用 (腿长 L, 轮心相对髋的前后偏移 x) 给定，即令 F = (x, -L)。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- 机构常数（从 MJCF 提取，方向为 (x, y, z)；运动在 y-z 平面）----
A2 = np.array([-0.087263437, -0.036267659])    # L2 -> L5
A3 = np.array([0.0872634434, 0.071002411])     # L5 -> L3
A1 = np.array([-0.038083202, 0.052675149])     # L5 -> L1
A6 = np.array([-0.106655319, -0.044327161])    # L1 -> L6
AF = np.array([0.232001957, -0.210458263])     # L6 -> wheel center
B4 = np.array([0.087263441, -0.036267658])     # hip -> L4 tip (P4)
P3 = np.array([0.0872634434, 0.071002411])     # L3 -> P3 (局部)
Q6 = np.array([0.038083212, -0.052675111])     # L6 -> Q6 (局部)
Q2 = np.array([-0.193918746, -0.080594782])    # L2 -> Q2 (局部)


def rot(theta: float) -> np.ndarray:
    """绕 x 轴旋转 θ 在 (y,z) 平面的 2x2 表示。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def forward(thetas: np.ndarray) -> dict:
    """给定 (θ2, θ4, θ5, θ3, θ1, θ6) 返回各关键点。"""
    t2, t4, t5, t3, t1, t6 = thetas
    p_l5 = rot(t2) @ A2
    p_l3 = p_l5 + rot(t2 + t5) @ A3
    p_l1 = p_l5 + rot(t2 + t5) @ A1
    p_l6 = p_l1 + rot(t2 + t5 + t1) @ A6
    p_f = p_l6 + rot(t2 + t5 + t1 + t6) @ AF
    p_p4 = rot(t4) @ B4
    p_q3 = p_l3 + rot(t2 + t5 + t3) @ P3
    p_q6 = p_l6 + rot(t2 + t5 + t1 + t6) @ Q6
    p_q2 = rot(t2) @ Q2
    return {"L5": p_l5, "L3": p_l3, "L1": p_l1, "L6": p_l6, "F": p_f,
            "P4": p_p4, "Q3": p_q3, "Q6": p_q6, "Q2": p_q2}


def kinematic_equations(thetas: np.ndarray, x_target: float, L_target: float) -> np.ndarray:
    pts = forward(thetas)
    return np.concatenate([
        pts["Q3"] - pts["P4"],
        pts["Q6"] - pts["Q2"],
        pts["F"] - np.array([x_target, -L_target]),
    ])


def inverse_kinematics(x_target: float, L_target: float,
                       seeds: list[np.ndarray] | None = None) -> list[dict]:
    """求 (θ2,θ4,…) 使轮心落在 (x_target, -L_target)。返回若干解。"""
    if seeds is None:
        rng = np.random.default_rng(0)
        seeds = [rng.uniform(-1.5, 1.5, 6) for _ in range(24)]
    solutions = []
    for s in seeds:
        try:
            res = least_squares(kinematic_equations, s, args=(x_target, L_target),
                                method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                max_nfev=4000)
        except Exception:
            continue
        if res.cost > 1e-12:
            continue
        sol = res.x
        if any(np.max(np.abs(sol - o["thetas"])) < 1e-4 for o in solutions):
            continue
        solutions.append({"thetas": sol, "cost": float(res.cost)})
    # 去重（按 θ2、θ4 排序）
    solutions.sort(key=lambda d: (round(d["thetas"][0], 4), round(d["thetas"][1], 4)))
    return solutions


def nominal_angles(env) -> np.ndarray:
    """从当前 qpos 读出六元角。"""
    m, d = env.sim.model, env.sim.data
    # 关节顺序: L2, L5, L3, L1, L6, wheel (每个 hinge 相对父体)
    jnames = ["left_J_chassis_link2", "left_J_link2_link5", "left_J_link5_link3",
              "left_J_link5_link1", "left_J_link1_link6"]
    import mujoco
    vals = []
    for n in jnames:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        vals.append(float(d.qpos[m.jnt_qposadr[jid]]))
    # forward() 的角度约定: θ2, θ4, θ5, θ3, θ1, θ6
    return np.array([vals[0], 0.0, vals[1], vals[2], vals[3], vals[4]])


if __name__ == "__main__":
    from uz05.env import UZ05Env

    env = UZ05Env(stage="stand", stand_level=2, seed=0, init_scale=0.0)
    env.params.domain_randomization.enabled = False
    env.reset(seed=0)
    th = nominal_angles(env)
    pts = forward(th)
    print("nominal θ:", np.round(th, 5))
    print("FK 轮心 (y,z):", np.round(pts["F"], 5),
          " ⇒ 腿长", round(float(np.linalg.norm(pts["F"])), 5))
    print("实测腿长:", np.round(env.sim.leg_lengths(), 5))
    print("闭环误差 Q3-P4:", np.round(pts["Q3"] - pts["P4"], 6),
          " Q6-Q2:", np.round(pts["Q6"] - pts["Q2"], 6))

    print("\n=== 解 IK：目标腿长 0.15 / 0.20 / 0.25 / 0.30 / 0.35（x=0）===")
    for L in (0.15, 0.20, 0.25, 0.30, 0.35):
        sols = inverse_kinematics(0.0, L, seeds=[th] + [
            np.random.default_rng(i).uniform(-1.6, 1.6, 6) for i in range(30)])
        if not sols:
            print(f"  L={L:.2f}: 无解")
            continue
        print(f"  L={L:.2f}: {len(sols)} 解")
        for s in sols[:4]:
            t = s["thetas"]
            print(f"      θ2={t[0]:+.4f} θ4={t[1]:+.4f} "
                  f"(θ5={t[2]:+.3f} θ3={t[3]:+.3f} θ1={t[4]:+.3f} θ6={t[5]:+.3f})")
    env.close()
