"""测量 UZ-05 站立时的俯仰振荡 / 漂移，并做开环系统辨识。

用法::

    python scripts/diag_osc.py openloop      # 开环：零动作 / 常值电流 / 常值关节偏置
    python scripts/diag_osc.py policy        # 策略闭环：pitch/漂移/频谱
    python scripts/diag_osc.py geo           # 关节角 vs 轮心相对髋的位置

输出为机器可读的 key: value 行。
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from uz05.env import UZ05Env
from uz05.spec import ACTION_SLICES, EnvParams


def _rpy(sim) -> np.ndarray:
    q = sim.base_quat
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def make_env(stage="stand", stand_level=2, assist=0.0, seed=0, init_scale=0.0,
             lock_legs=True, dr_off=True):
    env = UZ05Env(stage=stage, stand_level=stand_level, assist_scale=assist,
                  seed=seed, init_scale=init_scale, lock_stand_leg_actions=lock_legs)
    if dr_off:
        env.params.domain_randomization.enabled = False
    return env


def rollout(env, action_fn, steps=1000):
    """跑一段，返回逐帧记录。action_fn(env, prev_info) -> action(6,)"""
    obs, _ = env.reset(seed=0)
    rec = {k: [] for k in (
        "t", "x", "y", "z", "roll", "pitch", "yaw", "pitch_rate", "roll_rate",
        "vx", "vy", "wheel_l", "wheel_r", "cur_l", "cur_r", "a0", "a1",
        "leg_torque", "leg_len_l", "leg_len_r", "hip0", "hip1", "contact_l",
        "contact_r", "reward",
    )}
    info: dict = {}
    for step in range(steps):
        action = action_fn(env, info, step)
        obs, reward, term, trunc, info = env.step(action)
        rpy = _rpy(env.sim)
        lin = env.sim.body_frame(env.sim.base_lin_vel_world)
        rec["t"].append(step * env.params.control_dt)
        rec["x"].append(float(env.sim.data.qpos[0]))
        rec["y"].append(float(env.sim.data.qpos[1]))
        rec["z"].append(float(env.sim.data.qpos[2]))
        rec["roll"].append(float(rpy[0]))
        rec["pitch"].append(float(rpy[1]))
        rec["yaw"].append(float(rpy[2]))
        rec["pitch_rate"].append(float(env.sim.base_ang_vel_world[1]))
        rec["roll_rate"].append(float(env.sim.base_ang_vel_world[0]))
        rec["vx"].append(float(lin[0]))
        rec["vy"].append(float(lin[1]))
        rec["wheel_l"].append(float(env.sim.wheel_velocities()[0]))
        rec["wheel_r"].append(float(env.sim.wheel_velocities()[1]))
        rec["cur_l"].append(float(info.get("wheel_current_left", 0.0)))
        rec["cur_r"].append(float(info.get("wheel_current_right", 0.0)))
        rec["a0"].append(float(action[0]))
        rec["a1"].append(float(action[1]))
        rec["leg_torque"].append(float(info.get("leg_torque_abs_mean", 0.0)))
        rec["leg_len_l"].append(float(info.get("leg_length_left", 0.0)))
        rec["leg_len_r"].append(float(info.get("leg_length_right", 0.0)))
        jp = env.sim.joint_positions()
        rec["hip0"].append(float(jp[0]))
        rec["hip1"].append(float(jp[1]))
        rec["contact_l"].append(float(info.get("wheel_force_left", 0.0)))
        rec["contact_r"].append(float(info.get("wheel_force_right", 0.0)))
        rec["reward"].append(float(reward))
        if term or trunc:
            break
    return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}, info


def spectral_peak(sig: np.ndarray, dt: float) -> tuple[float, float]:
    """返回 (峰值频率 Hz, 该频率占比)。用去均值 + Hann 窗的 rFFT。"""
    if sig.size < 32:
        return 0.0, 0.0
    x = sig - sig.mean()
    w = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(x.size, dt)
    spec[0] = 0.0
    total = spec.sum()
    if total <= 0:
        return 0.0, 0.0
    k = int(np.argmax(spec))
    return float(freqs[k]), float(spec[k] / total)


def report(tag: str, rec: dict, skip: int = 100) -> None:
    skip = min(skip, max(0, rec["pitch"].size - 2))
    pitch = rec["pitch"][skip:]
    prate = rec["pitch_rate"][skip:]
    x = rec["x"][skip:]
    x0 = rec["x"][0] if rec["x"].size else 0.0
    freq, frac = spectral_peak(pitch, 0.008)
    print(f"[{tag}] frames={rec['pitch'].size}/{int(skip)}")
    print(f"[{tag}] pitch_rms_deg={np.degrees(np.sqrt(np.mean(pitch ** 2))):.3f}")
    print(f"[{tag}] pitch_ptp_deg={np.degrees(pitch.max() - pitch.min()):.3f}")
    print(f"[{tag}] pitch_rate_rms={np.sqrt(np.mean(prate ** 2)):.4f}")
    print(f"[{tag}] pitch_peak_hz={freq:.2f}")
    print(f"[{tag}] pitch_peak_frac={frac:.3f}")
    print(f"[{tag}] drift_min_cm={100 * (x.min() - x0):.2f}")
    print(f"[{tag}] drift_max_cm={100 * (x.max() - x0):.2f}")
    print(f"[{tag}] drift_final_cm={100 * (x[-1] - x0):.2f}")
    # 末段 200 帧平均漂移（与验收口径一致）
    tail = rec["x"][-200:] - x0 if rec["x"].size >= 200 else x - x0
    print(f"[{tag}] drift_tail_mean_cm={100 * np.mean(np.abs(tail)):.2f}")
    print(f"[{tag}] vx_rms={np.sqrt(np.mean(rec['vx'][skip:] ** 2)):.4f}")
    print(f"[{tag}] wheel_vel_rms={np.sqrt(np.mean(rec['wheel_l'][skip:] ** 2)):.3f}")
    print(f"[{tag}] current_rms_A={np.sqrt(np.mean(rec['cur_l'][skip:] ** 2)):.3f}")
    print(f"[{tag}] current_max_A={np.abs(rec['cur_l']).max():.3f}")
    print(f"[{tag}] leg_torque_mean_Nm={rec['leg_torque'].mean():.3f}")
    print(f"[{tag}] roll_rms_deg={np.degrees(np.sqrt(np.mean(rec['roll'][skip:] ** 2))):.3f}")
    print(f"[{tag}] yaw_final_deg={np.degrees(rec['yaw'][-1]):.3f}")
    print(f"[{tag}] height_final={rec['z'][-1]:.4f}")
    print(f"[{tag}] contact_l_mean={rec['contact_l'].mean():.1f} contact_r_mean={rec['contact_r'].mean():.1f}")


# --------------------------------------------------------------------------
def cmd_openloop(args) -> None:
    """零动作 → 观察自然发散；常值电流 → 观察俯仰权限；常值关节偏置 → 观察质心移动。"""
    env = make_env(init_scale=0.0)
    print("# 零动作（自由落体式倒立摆）")
    rec, info = rollout(env, lambda e, i, s: np.zeros(6), steps=args.steps)
    report("zero_action", rec)
    print(f"[zero_action] term={info.get('termination_reason')}")
    env.close()

    for current in (2.0, 4.0, 8.0):
        env = make_env(init_scale=0.0)
        a = np.zeros(6)
        a[0] = current / env.params.wheel.policy_current_scale_a
        rec, info = rollout(env, lambda e, i, s, a=a: a, steps=200)
        p = np.concatenate([rec["pitch"], np.full(max(0, 200 - rec["pitch"].size), np.nan)])
        vx = rec["vx"]
        print(f"# 常值共模电流 {current} A: pitch@20 帧={np.degrees(p[20]):.4f} deg "
              f"pitch@100={np.degrees(p[100]):.4f} pitch@200={np.degrees(p[199]):.4f} "
              f"vx@200={vx[-1]:.4f} frames={rec['pitch'].size} term={info.get('termination_reason')}")
        env.close()

    # 关节偏置：4 个主动关节同时加同一个偏置，看 wheel 相对 hip 的位移方向
    for offset in (-0.15, -0.05, 0.05, 0.15):
        env = make_env(init_scale=0.0)
        a = np.zeros(6)
        a[2:] = offset
        rec, info = rollout(env, lambda e, i, s, a=a: a, steps=150)
        print(f"# 关节偏置 {offset:+.2f} rad (归一化): x@150={100 * (rec['x'][-1] - rec['x'][0]):.2f} cm "
              f"pitch@150={np.degrees(rec['pitch'][-1]):.3f} deg "
              f"z@150={rec['z'][-1]:.4f} leglen={rec['leg_len_l'][-1]:.4f} "
              f"hip0={rec['hip0'][-1]:+.4f}")
        env.close()


def cmd_geo(args) -> None:
    """几何：关节角 → 轮心相对髋的位置（机体系）。"""
    env = make_env(init_scale=0.0)
    sim = env.sim
    base = np.asarray(EnvParams().robot.pd_neutral_joint_pos, dtype=np.float64)
    print("# hip 关节偏置对 (轮心 - 髋site) 机体系位移与腿长的影响")
    for dl2 in (-0.3, -0.15, 0.0, 0.15, 0.3):
        for dl4 in (-0.3, 0.0, 0.3):
            q = base.copy()
            q[0] += dl2
            q[1] += dl4
            sim.data.qpos[sim.hip_qpos_adr] = q
            sim.forward()
            hip_l = sim.data.site_xpos[sim.hip_sites[0]]
            whl_l = sim.data.site_xpos[sim.wheel_sites[0]]
            rel_world = whl_l - hip_l
            rel_body = sim.body_frame(rel_world)
            print(f"dL2={dl2:+.2f} dL4={dl4:+.2f} rel_body=({rel_body[0]:+.4f},{rel_body[1]:+.4f},"
                  f"{rel_body[2]:+.4f}) leglen={np.linalg.norm(rel_world):.4f}")
    env.close()


def cmd_policy(args) -> None:
    from stable_baselines3 import PPO
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from train_uz05 import AsymmetricActorCriticPolicy
    model = PPO.load(args.checkpoint, custom_objects={
        "policy_class": AsymmetricActorCriticPolicy}, device="cpu")
    env = make_env(init_scale=0.0, lock_legs=not args.unlock_legs)
    rec, info = rollout(env, lambda e, i, s: model.predict(
        e._obs(), deterministic=True)[0], steps=args.steps)
    report("policy", rec)
    print(f"[policy] term={info.get('termination_reason')}")
    env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("openloop", "policy", "geo"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--checkpoint", default="checkpoints/ppo_stand_s2_pitchquiet_v1/checkpoint_iter_40")
    p.add_argument("--unlock-legs", action="store_true")
    args = p.parse_args()
    {"openloop": cmd_openloop, "policy": cmd_policy, "geo": cmd_geo}[args.mode](args)


if __name__ == "__main__":
    main()
