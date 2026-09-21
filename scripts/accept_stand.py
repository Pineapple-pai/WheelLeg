"""UZ-05 零指令站定验收：腿+轮协同控制器 vs 旧策略 checkpoint。

验收口径（与目标一致）：
  * 存活（跑满整集）
  * pitch 振荡：RMS 与峰峰值（"高频点头"的直接量）
  * pitch 速率 RMS（"点头"的快慢）
  * 漂移：全程峰值 / 末段均值（"漂移幅度很小"）
  * 轮电流颤振 dI（实机可执行性）
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.balance import CoordinatedBalanceParams  # noqa: E402
from uz05.env import UZ05Env  # noqa: E402

DT = 0.008


def _rpy(sim):
    q = sim.base_quat
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def rollout(steps=1000, seed=0, init_scale=0.0, dr=False, coord_mix=1.0,
            params: CoordinatedBalanceParams | None = None, action_fn=None,
            episode_steps: int | None = None):
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=init_scale, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=coord_mix,
                  coord_params=params)
    if not dr:
        env.params.domain_randomization.enabled = False
    if episode_steps is not None:
        env.stage = replace(env.stage, episode_steps=int(episode_steps))
    env.reset(seed=seed)
    rec = {k: [] for k in ("th", "w", "x", "y", "I", "u", "Ileg", "hip")}
    info: dict = {}
    n = 0
    for k in range(steps):
        action = np.zeros(6) if action_fn is None else action_fn(env, k)
        _, _, term, trunc, info = env.step(action)
        rpy = _rpy(env.sim)
        rec["th"].append(float(rpy[1]))
        rec["w"].append(float(env.sim.base_ang_vel_world[1]))
        rec["x"].append(float(env.sim.data.qpos[0] - env.nominal_xy[0]))
        rec["y"].append(float(env.sim.data.qpos[1]))
        rec["I"].append(float(info.get("wheel_current_left", 0.0)))
        rec["u"].append(float(env.previous_action[2]))
        rec["Ileg"].append(float(info.get("leg_torque_abs_mean", 0.0)))
        rec["hip"].append(float(env.sim.joint_positions()[0]))
        n = k + 1
        if term or trunc:
            break
    env.close()
    out = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    out["n"] = n
    out["term"] = info.get("termination_reason", "?")
    return out


def metrics(rec: dict, steps: int) -> dict:
    n = rec["n"]
    th = rec["th"]
    x = rec["x"]
    I = rec["I"]
    tail = slice(max(0, n - 200), n)
    sig = th - th.mean()
    sp = np.abs(np.fft.rfft(sig * np.hanning(n))) ** 2
    fr = np.fft.rfftfreq(n, DT)
    sp[0] = 0.0
    k = int(np.argmax(sp))
    hip = rec["hip"]
    return {
        "n": n,
        "survived": n >= steps,
        "term": rec["term"],
        "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(th ** 2)))),
        "pitch_ptp_deg": float(np.degrees(th.max() - th.min())),
        "pitch_tail_deg": float(np.degrees(np.mean(np.abs(th[tail])))),
        "pitch_rate_rms": float(np.sqrt(np.mean(rec["w"] ** 2))),
        "pitch_dom_hz": float(fr[k]),
        "drift_peak_cm": float(100 * np.abs(x).max()),
        "drift_tail_cm": float(100 * np.mean(np.abs(x[tail]))),
        "drift_final_cm": float(100 * x[n - 1]),
        "yaw_drift_cm": float(100 * np.abs(rec["y"] - rec["y"][0]).max()),
        "current_rms_A": float(np.sqrt(np.mean(I ** 2))),
        "current_dI_A": float(np.sqrt(np.mean(np.diff(I) ** 2))) if n > 2 else 0.0,
        "hip_offset_rms_rad": float(np.sqrt(np.mean((hip - hip.mean()) ** 2))),
        "leg_torque_mean_Nm": float(rec["Ileg"].mean()),
    }


def show(tag: str, m: dict) -> None:
    print(f"[{tag}] n={float(m['n']):.0f} survived={m['survived']:.2f} term={m['term']}")
    print(f"[{tag}] pitch_rms_deg={m['pitch_rms_deg']:.3f} "
          f"pitch_ptp_deg={m['pitch_ptp_deg']:.3f} pitch_tail_deg={m['pitch_tail_deg']:.3f}")
    print(f"[{tag}] pitch_rate_rms={m['pitch_rate_rms']:.4f} dom={m['pitch_dom_hz']:.2f}Hz")
    print(f"[{tag}] drift_peak_cm={m['drift_peak_cm']:.3f} drift_tail_cm={m['drift_tail_cm']:.3f} "
          f"drift_final_cm={m['drift_final_cm']:+.3f} yaw_peak_cm={m['yaw_drift_cm']:.3f}")
    print(f"[{tag}] current_rms_A={m['current_rms_A']:.3f} current_dI_A={m['current_dI_A']:.4f} "
          f"leg_torque_Nm={m['leg_torque_mean_Nm']:.2f} hip_ripple_rad={m['hip_offset_rms_rad']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--gains", default="", help="coord_gains 风格 JSON")
    ap.add_argument("--baseline", default="", help="旧策略 checkpoint（对照）")
    ap.add_argument("--episode-steps", type=int, default=0,
                    help="覆盖 episode 长度（0 = 用 stand 阶段默认 1000 步）")
    a = ap.parse_args()

    params = None
    if a.gains:
        params = CoordinatedBalanceParams(**json.loads(Path(a.gains).read_text()))
    print("=== 腿 + 轮协同平衡控制器 ===")
    for init in (0.0, 0.5, 1.0):
        for dr in (False, True):
            agg = []
            for ep in range(a.episodes):
                rec = rollout(steps=a.steps, seed=ep, init_scale=init, dr=dr, params=params,
                              episode_steps=a.episode_steps or None)
                agg.append(metrics(rec, a.steps))
            m = {k: float(np.mean([d[k] for d in agg])) for k in agg[0]
                 if isinstance(agg[0][k], (int, float)) and not isinstance(agg[0][k], bool)}
            m["survived"] = float(np.mean([d["survived"] for d in agg]))
            m["term"] = agg[0]["term"]
            show(f"coord init={init:.1f} dr={int(dr)}", m)

    if a.baseline:
        from stable_baselines3 import PPO
        from train_uz05 import AsymmetricActorCriticPolicy
        model = PPO.load(a.baseline, custom_objects={
            "policy_class": AsymmetricActorCriticPolicy}, device="cpu")
        print("=== 旧策略 checkpoint（对照） ===")
        for lock in (True, False):
            agg = []
            for ep in range(a.episodes):
                env_holder = {}

                def act(env, k, _lock=lock, _holder=env_holder):
                    obs = env._obs()
                    return model.predict(obs, deterministic=True)[0]

                rec = rollout(steps=a.steps, seed=ep, coord_mix=0.0, action_fn=act)
                agg.append(metrics(rec, a.steps))
            m = {k: float(np.mean([d[k] for d in agg])) for k in agg[0]
                 if isinstance(agg[0][k], (int, float)) and not isinstance(agg[0][k], bool)}
            m["survived"] = float(np.mean([d["survived"] for d in agg]))
            m["term"] = agg[0]["term"]
            show(f"policy lock_legs={lock}", m)


if __name__ == "__main__":
    main()
