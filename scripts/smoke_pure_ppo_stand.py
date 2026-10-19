#!/usr/bin/env python3
"""Run the required 1,000-step random-action and SB3 VecEnv smoke tests."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from uz05.pure_ppo_contract import model_xml_sha256, source_fingerprint
from uz05.pure_ppo_stand import ACTION_DIM, OBS_DIM, PurePPOStandEnv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "pure_ppo_stand_v1" / "smoke.json"


def _make(seed: int):
    return lambda: PurePPOStandEnv(seed=seed)


def _reward_calibration(mode: str, *, seed: int, steps: int = 2000) -> dict[str, Any]:
    env = PurePPOStandEnv(seed=seed)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    rewards = []
    raw: dict[str, list[float]] = {}
    weighted: dict[str, list[float]] = {}
    reasons: Counter[str] = Counter()
    resets = 0
    for _ in range(steps):
        action = (
            np.zeros(ACTION_DIM, dtype=np.float32)
            if mode == "zero_action"
            else rng.normal(0.0, np.exp(-1.5), ACTION_DIM).clip(-1.0, 1.0).astype(np.float32)
        )
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        for key, value in info["reward_raw"].items():
            raw.setdefault(key, []).append(float(value))
        for key, value in info["reward_components"].items():
            weighted.setdefault(key, []).append(float(value))
        if terminated or truncated:
            reasons[info["episode_metrics"]["termination_reason"]] += 1
            obs, reset_info = env.reset()
            resets += 1
            if reset_info["domain_randomization_enabled"] or reset_info["observation_noise_enabled"]:
                raise AssertionError("reward calibration unexpectedly enabled noise or DR")
    env.close()

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }

    return {
        "action_distribution": (
            "all zeros" if mode == "zero_action" else "independent Gaussian N(0, exp(-1.5)^2), clipped to [-1,1]"
        ),
        "transitions": steps,
        "resets": resets,
        "termination_reasons": dict(reasons),
        "reward_total": summarize(rewards),
        "reward_raw": {key: summarize(values) for key, values in raw.items()},
        "reward_weighted": {key: summarize(values) for key, values in weighted.items()},
    }


def run() -> dict:
    start = time.perf_counter()
    rng = np.random.default_rng(7001)
    env = PurePPOStandEnv(seed=7001)
    check_env(env, skip_render_check=True)
    obs, reset_info = env.reset(seed=7001)
    reasons: Counter[str] = Counter()
    completed_lengths: list[int] = []
    action_saturation_steps = np.zeros(ACTION_DIM, dtype=np.int64)
    action_abs_sum = np.zeros(ACTION_DIM, dtype=np.float64)
    torque_peak = np.zeros(6, dtype=np.float64)
    reward_values: list[float] = []
    reward_raw_values: dict[str, list[float]] = {}
    reward_component_values: dict[str, list[float]] = {}
    transitions = 1000
    resets = 0
    max_equality_residual = float(reset_info["reset_equality_residual_max_m"])

    for step in range(transitions):
        action = rng.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if next_obs.shape != (OBS_DIM,) or not np.isfinite(next_obs).all():
            raise AssertionError(f"step {step}: invalid observation shape/value")
        if not np.isfinite(reward) or not info["sim_finite"]:
            raise AssertionError(f"step {step}: nonfinite reward or MuJoCo state")
        if not np.allclose(info["action_applied"], np.clip(action, -1, 1), atol=1e-7):
            raise AssertionError(f"step {step}: action was replaced or reordered")
        if np.asarray(info["wheel_target_rad_s"]).shape != (2,):
            raise AssertionError(f"step {step}: wheel target has wrong shape")
        if np.asarray(info["joint_target_rad"]).shape != (4,):
            raise AssertionError(f"step {step}: leg target has wrong shape")
        if np.asarray(info["wheel_current_a"]).shape != (2,):
            raise AssertionError(f"step {step}: wheel current has wrong shape")
        if np.asarray(info["wheel_torque_nm"]).shape != (2,):
            raise AssertionError(f"step {step}: wheel torque has wrong shape")
        if np.asarray(info["joint_torque_nm"]).shape != (4,):
            raise AssertionError(f"step {step}: leg torque has wrong shape")
        action_saturation_steps += np.abs(action) >= 0.99
        action_abs_sum += np.abs(action)
        torque_peak[:4] = np.maximum(torque_peak[:4], np.abs(info["joint_torque_nm"]))
        torque_peak[4:] = np.maximum(torque_peak[4:], np.abs(info["wheel_torque_nm"]))
        reward_values.append(float(reward))
        for key, value in info["reward_raw"].items():
            reward_raw_values.setdefault(key, []).append(float(value))
        for key, value in info["reward_components"].items():
            reward_component_values.setdefault(key, []).append(float(value))
        obs = next_obs
        if terminated or truncated:
            episode = info["episode_metrics"]
            reasons[episode["termination_reason"]] += 1
            completed_lengths.append(int(episode["steps"]))
            obs, reset_info = env.reset()
            resets += 1
            max_equality_residual = max(
                max_equality_residual,
                float(reset_info["reset_equality_residual_max_m"]),
            )
            if reset_info["domain_randomization_enabled"] or reset_info["observation_noise_enabled"]:
                raise AssertionError("training disabled-noise/DR contract changed on reset")

    env.close()

    # One policy-compatible fixed action must yield the same coordinate targets
    # at opposite pitch observations; only the motor feedback may react to state.
    probes = []
    probe_action = np.asarray([0.1, -0.1, 0.2, -0.2, 0.3, -0.3], dtype=np.float64)
    for pitch in (-np.deg2rad(5.0), np.deg2rad(5.0)):
        probe = PurePPOStandEnv(seed=812)
        probe.reset(seed=812, options={"initial_pitch_rad": pitch, "initial_pitch_rate_rad_s": 0.0})
        _, _, _, _, info = probe.step(probe_action)
        probes.append({
            "pitch_rad": pitch,
            "wheel_target_rad_s": info["wheel_target_rad_s"],
            "joint_target_rad": info["joint_target_rad"],
            "wheel_current_a": info["wheel_current_a"],
        })
        probe.close()
    if probes[0]["wheel_target_rad_s"] != probes[1]["wheel_target_rad_s"]:
        raise AssertionError("wheel coordinate targets changed with pitch")
    if probes[0]["joint_target_rad"] != probes[1]["joint_target_rad"]:
        raise AssertionError("leg coordinate targets changed with pitch")

    # Exercise SB3's vector wrapper, including its autoreset boundary behavior.
    vec = VecMonitor(DummyVecEnv([_make(31), _make(32)]))
    batch_obs = vec.reset()
    if batch_obs.shape != (2, OBS_DIM) or not np.isfinite(batch_obs).all():
        raise AssertionError("SB3 VecEnv reset contract failed")
    batch_actions = np.stack([rng.uniform(-1, 1, ACTION_DIM) for _ in range(2)]).astype(np.float32)
    batch_obs, batch_rewards, batch_dones, batch_infos = vec.step(batch_actions)
    if batch_obs.shape != (2, OBS_DIM) or not np.isfinite(batch_obs).all():
        raise AssertionError("SB3 VecEnv step observation contract failed")
    if not np.isfinite(batch_rewards).all() or batch_dones.shape != (2,):
        raise AssertionError("SB3 VecEnv reward/done contract failed")
    vec.close()

    reward_calibration = {
        "zero_action": _reward_calibration("zero_action", seed=551, steps=2000),
        "initial_policy_scale_gaussian": _reward_calibration("gaussian_init", seed=552, steps=2000),
    }

    if max_equality_residual > 1e-8:
        raise AssertionError(f"reset linkage residual too high: {max_equality_residual}")
    if "numerical_failure" in reasons:
        raise AssertionError("random-action smoke encountered a numerical failure")
    return {
        "status": "PASS",
        "model_xml_sha256": model_xml_sha256(),
        "source_fingerprint_sha256": source_fingerprint(),
        "transitions": transitions,
        "resets": resets,
        "completed_episode_lengths_policy_steps": completed_lengths,
        "termination_reasons": dict(reasons),
        "action_abs_mean": (action_abs_sum / transitions).tolist(),
        "action_saturation_fraction": (action_saturation_steps / transitions).tolist(),
        "peak_torque_nm_joint_then_wheel": torque_peak.tolist(),
        "reward_total_mean_p05_p95": {
            "mean": float(np.mean(reward_values)),
            "p05": float(np.percentile(reward_values, 5)),
            "p95": float(np.percentile(reward_values, 95)),
        },
        "reward_raw_mean_p05_p95": {
            key: {
                "mean": float(np.mean(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
            for key, values in reward_raw_values.items()
        },
        "reward_weighted_mean_p05_p95": {
            key: {
                "mean": float(np.mean(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
            }
            for key, values in reward_component_values.items()
        },
        "offline_reward_calibration": reward_calibration,
        "max_reset_equality_residual_m": max_equality_residual,
        "opposite_pitch_fixed_action_probes": probes,
        "sb3_vecenv_batch_shape": list(batch_obs.shape),
        "wall_seconds": time.perf_counter() - start,
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run()
    except Exception as exc:
        report = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "model_xml_sha256": model_xml_sha256(),
            "source_fingerprint_sha256": source_fingerprint(),
        }
        print(report["error"], file=sys.stderr)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
