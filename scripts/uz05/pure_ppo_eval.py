"""Deterministic evaluation helpers shared by training and standalone eval."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np

from .pure_ppo_stand import (
    EPISODE_STEPS,
    PITCH_INIT_LIMIT_RAD,
    PITCH_RATE_INIT_LIMIT_RAD_S,
    PurePPOStandEnv,
)


EVAL_MODES = ("id", "pitch_5deg", "rate_ood", "midrun_impulse")
PITCH_RMS_LIMIT_RAD = np.deg2rad(3.0)
PITCH_PEAK_LIMIT_RAD = np.deg2rad(10.0)
DRIFT_LIMIT_M = 0.10
CONTACT_FRACTION_MIN = 0.95
SATURATION_FRACTION_MAX = 0.01
SATURATION_CONSECUTIVE_MAX_STEPS = 3
MIDRUN_PITCH_RATE_IMPULSE_RAD_S = 0.20


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt((p * (1.0 - p) / total) + z * z / (4.0 * total * total)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def initial_state_options(mode: str) -> dict[str, Any]:
    if mode == "id" or mode == "midrun_impulse":
        return {}
    if mode == "pitch_5deg":
        return {"pitch_bounds_rad": (-np.deg2rad(5.0), np.deg2rad(5.0))}
    if mode == "rate_ood":
        return {
            "pitch_rate_bounds_rad_s": (-0.25, 0.25),
        }
    raise ValueError(f"unknown evaluation mode {mode!r}; expected one of {EVAL_MODES}")


def episode_passes_acceptance(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not (
        metrics["steps"] == EPISODE_STEPS
        and metrics["truncated"]
        and not metrics["terminated"]
    ):
        failures.append("incomplete_8s_survival")
    if metrics["pitch_rms_rad"] > PITCH_RMS_LIMIT_RAD:
        failures.append("pitch_rms")
    if metrics["pitch_peak_rad"] > PITCH_PEAK_LIMIT_RAD:
        failures.append("pitch_peak")
    if metrics["drift_max_m"] > DRIFT_LIMIT_M:
        failures.append("planar_drift")
    if metrics["wheel_left_contact_fraction"] < CONTACT_FRACTION_MIN:
        failures.append("left_wheel_contact")
    if metrics["wheel_right_contact_fraction"] < CONTACT_FRACTION_MIN:
        failures.append("right_wheel_contact")
    if metrics["body_contact_steps"] != 0:
        failures.append("body_contact")
    if max(metrics["wheel_saturation_fraction"], default=0.0) > SATURATION_FRACTION_MAX:
        failures.append("wheel_saturation_fraction")
    if max(metrics["wheel_saturation_max_consecutive_steps"], default=0) > SATURATION_CONSECUTIVE_MAX_STEPS:
        failures.append("wheel_saturation_duration")
    if max(metrics["leg_torque_saturation_fraction"], default=0.0) > SATURATION_FRACTION_MAX:
        failures.append("leg_torque_saturation_fraction")
    if max(metrics["leg_torque_saturation_max_consecutive_steps"], default=0) > SATURATION_CONSECUTIVE_MAX_STEPS:
        failures.append("leg_torque_saturation_duration")
    return not failures, failures


def evaluate_episode(
    policy: Any,
    *,
    seed: int,
    mode: str = "id",
    env: PurePPOStandEnv | None = None,
    env_factory: Callable[[int], PurePPOStandEnv] = PurePPOStandEnv,
) -> dict[str, Any]:
    if mode not in EVAL_MODES:
        raise ValueError(f"unknown evaluation mode {mode!r}")
    owned_env = env is None
    env = env or env_factory(seed)
    options = initial_state_options(mode)
    observation, reset_info = env.reset(seed=seed, options=options)
    impulse = 0.0
    done_metrics = None
    for step_index in range(EPISODE_STEPS):
        if mode == "midrun_impulse" and step_index == EPISODE_STEPS // 2:
            impulse = (
                MIDRUN_PITCH_RATE_IMPULSE_RAD_S
                if seed % 2 == 0
                else -MIDRUN_PITCH_RATE_IMPULSE_RAD_S
            )
            env.apply_pitch_rate_perturbation(impulse)
            observation = env.refresh_observation()
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done_metrics = info["episode_metrics"]
            break
    if done_metrics is None:
        raise RuntimeError("evaluation episode did not terminate or truncate within 8 seconds")
    passed, failures = episode_passes_acceptance(done_metrics) if mode == "id" else (None, [])
    row = {
        "seed": int(seed),
        "mode": mode,
        "initial_pitch_rad": reset_info["initial_pitch_rad"],
        "initial_pitch_rate_rad_s": reset_info["initial_pitch_rate_rad_s"],
        "midrun_pitch_rate_impulse_rad_s": impulse,
        "accepted_episode": passed,
        "acceptance_failures": failures,
        **done_metrics,
    }
    if owned_env:
        env.close()
    return row


def summarize_evaluation(episodes: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one evaluation episode is required")
    accepted = [row for row in episodes if row.get("accepted_episode") is True]
    success_ci = wilson_interval(len(accepted), len(episodes)) if mode == "id" else None
    reasons = Counter(row["termination_reason"] for row in episodes)
    pitch_rms_deg = np.rad2deg([row["pitch_rms_rad"] for row in episodes])
    pitch_peak_deg = np.rad2deg([row["pitch_peak_rad"] for row in episodes])
    roll_rms_deg = np.rad2deg([row["roll_rms_rad"] for row in episodes])
    drift = np.asarray([row["drift_max_m"] for row in episodes], dtype=np.float64)
    left_contact = np.asarray(
        [row["wheel_left_contact_fraction"] for row in episodes], dtype=np.float64
    )
    right_contact = np.asarray(
        [row["wheel_right_contact_fraction"] for row in episodes], dtype=np.float64
    )
    summary: dict[str, Any] = {
        "mode": mode,
        "episodes": len(episodes),
        "termination_reasons": dict(reasons),
        "pitch_rms_deg_mean": float(np.mean(pitch_rms_deg)),
        "pitch_rms_deg_p95": float(np.percentile(pitch_rms_deg, 95)),
        "pitch_peak_deg_mean": float(np.mean(pitch_peak_deg)),
        "pitch_peak_deg_p95": float(np.percentile(pitch_peak_deg, 95)),
        "pitch_peak_deg_max": float(np.max(pitch_peak_deg)),
        "roll_rms_deg_mean": float(np.mean(roll_rms_deg)),
        "planar_drift_m_mean": float(np.mean(drift)),
        "planar_drift_m_p95": float(np.percentile(drift, 95)),
        "planar_drift_m_max": float(np.max(drift)),
        "wheel_left_contact_fraction_mean": float(np.mean(left_contact)),
        "wheel_right_contact_fraction_mean": float(np.mean(right_contact)),
        "wheel_saturation_fraction_max": float(
            max(max(row["wheel_saturation_fraction"], default=0.0) for row in episodes)
        ),
        "wheel_saturation_max_consecutive_steps": int(
            max(max(row["wheel_saturation_max_consecutive_steps"], default=0) for row in episodes)
        ),
        "body_contact_episodes": int(sum(row["body_contact_steps"] > 0 for row in episodes)),
        "accepted_episodes": len(accepted) if mode == "id" else None,
        "acceptance_rate": len(accepted) / len(episodes) if mode == "id" else None,
        "wilson_95_ci": list(success_ci) if success_ci is not None else None,
    }
    if mode == "id":
        bounds_deg = [(-3.0, -1.0), (-1.0, 1.0), (1.0, 3.0)]
        bins = {}
        for low, high in bounds_deg:
            subset = [
                row
                for row in episodes
                if low <= np.rad2deg(row["initial_pitch_rad"]) < high
                or (high == 3.0 and np.isclose(np.rad2deg(row["initial_pitch_rad"]), high))
            ]
            bins[f"{low:g}_to_{high:g}_deg"] = {
                "episodes": len(subset),
                "accepted": int(sum(row["accepted_episode"] for row in subset)),
                "acceptance_rate": (
                    float(np.mean([row["accepted_episode"] for row in subset])) if subset else None
                ),
            }
        summary["pitch_symmetry_bins"] = bins
    return summary
