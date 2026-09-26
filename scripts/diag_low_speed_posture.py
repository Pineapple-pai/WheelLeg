"""Fixed-seed phase diagnostic for low-speed translation checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from train_uz05 import AsymmetricActorCriticPolicy
from uz05.env import UZ05Env
from uz05.policy_compat import ensure_forward_positive_wheel_actions


PHASES = ((0, 32, "startup_0_32"), (32, 64, "transition_32_64"), (64, 128, "cruise_64_128"))


def _stats(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        return {"steps": 0}
    pitch = np.asarray([row["abs_pitch_deg"] for row in rows])
    pitch_rate = np.asarray([row["abs_pitch_rate_rad_s"] for row in rows])
    return {
        "steps": len(rows),
        "abs_pitch_p95_deg": float(np.quantile(pitch, 0.95)),
        "abs_pitch_rate_p95_rad_s": float(np.quantile(pitch_rate, 0.95)),
        "penalty_mean": float(np.mean([row["posture_penalty"] for row in rows])),
        "body_vx_mean_m_s": float(np.mean([row["body_vx"] for row in rows])),
        "command_vx_mean_m_s": float(np.mean([row["command_vx"] for row in rows])),
    }


def evaluate(
    label: str, checkpoint: str, seeds: list[int], posture_gate_gain: float
) -> dict[str, object]:
    model = PPO.load(
        checkpoint,
        custom_objects={"policy_class": AsymmetricActorCriticPolicy},
        device="cpu",
    )
    if ensure_forward_positive_wheel_actions(model):
        print(f"[{label}] migrated legacy wheel-action sign")

    phase_rows = {name: [] for _, _, name in PHASES}
    pitch_all, rate_all, vx_error_all, survived = [], [], [], 0
    episodes = []
    for seed in seeds:
        env = UZ05Env(
            stage="low_speed",
            stand_level=2,
            seed=seed,
            init_scale=0.2,
            vx_range_override=(0.08, 0.12),
            zero_command_prob_override=0.25,
            reverse_prob_override=0.0,
            deployment_mode=True,
            episode_steps_override=128,
            low_speed_posture_gate_gain=posture_gate_gain,
        )
        obs, _ = env.reset(seed=seed)
        done = False
        info = {}
        for step in range(128):
            action = model.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            target = float(info.get("command_target_vx", 0.0))
            body_vx = float(info.get("body_vx_after", 0.0))
            pitch = abs(float(info.get("pitch", 0.0)))
            pitch_rate = abs(float(info.get("pitch_rate", 0.0)))
            pitch_all.append(np.degrees(pitch))
            rate_all.append(pitch_rate)
            if abs(target) > 0.01:
                vx_error_all.append(abs(body_vx - float(info.get("command_vx", target))))
                for start, end, name in PHASES:
                    if start <= step < end:
                        terms = info.get("reward_terms") or {}
                        phase_rows[name].append({
                            "abs_pitch_deg": float(np.degrees(pitch)),
                            "abs_pitch_rate_rad_s": pitch_rate,
                            "posture_penalty": float(terms.get("low_speed_stable_motion", 0.0)),
                            "body_vx": body_vx,
                            "command_vx": float(info.get("command_vx", target)),
                        })
            if done:
                break
        survived += int(bool(info.get("termination_reason") == "time_limit"))
        episodes.append({
            "seed": seed,
            "steps": int(info.get("episode_steps", step + 1)),
            "termination": str(info.get("termination_reason", "unknown")),
        })
        env.close()

    return {
        "label": label,
        "checkpoint": str(Path(checkpoint)),
        "episodes": len(seeds),
        "survive_rate": survived / max(len(seeds), 1),
        "all_step_abs_pitch_p95_deg": float(np.quantile(pitch_all, 0.95)),
        "all_step_abs_pitch_rate_p95_rad_s": float(np.quantile(rate_all, 0.95)),
        "moving_vx_mae_m_s": float(np.mean(vx_error_all)) if vx_error_all else float("nan"),
        "phases": {name: _stats(rows) for name, rows in phase_rows.items()},
        "episode_terminations": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-label", default="v10")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2400)
    parser.add_argument("--posture-gate-gain", type=float, default=1.0)
    args = parser.parse_args()
    seeds = list(range(args.seed, args.seed + args.episodes))
    for label, checkpoint in ((args.baseline_label, args.v10), ("candidate", args.candidate)):
        print(__import__("json").dumps(
            evaluate(label, checkpoint, seeds, args.posture_gate_gain), indent=2
        ))


if __name__ == "__main__":
    main()
