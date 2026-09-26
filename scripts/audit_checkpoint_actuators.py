#!/usr/bin/env python3
"""Audit actuator demand of a PPO checkpoint before hardware deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from train_uz05 import AsymmetricActorCriticPolicy
from uz05.env import UZ05Env
from uz05.spec import EnvParams


def stats(values: list[float], limit: float) -> dict[str, float]:
    data = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "mean": float(data.mean()),
        "p95": float(np.quantile(data, 0.95)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(data.max()),
        "limit": float(limit),
        "p99_utilization": float(np.quantile(data, 0.99) / limit),
        "saturation_rate": float(np.mean(data >= limit * 0.999)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage", default="stand")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--stand-level", type=int, default=2)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--deployment-mode", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    model = PPO.load(
        args.checkpoint,
        custom_objects={"policy_class": AsymmetricActorCriticPolicy},
        device="cpu",
    )
    params = EnvParams()
    leg = [[] for _ in range(4)]
    wheel = [[] for _ in range(2)]
    actions = [[] for _ in range(6)]
    quantization_error = []
    completed = 0
    termination_reasons: dict[str, int] = {}

    for episode in range(args.episodes):
        env = UZ05Env(
            stage=args.stage,
            seed=args.seed + episode,
            stand_level=args.stand_level,
            init_scale=args.init_scale,
            deployment_mode=args.deployment_mode,
        )
        obs, _ = env.reset(seed=args.seed + episode)
        done = False
        info = {}
        while not done:
            action = np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1)
            obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            for index, value in enumerate(action):
                actions[index].append(float(value))
            for index, value in enumerate(info["leg_torque_vector"]):
                leg[index].append(float(value))
            for index, value in enumerate(info["wheel_current_vector"]):
                wheel[index].append(float(value))
            quantization_error.append(
                float(info.get("mit_position_quantization_abs_max", 0.0))
            )
        completed += int(bool(truncated and not terminated))
        reason = str(info.get("termination_reason", "unknown"))
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        env.close()

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "stage": args.stage,
        "episodes": args.episodes,
        "deployment_mode": args.deployment_mode,
        "survive_rate": completed / args.episodes,
        "leg_torque_nm": {
            f"joint_{i}": stats(values, params.joint.torque_limit)
            for i, values in enumerate(leg)
        },
        "wheel_current_a": {
            side: stats(values, params.wheel.current_limit_a)
            for side, values in zip(("left", "right"), wheel)
        },
        "normalized_action": {
            f"action_{i}": stats(values, 1.0) for i, values in enumerate(actions)
        },
        "mit_position_quantization_abs_max_rad": float(max(quantization_error)),
        "termination_reasons": termination_reasons,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

