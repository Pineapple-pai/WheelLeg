#!/usr/bin/env python3
"""Run a paired inference-only action-scale counterfactual on fixed test seeds.

This is a diagnostic, not a trained policy: the checkpoint and all simulator
settings remain fixed while exactly one action mapping scale is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from uz05 import pure_ppo_stand as task
from uz05.pure_ppo_contract import read_and_require_contract
from uz05.pure_ppo_eval import evaluate_episode, summarize_evaluation


ROOT = Path(__file__).resolve().parents[1]
TEST_SEED_BASE = {0: 20_000, 1: 21_000, 2: 22_000}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--factor", choices=("leg_position_scale", "wheel_speed_scale"), required=True)
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    contract = read_and_require_contract(
        checkpoint.with_name(f"{checkpoint.stem}_contract.json")
    )
    if int(contract["seed"]) != args.training_seed:
        raise ValueError("checkpoint contract seed does not match --training-seed")
    if args.episodes < 100:
        raise ValueError("paired action-scale diagnosis requires at least 100 episodes")
    if args.factor == "leg_position_scale":
        variable = "LEG_POSITION_SCALE_RAD"
    else:
        variable = "WHEEL_SPEED_SCALE_RAD_S"
    baseline_value = float(getattr(task, variable))
    if not np.isfinite(args.value) or args.value <= 0 or args.value == baseline_value:
        raise ValueError("variant scale must be positive and differ from the locked value")

    # The contract is checked against the baseline before the single mapping
    # override; no source file, checkpoint weight, or other parameter changes.
    setattr(task, variable, float(args.value))
    model = PPO.load(str(checkpoint), device="cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    env = task.PurePPOStandEnv(seed=TEST_SEED_BASE[args.training_seed])
    rows = []
    try:
        for seed in range(
            TEST_SEED_BASE[args.training_seed],
            TEST_SEED_BASE[args.training_seed] + args.episodes,
        ):
            rows.append(evaluate_episode(model, seed=seed, mode="id", env=env))
    finally:
        env.close()

    summary = summarize_evaluation(rows, "id")
    summary["status"] = "PASS" if summary["acceptance_rate"] >= 0.95 else "FAIL"
    summary["leg_torque_saturation_fraction_mean_by_joint"] = np.mean(
        [row["leg_torque_saturation_fraction"] for row in rows], axis=0
    ).tolist()
    summary["wheel_current_saturation_fraction_mean_by_wheel"] = np.mean(
        [row["wheel_current_saturation_fraction"] for row in rows], axis=0
    ).tolist()
    summary["wheel_torque_saturation_fraction_mean_by_wheel"] = np.mean(
        [row["wheel_torque_saturation_fraction"] for row in rows], axis=0
    ).tolist()
    summary["leg_saturation_failure_episodes"] = sum(
        "leg_torque_saturation_fraction" in row["acceptance_failures"]
        or "leg_torque_saturation_duration" in row["acceptance_failures"]
        for row in rows
    )
    summary["wheel_saturation_failure_episodes"] = sum(
        any(name.startswith("wheel_saturation") for name in row["acceptance_failures"])
        for row in rows
    )
    baseline_path = (
        ROOT / "runs" / "pure_ppo_stand_v1" / f"seed{args.training_seed}"
        / "test_all" / "id_episodes.jsonl"
    )
    baseline_rows = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines()]
    baseline_stats = {
        "episodes": len(baseline_rows),
        "accepted_episodes": sum(row["accepted_episode"] is True for row in baseline_rows),
        "acceptance_rate": float(np.mean([row["accepted_episode"] is True for row in baseline_rows])),
        "leg_torque_saturation_fraction_mean_by_joint": np.mean(
            [row["leg_torque_saturation_fraction"] for row in baseline_rows], axis=0
        ).tolist(),
        "wheel_current_saturation_fraction_mean_by_wheel": np.mean(
            [row["wheel_current_saturation_fraction"] for row in baseline_rows], axis=0
        ).tolist(),
    }
    report = {
        "status": summary["status"],
        "experiment": "paired_inference_only_action_scale_counterfactual",
        "retrained": False,
        "single_factor": args.factor,
        "baseline_scale": baseline_value,
        "variant_scale": float(args.value),
        "training_seed": args.training_seed,
        "test_seed_start": TEST_SEED_BASE[args.training_seed],
        "episodes": args.episodes,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "baseline_contract_source_fingerprint": contract["source_fingerprint_sha256"],
        "paired_baseline_reference": baseline_stats,
        "acceptance_summary": summary,
        "interpretation_limit": "A changed mapping on weights trained for the baseline scale is a diagnostic counterfactual only; it is not a deployable or final Pure PPO checkpoint.",
    }
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if summary["acceptance_rate"] >= 0.95 else 2


if __name__ == "__main__":
    raise SystemExit(main())
