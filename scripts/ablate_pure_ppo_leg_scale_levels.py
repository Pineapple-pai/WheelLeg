#!/usr/bin/env python3
"""Paired inference-only leg-scale sweep for a 0.35 rad ablation checkpoint.

This diagnostic reuses one frozen actor and test seed set. It never creates a
trained checkpoint and cannot qualify for export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from uz05 import pure_ppo_stand as task


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "pure_ppo_stand_leg_scale_035_v1"
TEST_SEED_BASE = {0: 23_000, 1: 23_100, 2: 23_200}
DIAGNOSTIC_SCALES = (0.25, 0.30)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 100:
        raise ValueError("paired ID diagnostic requires at least 100 episodes")

    task.LEG_POSITION_SCALE_RAD = 0.35
    task.TASK_ID = EXPERIMENT_ID
    from uz05 import pure_ppo_contract as contract

    contract.CONTRACT_VERSION = EXPERIMENT_ID
    from uz05.pure_ppo_eval import evaluate_episode, summarize_evaluation
    from uz05.pure_ppo_stand import ACTOR_OBS_DIM, ACTION_DIM, OBS_DIM

    checkpoint = args.checkpoint.resolve()
    contract_path = checkpoint.with_name(f"{checkpoint.stem}_contract.json")
    saved_contract = contract.read_and_require_contract(contract_path)
    if int(saved_contract["seed"]) != args.training_seed:
        raise ValueError("checkpoint contract seed does not match --training-seed")
    model = PPO.load(str(checkpoint), device="cpu")
    if tuple(model.observation_space.shape) != (OBS_DIM,):
        raise ValueError("checkpoint observation shape mismatch")
    if tuple(model.action_space.shape) != (ACTION_DIM,):
        raise ValueError("checkpoint action shape mismatch")
    if int(getattr(model.policy, "_actor_dim", -1)) != ACTOR_OBS_DIM:
        raise ValueError("checkpoint actor observation layout mismatch")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for scale in (0.35, *DIAGNOSTIC_SCALES):
        task.LEG_POSITION_SCALE_RAD = scale
        rows = [
            evaluate_episode(
                model,
                seed=TEST_SEED_BASE[args.training_seed] + offset,
                mode="id",
            )
            for offset in range(args.episodes)
        ]
        summary = summarize_evaluation(rows, "id")
        summary["leg_torque_saturation_fraction_mean_by_joint"] = np.mean(
            [row["leg_torque_saturation_fraction"] for row in rows], axis=0
        ).tolist()
        summary["wheel_current_saturation_fraction_mean_by_wheel"] = np.mean(
            [row["wheel_current_saturation_fraction"] for row in rows], axis=0
        ).tolist()
        summary["inference_only_counterfactual"] = scale != 0.35
        summary["export_eligible"] = False
        summaries[f"leg_scale_{scale:.2f}"] = summary
        scale_dir = output_dir / f"leg_scale_{scale:.2f}"
        scale_dir.mkdir()
        with (scale_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (scale_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"scale={scale:.2f}: {summary['accepted_episodes']}/{summary['episodes']} "
            f"accepted; drift p95={summary['planar_drift_m_p95']:.5f} m; "
            f"wheel saturation max={summary['wheel_saturation_fraction_max']:.3f}",
            flush=True,
        )

    report = {
        "status": "DIAGNOSTIC_ONLY",
        "task_id": EXPERIMENT_ID,
        "training_seed": args.training_seed,
        "independent_test_seed_start": TEST_SEED_BASE[args.training_seed],
        "episodes_per_scale": args.episodes,
        "fixed_checkpoint_sha256": _sha256(checkpoint),
        "checkpoint": str(checkpoint),
        "contract_sha256": _sha256(contract_path),
        "entrypoint_sha256": _sha256(Path(__file__).resolve()),
        "scales_rad": [0.35, *DIAGNOSTIC_SCALES],
        "inference_only_counterfactual": True,
        "export_eligible": False,
        "summaries": summaries,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
