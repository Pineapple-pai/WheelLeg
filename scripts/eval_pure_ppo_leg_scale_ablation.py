#!/usr/bin/env python3
"""Evaluate a fresh checkpoint trained with the fixed 0.35 rad leg scale."""

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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    task.LEG_POSITION_SCALE_RAD = 0.35
    task.TASK_ID = EXPERIMENT_ID
    from uz05 import pure_ppo_contract as contract

    contract.CONTRACT_VERSION = EXPERIMENT_ID
    from uz05.pure_ppo_eval import EVAL_MODES, evaluate_episode, summarize_evaluation
    from uz05.pure_ppo_stand import ACTOR_OBS_DIM, ACTION_DIM, OBS_DIM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--mode", choices=(*EVAL_MODES, "all"), default="all")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    contract_path = checkpoint.with_name(f"{checkpoint.stem}_contract.json")
    saved_contract = contract.read_and_require_contract(contract_path)
    if int(saved_contract["seed"]) != args.training_seed:
        raise ValueError("checkpoint contract seed does not match --training-seed")
    if args.episodes < 1 or (args.mode in ("id", "all") and args.episodes < 100):
        raise ValueError("ID evaluation requires at least 100 episodes")
    model = PPO.load(str(checkpoint), device="cpu")
    if tuple(model.observation_space.shape) != (OBS_DIM,):
        raise ValueError("checkpoint full observation shape mismatch")
    if tuple(model.action_space.shape) != (ACTION_DIM,):
        raise ValueError("checkpoint action shape mismatch")
    if int(getattr(model.policy, "_actor_dim", -1)) != ACTOR_OBS_DIM:
        raise ValueError("checkpoint actor observation layout mismatch")

    modes = EVAL_MODES if args.mode == "all" else (args.mode,)
    output_dir = args.output_dir or (
        ROOT / "runs" / EXPERIMENT_ID / f"seed{args.training_seed}" / f"test_{args.mode}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries = {}
    all_pass = True
    for mode in modes:
        rows = [
            evaluate_episode(
                model,
                seed=TEST_SEED_BASE[args.training_seed] + offset,
                mode=mode,
            )
            for offset in range(args.episodes)
        ]
        summary = summarize_evaluation(rows, mode)
        summary["leg_torque_saturation_fraction_mean_by_joint"] = np.mean(
            [row["leg_torque_saturation_fraction"] for row in rows], axis=0
        ).tolist()
        summary["wheel_current_saturation_fraction_mean_by_wheel"] = np.mean(
            [row["wheel_current_saturation_fraction"] for row in rows], axis=0
        ).tolist()
        mode_pass = mode != "id" or float(summary["acceptance_rate"]) >= 0.95
        summary["status"] = "PASS" if mode_pass else "FAIL"
        summaries[mode] = summary
        all_pass &= mode_pass
        with (output_dir / f"{mode}_episodes.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (output_dir / f"{mode}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{mode}] {summary['accepted_episodes'] if mode == 'id' else 'n/a'}/"
            f"{summary['episodes']} accepted; pitch_rms_mean="
            f"{summary['pitch_rms_deg_mean']:.4f}deg; "
            f"drift_p95={summary['planar_drift_m_p95']:.5f}m",
            flush=True,
        )
    result = {
        "status": "PASS" if all_pass else "FAIL",
        "task_id": EXPERIMENT_ID,
        "action_protocol": saved_contract["action_protocol"],
        "leg_position_scale_rad": 0.35,
        "training_seed": args.training_seed,
        "mode": args.mode,
        "episodes_per_mode": args.episodes,
        "independent_test_seed_start": TEST_SEED_BASE[args.training_seed],
        "independent_test_seed_end_exclusive": TEST_SEED_BASE[args.training_seed] + args.episodes,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "contract_sha256": _sha256(contract_path),
        "source_fingerprint_sha256": saved_contract["source_fingerprint_sha256"],
        "evaluation_entrypoint_sha256": _sha256(Path(__file__).resolve()),
        "device": str(model.device),
        "summaries": summaries,
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
