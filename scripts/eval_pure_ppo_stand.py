#!/usr/bin/env python3
"""Evaluate a versioned Pure PPO stand checkpoint on fixed held-out seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from stable_baselines3 import PPO

from uz05.pure_ppo_contract import (
    read_and_require_contract,
    model_xml_sha256,
    source_fingerprint,
)
from uz05.pure_ppo_eval import EVAL_MODES, evaluate_episode, summarize_evaluation
from uz05.pure_ppo_stand import ACTOR_OBS_DIM, ACTION_DIM, OBS_DIM


ROOT = Path(__file__).resolve().parents[1]
TEST_SEED_BASE = {0: 20_000, 1: 21_000, 2: 22_000}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--mode", choices=(*EVAL_MODES, "all"), default="all")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    contract_path = args.contract or checkpoint.with_name(f"{checkpoint.stem}_contract.json")
    contract_path = contract_path.resolve()
    contract = read_and_require_contract(contract_path)
    if int(contract["seed"]) != args.training_seed:
        raise ValueError("checkpoint contract seed does not match --training-seed")
    if args.episodes < 1 or (args.mode in ("id", "all") and args.episodes < 100):
        raise ValueError("ID evaluation requires at least 100 independent episodes")

    model = PPO.load(str(checkpoint), device=args.device)
    if tuple(model.observation_space.shape) != (OBS_DIM,):
        raise ValueError(f"checkpoint full observation shape mismatch: {model.observation_space.shape}")
    if tuple(model.action_space.shape) != (ACTION_DIM,):
        raise ValueError(f"checkpoint action shape mismatch: {model.action_space.shape}")
    if int(getattr(model.policy, "_actor_dim", -1)) != ACTOR_OBS_DIM:
        raise ValueError("checkpoint actor input is not the current 38D actor layout")

    modes = EVAL_MODES if args.mode == "all" else (args.mode,)
    output_dir = args.output_dir or (
        ROOT / "runs" / "pure_ppo_stand_v1" / f"seed{args.training_seed}" / f"test_{args.mode}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries = {}
    all_pass = True
    for mode in modes:
        rows = []
        base = TEST_SEED_BASE[args.training_seed]
        for seed in range(base, base + args.episodes):
            rows.append(evaluate_episode(model, seed=seed, mode=mode))
        summary = summarize_evaluation(rows, mode)
        mode_pass = mode != "id" or float(summary["acceptance_rate"]) >= 0.95
        summary["status"] = "PASS" if mode_pass else "FAIL"
        summaries[mode] = summary
        all_pass &= mode_pass
        with (output_dir / f"{mode}_episodes.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        (output_dir / f"{mode}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[{mode}] {summary['accepted_episodes'] if mode == 'id' else 'n/a'}/"
            f"{summary['episodes']} accepted; pitch_rms_mean="
            f"{summary['pitch_rms_deg_mean']:.4f}deg; drift_p95="
            f"{summary['planar_drift_m_p95']:.5f}m",
            flush=True,
        )

    result = {
        "status": "PASS" if all_pass else "FAIL",
        "task_id": "pure_ppo_stand_v1",
        "training_seed": args.training_seed,
        "mode": args.mode,
        "episodes_per_mode": args.episodes,
        "independent_test_seed_start": TEST_SEED_BASE[args.training_seed],
        "independent_test_seed_end_exclusive": TEST_SEED_BASE[args.training_seed] + args.episodes,
        "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "contract": str(contract_path.relative_to(ROOT)) if contract_path.is_relative_to(ROOT) else str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "model_xml_sha256": model_xml_sha256(),
        "source_fingerprint_sha256": source_fingerprint(),
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
