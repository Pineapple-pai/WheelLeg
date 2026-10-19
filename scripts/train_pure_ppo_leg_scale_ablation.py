#!/usr/bin/env python3
"""Train a fresh PPO policy for the evidence-driven 0.35 rad leg-scale ablation.

Only ``LEG_POSITION_SCALE_RAD`` changes from ``pure_ppo_stand_v1``. The task,
reward weights, observation path, controller, PPO settings, simulator, and
termination limits remain fixed. This entry point versions the runtime change
in its task ID, configuration, gates, and checkpoint contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from uz05 import pure_ppo_stand as task


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "pure_ppo_stand_leg_scale_035_v1"
LEG_SCALE_RAD = 0.35
BASELINE_LEG_SCALE_RAD = 0.50


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_experiment() -> None:
    if task.LEG_POSITION_SCALE_RAD != BASELINE_LEG_SCALE_RAD:
        raise RuntimeError("baseline source leg scale changed; this ablation is no longer one-factor")
    task.LEG_POSITION_SCALE_RAD = LEG_SCALE_RAD
    task.TASK_ID = EXPERIMENT_ID

    # Import the contract and trainer only after the runtime scale and task ID
    # are locked so the contract snapshots the variant value.
    from uz05 import pure_ppo_contract as contract

    contract.CONTRACT_VERSION = EXPERIMENT_ID
    import preflight_pure_ppo_stand as preflight
    import smoke_pure_ppo_stand as smoke
    import train_pure_ppo_stand as trainer

    trainer.CONTRACT_VERSION = EXPERIMENT_ID
    trainer.GATE_DIR = ROOT / "artifacts" / EXPERIMENT_ID
    trainer.RUN_ROOT = ROOT / "runs" / EXPERIMENT_ID
    trainer.CHECKPOINT_ROOT = ROOT / "checkpoints" / EXPERIMENT_ID
    trainer.VALIDATION_SEEDS = tuple(range(11_000, 11_050))

    original_config = trainer._training_config

    def variant_config(seed: int, updates: int, n_envs: int, device: str):
        config = original_config(seed, updates, n_envs, device)
        config["single_factor_ablation"] = {
            "factor": "leg_position_scale_rad",
            "baseline": BASELINE_LEG_SCALE_RAD,
            "value": LEG_SCALE_RAD,
            "ablation_entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "reason": "paired held-out counterfactual removed observed fourth-leg PD saturation",
        }
        return config

    trainer._training_config = variant_config
    trainer.GATE_DIR.mkdir(parents=True, exist_ok=True)
    (trainer.RUN_ROOT).mkdir(parents=True, exist_ok=True)
    (trainer.CHECKPOINT_ROOT).mkdir(parents=True, exist_ok=True)

    preflight_report = preflight.run()
    smoke_report = smoke.run()
    for name, report in (("preflight", preflight_report), ("smoke", smoke_report)):
        report["task_id"] = EXPERIMENT_ID
        report["single_factor_ablation"] = {
            "factor": "leg_position_scale_rad",
            "baseline": BASELINE_LEG_SCALE_RAD,
            "value": LEG_SCALE_RAD,
            "ablation_entrypoint_sha256": _sha256(Path(__file__).resolve()),
        }
        (trainer.GATE_DIR / f"{name}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    dry_args = SimpleNamespace(seed=0, device="cuda")
    dry_status = trainer.run_dry_run(dry_args)
    if dry_status != 0:
        raise RuntimeError("ablation PPO dry run did not pass")
    return trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    trainer = _configure_experiment()
    train_args = SimpleNamespace(
        seed=args.seed,
        updates=args.updates,
        num_envs=args.num_envs,
        eval_interval=50,
        device=args.device,
        dummy_vec_env=False,
    )
    return trainer.run_formal_training(train_args)


if __name__ == "__main__":
    raise SystemExit(main())
