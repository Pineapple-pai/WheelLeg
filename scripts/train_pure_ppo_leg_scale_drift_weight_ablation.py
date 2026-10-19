#!/usr/bin/env python3
"""Fresh PPO ablation: raise only the fixed position-drift reward coefficient.

The preceding experiment used a fresh random actor with a fixed 0.35 rad leg
target scale and exposed low wheel contact/high drift. This run keeps that
action map and all other settings fixed; only the position-drift reward weight
changes from 0.10 to 0.50 before random initialization.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from uz05 import pure_ppo_stand as task


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "pure_ppo_stand_leg_scale_035_drift_weight_5x_v1"
LEG_SCALE_RAD = 0.35
BASELINE_DRIFT_WEIGHT = 0.10
DRIFT_WEIGHT = 0.50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_experiment(*, reuse_gates: bool = False):
    if task.LEG_POSITION_SCALE_RAD != 0.50:
        raise RuntimeError("baseline leg scale changed; this is no longer the intended one-factor test")
    if task.REWARD_CONFIG.position_drift != BASELINE_DRIFT_WEIGHT:
        raise RuntimeError("baseline position-drift coefficient changed")

    task.LEG_POSITION_SCALE_RAD = LEG_SCALE_RAD
    task.TASK_ID = EXPERIMENT_ID
    base_reward = task.compute_stand_reward
    task.REWARD_CONFIG = dataclasses.replace(
        task.REWARD_CONFIG, position_drift=DRIFT_WEIGHT
    )

    def compute_variant_reward(**kwargs):
        kwargs["config"] = task.REWARD_CONFIG
        return base_reward(**kwargs)

    task.compute_stand_reward = compute_variant_reward

    # Import versioned dependencies after locking the complete task config.
    from uz05 import pure_ppo_contract as contract

    contract.CONTRACT_VERSION = EXPERIMENT_ID
    import preflight_pure_ppo_stand as preflight
    import smoke_pure_ppo_stand as smoke
    import train_pure_ppo_stand as trainer

    trainer.CONTRACT_VERSION = EXPERIMENT_ID
    trainer.GATE_DIR = ROOT / "artifacts" / EXPERIMENT_ID
    trainer.RUN_ROOT = ROOT / "runs" / EXPERIMENT_ID
    trainer.CHECKPOINT_ROOT = ROOT / "checkpoints" / EXPERIMENT_ID
    trainer.VALIDATION_SEEDS = tuple(range(12_000, 12_050))

    original_config = trainer._training_config

    def ablation_config(seed: int, updates: int, n_envs: int, device: str):
        config = original_config(seed, updates, n_envs, device)
        config["single_factor_ablation"] = {
            "base_experiment_id": "pure_ppo_stand_leg_scale_035_v1",
            "factor": "reward.position_drift",
            "baseline": BASELINE_DRIFT_WEIGHT,
            "value": DRIFT_WEIGHT,
            "leg_position_scale_rad_held_fixed": LEG_SCALE_RAD,
            "ablation_entrypoint_sha256": _sha256(Path(__file__).resolve()),
            "reason": (
                "fresh 0.35 rad run held pitch but failed the independent drift, "
                "wheel-contact, and saturation gates; paired scale-only changes "
                "reduced wheel saturation while worsening drift"
            ),
        }
        return config

    trainer._training_config = ablation_config
    trainer.GATE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    trainer.CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task_id": EXPERIMENT_ID,
        "single_factor_ablation": {
            "base_experiment_id": "pure_ppo_stand_leg_scale_035_v1",
            "factor": "reward.position_drift",
            "baseline": BASELINE_DRIFT_WEIGHT,
            "value": DRIFT_WEIGHT,
            "leg_position_scale_rad_held_fixed": LEG_SCALE_RAD,
            "ablation_entrypoint_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    if reuse_gates:
        for name in ("preflight", "smoke", "dry_run"):
            gate_path = trainer.GATE_DIR / f"{name}.json"
            if not gate_path.is_file():
                raise RuntimeError(f"cannot reuse missing gate report: {gate_path}")
            report = json.loads(gate_path.read_text(encoding="utf-8"))
            if report.get("task_id") != EXPERIMENT_ID:
                raise RuntimeError(f"cannot reuse {name}: task ID does not match")
            factor = report.get("single_factor_ablation", {})
            if (
                factor.get("factor") != "reward.position_drift"
                or factor.get("baseline") != BASELINE_DRIFT_WEIGHT
                or factor.get("value") != DRIFT_WEIGHT
                or factor.get("leg_position_scale_rad_held_fixed") != LEG_SCALE_RAD
            ):
                raise RuntimeError(f"cannot reuse {name}: one-factor config does not match")
        trainer._check_training_gates()
    else:
        for name, report in (("preflight", preflight.run()), ("smoke", smoke.run())):
            report.update(metadata)
            (trainer.GATE_DIR / f"{name}.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

        dry_status = trainer.run_dry_run(SimpleNamespace(seed=0, device="cuda"))
        if dry_status != 0:
            raise RuntimeError("drift-weight ablation PPO dry run did not pass")
        dry_path = trainer.GATE_DIR / "dry_run.json"
        dry_report = json.loads(dry_path.read_text(encoding="utf-8"))
        dry_report.update(metadata)
        dry_path.write_text(
            json.dumps(dry_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--reuse-gates",
        action="store_true",
        help="reuse this experiment's matching passing preflight, smoke, and dry-run reports",
    )
    args = parser.parse_args()
    trainer = _configure_experiment(reuse_gates=args.reuse_gates)
    return trainer.run_formal_training(
        SimpleNamespace(
            seed=args.seed,
            updates=args.updates,
            num_envs=args.num_envs,
            eval_interval=50,
            device=args.device,
            dummy_vec_env=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
