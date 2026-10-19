#!/usr/bin/env python3
"""Export only an accepted Pure PPO actor and verify SB3/ONNX in MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from eval_pure_ppo_stand import TEST_SEED_BASE
from uz05.pure_ppo_contract import read_and_require_contract, source_fingerprint
from uz05.pure_ppo_eval import EVAL_MODES, evaluate_episode, summarize_evaluation
from uz05.pure_ppo_stand import (
    ACTOR_OBS_DIM,
    ACTION_DIM,
    EPISODE_STEPS,
    OBS_DIM,
    PurePPOStandEnv,
)


class _ActorOnly(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, actor_observation):
        latent = self.policy.mlp_extractor.forward_actor(actor_observation)
        mean = self.policy.action_net(latent)
        return torch.clamp(mean, -1.0, 1.0)


class _OnnxPolicy:
    def __init__(self, path: Path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, observation, deterministic: bool = True):
        del deterministic
        values = np.asarray(observation, dtype=np.float32)
        batched = values.ndim == 2
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != OBS_DIM:
            raise ValueError(f"expected full observation [..., {OBS_DIM}], got {values.shape}")
        actions = self.session.run(
            ["action"], {self.input_name: values[:, :ACTOR_OBS_DIM]}
        )[0]
        return (actions if batched else actions[0]), None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_observations(model: PPO, count: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(610_001)
    env = PurePPOStandEnv(seed=610_001)
    obs, _ = env.reset(seed=610_001)
    values = []
    while len(values) < count:
        values.append(obs.copy())
        action = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.asarray(values[:count], dtype=np.float32)


def _compare_episode_metrics(sb3_rows: list[dict], onnx_rows: list[dict]) -> dict[str, Any]:
    if len(sb3_rows) != len(onnx_rows):
        raise AssertionError("SB3 and ONNX episode counts differ")
    fields = (
        "pitch_rms_rad",
        "pitch_peak_rad",
        "roll_rms_rad",
        "drift_max_m",
        "wheel_left_contact_fraction",
        "wheel_right_contact_fraction",
    )
    deltas = {field: [] for field in fields}
    acceptance_agreement = 0
    termination_agreement = 0
    for sb3, onnx in zip(sb3_rows, onnx_rows):
        if sb3["seed"] != onnx["seed"] or sb3["mode"] != onnx["mode"]:
            raise AssertionError("SB3 and ONNX episodes use different seed/mode ordering")
        acceptance_agreement += sb3.get("accepted_episode") == onnx.get("accepted_episode")
        termination_agreement += sb3["termination_reason"] == onnx["termination_reason"]
        for field in fields:
            deltas[field].append(abs(float(sb3[field]) - float(onnx[field])))
    maxima = {key: float(np.max(value, initial=0.0)) for key, value in deltas.items()}
    tolerances = {
        "pitch_rms_rad": 1e-3,
        "pitch_peak_rad": 2e-3,
        "roll_rms_rad": 1e-3,
        "drift_max_m": 1e-4,
        "wheel_left_contact_fraction": 1e-3,
        "wheel_right_contact_fraction": 1e-3,
    }
    passed = (
        acceptance_agreement == len(sb3_rows)
        and termination_agreement == len(sb3_rows)
        and all(maxima[key] <= tolerances[key] for key in fields)
    )
    return {
        "episodes": len(sb3_rows),
        "acceptance_agreement_episodes": int(acceptance_agreement),
        "termination_agreement_episodes": int(termination_agreement),
        "max_abs_metric_delta": maxima,
        "tolerance": tolerances,
        "status": "PASS" if passed else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    evaluation_path = args.evaluation_summary.resolve()
    if not checkpoint.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("checkpoint and accepted evaluation summary must exist")
    contract_path = args.contract or checkpoint.with_name(f"{checkpoint.stem}_contract.json")
    contract_path = contract_path.resolve()
    contract = read_and_require_contract(contract_path)
    if int(contract["seed"]) != args.training_seed:
        raise ValueError("checkpoint contract seed does not match --training-seed")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    id_summary = evaluation.get("summaries", {}).get("id", {})
    if (
        evaluation.get("status") != "PASS"
        or evaluation.get("mode") not in ("id", "all")
        or evaluation.get("episodes_per_mode", 0) < 100
        or id_summary.get("status") != "PASS"
        or id_summary.get("episodes", 0) < 100
        or id_summary.get("acceptance_rate", 0.0) < 0.95
        or evaluation.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise ValueError("ONNX export requires a passing 100-episode independent ID evaluation for this checkpoint")

    output = (args.output or checkpoint.parent / "actor_stand_offset_v1.onnx").resolve()
    contract_output = output.with_suffix(output.suffix + ".contract.json")
    if output.exists() or contract_output.exists():
        raise FileExistsError("refusing to overwrite an existing ONNX artifact or contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    parity_dir = output.parent / f"{output.stem}_episode_parity"
    if parity_dir.exists():
        raise FileExistsError(f"refusing to overwrite ONNX episode parity data: {parity_dir}")

    # Inference comparison is done on CPU on both sides to avoid device-specific
    # arithmetic differences and to represent the deployment runtime.
    model = PPO.load(str(checkpoint), device="cpu")
    if tuple(model.observation_space.shape) != (OBS_DIM,) or tuple(model.action_space.shape) != (ACTION_DIM,):
        raise ValueError("checkpoint observation/action dimensions do not match the current contract")
    if int(getattr(model.policy, "_actor_dim", -1)) != ACTOR_OBS_DIM:
        raise ValueError("checkpoint actor input is not 38D")
    actor = _ActorOnly(model.policy).eval()
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp.onnx", dir=output.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.onnx.export(
            actor,
            torch.zeros((1, ACTOR_OBS_DIM), dtype=torch.float32),
            str(temp_path),
            input_names=["actor_observation"],
            output_names=["action"],
            dynamic_axes={"actor_observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
        session = ort.InferenceSession(str(temp_path), providers=["CPUExecutionProvider"])
        samples = _collect_observations(model, count=1000)
        sb3_actions, _ = model.predict(samples, deterministic=True)
        onnx_actions = session.run(
            ["action"], {"actor_observation": samples[:, :ACTOR_OBS_DIM]}
        )[0]
        action_error = np.abs(np.asarray(sb3_actions) - onnx_actions)
        action_parity = {
            "observations": int(len(samples)),
            "input_shape": [1, ACTOR_OBS_DIM],
            "output_shape": [1, ACTION_DIM],
            "max_abs_error": float(action_error.max(initial=0.0)),
            "mean_abs_error": float(action_error.mean()),
            "threshold_max_abs_error": 1e-4,
            "status": "PASS" if action_error.max(initial=0.0) <= 1e-4 else "FAIL",
        }
        if action_parity["status"] != "PASS":
            raise AssertionError(f"ONNX action parity failed: {action_parity}")

        parity_dir.mkdir(parents=True)
        onnx_policy = _OnnxPolicy(temp_path)
        episode_comparisons = {}
        summaries = {}
        for mode in EVAL_MODES:
            seeds = range(TEST_SEED_BASE[args.training_seed], TEST_SEED_BASE[args.training_seed] + 100)
            sb3_rows = [evaluate_episode(model, seed=seed, mode=mode) for seed in seeds]
            onnx_rows = [evaluate_episode(onnx_policy, seed=seed, mode=mode) for seed in range(
                TEST_SEED_BASE[args.training_seed], TEST_SEED_BASE[args.training_seed] + 100
            )]
            comparison = _compare_episode_metrics(sb3_rows, onnx_rows)
            episode_comparisons[mode] = comparison
            summaries[mode] = {
                "sb3": summarize_evaluation(sb3_rows, mode),
                "onnx": summarize_evaluation(onnx_rows, mode),
            }
            for name, rows in (("sb3", sb3_rows), ("onnx", onnx_rows)):
                with (parity_dir / f"{mode}_{name}_episodes.jsonl").open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            (parity_dir / f"{mode}_comparison.json").write_text(
                json.dumps({"comparison": comparison, "summaries": summaries[mode]},
                           indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        if any(row["status"] != "PASS" for row in episode_comparisons.values()):
            raise AssertionError("ONNX and SB3 MuJoCo episode metrics differ beyond tolerance")

        temp_path.replace(output)
        report = {
            "status": "PASS",
            "training_seed": args.training_seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "accepted_evaluation_summary": str(evaluation_path),
            "accepted_evaluation_summary_sha256": _sha256(evaluation_path),
            "onnx_path": str(output),
            "onnx_sha256": _sha256(output),
            "source_fingerprint_sha256": source_fingerprint(),
            "action_parity": action_parity,
            "complete_mujoco_episode_parity": episode_comparisons,
            "evaluation_summaries": summaries,
            "episodes_per_mode_per_runtime": 100,
        }
        payload = dict(contract)
        payload.update({
            "artifact_type": "actor_onnx",
            "onnx_sha256": report["onnx_sha256"],
            "checkpoint_sha256": report["checkpoint_sha256"],
            "accepted_evaluation_summary_sha256": report["accepted_evaluation_summary_sha256"],
            "action_parity": action_parity,
            "complete_mujoco_episode_parity": episode_comparisons,
        })
        contract_output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (parity_dir / "onnx_export_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
