#!/usr/bin/env python3
"""Replay fixed Pure PPO checkpoints and save read-only, step-level traces.

This diagnostic does not train, alter weights, modify policy actions, or touch
the historical run directory.  Its output directory must not already exist.
For an experiment-specific checkpoint, pass its recorded leg scale and drift
reward coefficient explicitly, plus the held-out seed list or a JSON file of
fixed initial states.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from uz05 import pure_ppo_stand as task
from uz05 import pure_ppo_contract as contract
from uz05.pure_ppo_eval import episode_passes_acceptance


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_source_fingerprint() -> str:
    """Recompute the code fingerprint without depending on runtime imports."""
    paths = [
        "scripts/uz05/model.py",
        "scripts/uz05/spec.py",
        "scripts/uz05/actuators.py",
        "scripts/uz05/observations.py",
        "scripts/uz05/policy.py",
        "scripts/uz05/env.py",
        "scripts/uz05/pure_ppo_stand.py",
        "scripts/uz05/pure_ppo_contract.py",
        "scripts/uz05/pure_ppo_eval.py",
        "scripts/train_pure_ppo_stand.py",
        "scripts/preflight_pure_ppo_stand.py",
        "scripts/smoke_pure_ppo_stand.py",
        "scripts/eval_pure_ppo_stand.py",
        "scripts/export_pure_ppo_stand_onnx.py",
    ]
    digest = hashlib.sha256()
    for relative in paths:
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("provide a nonempty list of unique seeds")
    return seeds


def validate_checkpoint_contract(
    checkpoint: Path, *, task_id: str, scale: float, drift_weight: float
) -> dict[str, Any]:
    saved = json.loads(checkpoint.with_name(f"{checkpoint.stem}_contract.json").read_text())
    if saved.get("contract_version") != task_id:
        raise ValueError(
            f"contract task id {saved.get('contract_version')!r} does not match {task_id!r}"
        )

    # The training source fingerprint differs from the current worktree because
    # the report records a post-run formatting-only change.  Recompute and
    # report both hashes; independently check all behavior-bearing fields below.
    actual_xml_sha = contract.model_xml_sha256()
    if saved.get("model_xml_sha256") != actual_xml_sha:
        raise ValueError(
            f"checkpoint model SHA {saved.get('model_xml_sha256')} != loaded XML {actual_xml_sha}"
        )
    checks = {
        "action_protocol": task.ACTION_PROTOCOL,
        "action_order": list(task.ACTION_ORDER),
        "action_shape": task.ACTION_DIM,
        "q_ref_rad": task.Q_REF.tolist(),
        "wheel_sign": task.WHEEL_SIGN.tolist(),
        "wheel_speed_scale_rad_s": task.WHEEL_SPEED_SCALE_RAD_S,
        "leg_position_scale_rad": scale,
        "policy_rate_hz": task.POLICY_RATE_HZ,
        "motor_rate_hz": task.MOTOR_RATE_HZ,
        "actor_observation_shape": task.ACTOR_OBS_DIM,
        "critic_observation_shape": task.PRIV_OBS_DIM,
        "observation_shape": task.OBS_DIM,
        "domain_randomization": False,
        "observation_noise": False,
    }
    for key, expected in checks.items():
        if saved.get(key) != expected:
            raise ValueError(f"checkpoint contract mismatch for {key}: {saved.get(key)!r} != {expected!r}")

    run_config_path = ROOT / "runs" / task_id / f"seed{saved['seed']}" / "config.json"
    run_config = json.loads(run_config_path.read_text())
    if run_config["model_xml_sha256"] != actual_xml_sha:
        raise ValueError("run config model SHA does not match the loaded model")
    if run_config["leg_position_scale_rad"] != scale:
        raise ValueError("run config leg scale does not match --leg-scale")
    if run_config["reward"]["position_drift"] != drift_weight:
        raise ValueError("run config drift reward does not match --drift-weight")
    if run_config["source_fingerprint_sha256"] != saved.get("source_fingerprint_sha256"):
        raise ValueError("run config and checkpoint source fingerprints disagree")

    # Verify the saved actuator contract against the live parameter values.  It
    # includes wheel current and torque limits and the joint PD gains/limits.
    live_payload = contract.contract_payload(
        seed=int(saved["seed"]), config_sha256=str(saved["config_sha256"]),
        commit=str(saved["commit"]),
    )
    for key in (
        "actuator_config", "control_dt_s", "motor_dt_s",
        "motor_ticks_per_policy_step",
    ):
        if saved.get(key) != live_payload.get(key):
            raise ValueError(f"checkpoint actuator/control contract mismatch for {key}")
    return {"contract": saved, "config": run_config, "actual_model_sha256": actual_xml_sha}


def read_reference(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(row["seed"]): row for row in rows}


def reference_diff(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    scalar_keys = (
        "steps", "termination_reason", "initial_pitch_rad", "initial_pitch_rate_rad_s",
        "pitch_rms_rad", "pitch_peak_rad", "drift_max_m", "wheel_left_contact_fraction",
        "wheel_right_contact_fraction", "wheel_saturation_fraction",
        "wheel_saturation_max_consecutive_steps", "leg_torque_saturation_fraction",
        "reward_component_mean", "action_mean",
    )
    for key in scalar_keys:
        if key not in expected:
            continue
        lhs, rhs = actual.get(key), expected.get(key)
        if isinstance(rhs, (float, int)) and not isinstance(rhs, bool):
            if not np.isclose(float(lhs), float(rhs), rtol=1e-9, atol=1e-10):
                diffs.append(f"{key}: replay={lhs!r}, saved={rhs!r}")
        elif isinstance(rhs, list):
            if not np.allclose(
                np.asarray(lhs, dtype=np.float64),
                np.asarray(rhs, dtype=np.float64),
                rtol=1e-9,
                atol=1e-10,
            ):
                diffs.append(f"{key}: replay={lhs!r}, saved={rhs!r}")
        elif isinstance(rhs, dict):
            if lhs.keys() != rhs.keys() or any(
                not np.isclose(float(lhs[name]), float(rhs[name]), rtol=1e-9, atol=1e-10)
                for name in rhs
            ):
                diffs.append(f"{key}: replay={lhs!r}, saved={rhs!r}")
        elif lhs != rhs:
            diffs.append(f"{key}: replay={lhs!r}, saved={rhs!r}")
    if "acceptance_failures" in expected and actual.get("acceptance_failures") != expected["acceptance_failures"]:
        diffs.append(
            f"acceptance_failures: replay={actual.get('acceptance_failures')!r}, "
            f"saved={expected['acceptance_failures']!r}"
        )
    return diffs


def load_initial_states(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("initial-state JSON must be a nonempty list")
    for row in rows:
        if "seed" not in row or "initial_pitch_rad" not in row or "initial_pitch_rate_rad_s" not in row:
            raise ValueError("each explicit initial state requires seed, initial_pitch_rad, and initial_pitch_rate_rad_s")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--leg-scale", type=float, required=True)
    parser.add_argument("--drift-weight", type=float, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seeds", type=parse_seeds, help="comma-separated exact reset seeds")
    group.add_argument("--seed-start", type=int, help="first reset seed in a consecutive range")
    group.add_argument("--initial-states-json", type=Path, help="fixed initial states for paired inference")
    parser.add_argument("--episodes", type=int, default=100, help="episode count for --seed-start")
    parser.add_argument("--reference-episodes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")

    # Apply only the experiment's recorded constants.  The Actor's six outputs
    # are consumed directly; no action is adjusted by this script.
    task.LEG_POSITION_SCALE_RAD = float(args.leg_scale)
    task.TASK_ID = args.task_id
    task.REWARD_CONFIG = dataclasses.replace(
        task.REWARD_CONFIG, position_drift=float(args.drift_weight)
    )
    base_reward = task.compute_stand_reward

    def compute_variant_reward(**kwargs):
        kwargs["config"] = task.REWARD_CONFIG
        return base_reward(**kwargs)

    task.compute_stand_reward = compute_variant_reward
    contract.CONTRACT_VERSION = args.task_id

    verified = validate_checkpoint_contract(
        checkpoint,
        task_id=args.task_id,
        scale=float(args.leg_scale),
        drift_weight=float(args.drift_weight),
    )
    saved_contract = verified["contract"]
    reference = read_reference(args.reference_episodes)
    model = PPO.load(str(checkpoint), device="cpu")
    if tuple(model.observation_space.shape) != (task.OBS_DIM,):
        raise ValueError("checkpoint observation space mismatch")
    if tuple(model.action_space.shape) != (task.ACTION_DIM,):
        raise ValueError("checkpoint six-action space mismatch")
    if int(getattr(model.policy, "_actor_dim", -1)) != task.ACTOR_OBS_DIM:
        raise ValueError("checkpoint actor observation dimension mismatch")

    records: list[dict[str, Any]] = []
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.seeds:
        cases = [{"seed": seed, "label": f"seed_{seed}", "options": {}} for seed in args.seeds]
    elif args.seed_start is not None:
        cases = [
            {"seed": seed, "label": f"seed_{seed}", "options": {}}
            for seed in range(args.seed_start, args.seed_start + args.episodes)
        ]
    else:
        cases = []
        for index, row in enumerate(load_initial_states(args.initial_states_json)):
            cases.append({
                "seed": int(row["seed"]),
                "label": str(row.get("label", f"explicit_{index}")),
                "options": {
                    "initial_pitch_rad": float(row["initial_pitch_rad"]),
                    "initial_pitch_rate_rad_s": float(row["initial_pitch_rate_rad_s"]),
                },
            })

    output.mkdir(parents=True, exist_ok=False)
    headers = [
        "episode_seed", "episode_label", "step", "time_s",
        "initial_pitch_rad", "initial_pitch_rate_rad_s",
        "pitch_rad", "pitch_rate_rad_s", "roll_rad", "base_vx_m_s", "base_vy_m_s",
        "base_height_m", "planar_drift_m", "planar_drift_delta_m",
        *[f"actor_mean_raw_{i}" for i in range(6)],
        *[f"action_policy_clipped_{i}" for i in range(6)],
        *[f"action_applied_{i}" for i in range(6)],
        "wheel_target_left_rad_s", "wheel_target_right_rad_s",
        "wheel_speed_before_left_rad_s", "wheel_speed_before_right_rad_s",
        "wheel_speed_after_left_rad_s", "wheel_speed_after_right_rad_s",
        "wheel_speed_error_left_rad_s", "wheel_speed_error_right_rad_s",
        "wheel_current_left_a", "wheel_current_right_a", "wheel_torque_left_nm", "wheel_torque_right_nm",
        "wheel_current_saturated_left", "wheel_current_saturated_right",
        "wheel_torque_saturated_left", "wheel_torque_saturated_right",
        "wheel_saturated_left", "wheel_saturated_right",
        "wheel_saturation_run_left", "wheel_saturation_run_right",
        *[f"joint_target_{i}_rad" for i in range(4)],
        *[f"joint_position_before_{i}_rad" for i in range(4)],
        *[f"joint_position_after_{i}_rad" for i in range(4)],
        *[f"joint_velocity_before_{i}_rad_s" for i in range(4)],
        *[f"joint_velocity_after_{i}_rad_s" for i in range(4)],
        *[f"joint_torque_{i}_nm" for i in range(4)],
        *[f"leg_torque_saturated_{i}" for i in range(4)],
        "left_wheel_contact", "right_wheel_contact", "body_contact",
        "reward_total", *[f"reward_component_{name}" for name in (
            "upright", "alive", "pitch_rate", "horizontal_velocity", "position_drift",
            "height_error", "leg_length_error", "action_rate", "effort", "termination",
        )],
        *[f"reward_raw_{name}" for name in (
            "upright", "alive", "pitch_rate_cost", "horizontal_velocity_cost", "position_drift_cost",
            "height_error_cost", "leg_length_error_cost", "action_rate_cost", "effort_cost",
            "terminal_failure",
        )],
        "termination_reason",
    ]
    episode_summaries = []
    trace_path = output / "step_trace.csv.gz"
    env = task.PurePPOStandEnv()
    try:
        with gzip.open(trace_path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers)
            writer.writeheader()
            for case in cases:
                observation, reset_info = env.reset(seed=case["seed"], options=case["options"])
                drift_previous = 0.0
                sat_run = np.zeros(2, dtype=np.int64)
                sat_runs: list[list[dict[str, int]]] = [[], []]
                open_start: list[int | None] = [None, None]
                done_metrics = None
                for step in range(1, task.EPISODE_STEPS + 1):
                    obs_tensor, _ = model.policy.obs_to_tensor(observation)
                    with torch.no_grad():
                        distribution = model.policy.get_distribution(obs_tensor).distribution
                        actor_mean = distribution.mean.detach().cpu().numpy().reshape(-1)
                    action, _ = model.predict(observation, deterministic=True)
                    action = np.asarray(action, dtype=np.float64).reshape(task.ACTION_DIM)

                    wheel_before = env.sim.wheel_velocities().copy()
                    joint_pos_before = env.sim.joint_positions().copy()
                    joint_vel_before = env.sim.joint_velocities().copy()
                    observation, reward, terminated, truncated, info = env.step(action)
                    wheel_after = env.sim.wheel_velocities().copy()
                    joint_pos_after = env.sim.joint_positions().copy()
                    joint_vel_after = env.sim.joint_velocities().copy()
                    wheel_i = np.asarray(info["wheel_current_a"], dtype=np.float64)
                    wheel_tau = np.asarray(info["wheel_torque_nm"], dtype=np.float64)
                    leg_tau = np.asarray(info["joint_torque_nm"], dtype=np.float64)
                    contact = np.asarray(info["wheel_contact"], dtype=np.float64)
                    wheel_cur_sat = np.abs(wheel_i) >= 0.99 * env.params.wheel.current_limit_a
                    wheel_tau_sat = np.abs(wheel_tau) >= 0.99 * env.params.wheel.joint_torque_limit
                    wheel_sat = wheel_cur_sat | wheel_tau_sat
                    for side in range(2):
                        if wheel_sat[side]:
                            sat_run[side] += 1
                            if open_start[side] is None:
                                open_start[side] = step
                        else:
                            if open_start[side] is not None:
                                sat_runs[side].append({"start_step": open_start[side], "end_step": step - 1})
                                open_start[side] = None
                            sat_run[side] = 0

                    values: dict[str, Any] = {
                        "episode_seed": case["seed"], "episode_label": case["label"],
                        "step": step, "time_s": step * task.POLICY_DT_S,
                        "initial_pitch_rad": reset_info["initial_pitch_rad"],
                        "initial_pitch_rate_rad_s": reset_info["initial_pitch_rate_rad_s"],
                        "pitch_rad": info["pitch_rad"], "pitch_rate_rad_s": info["pitch_rate_rad_s"],
                        "roll_rad": info["roll_rad"], "base_vx_m_s": info["base_vx_m_s"],
                        "base_vy_m_s": info["base_vy_m_s"], "base_height_m": info["base_height_m"],
                        "planar_drift_m": info["planar_drift_m"],
                        "planar_drift_delta_m": info["planar_drift_m"] - drift_previous,
                        "wheel_target_left_rad_s": info["wheel_target_rad_s"][0],
                        "wheel_target_right_rad_s": info["wheel_target_rad_s"][1],
                        "wheel_speed_before_left_rad_s": wheel_before[0],
                        "wheel_speed_before_right_rad_s": wheel_before[1],
                        "wheel_speed_after_left_rad_s": wheel_after[0],
                        "wheel_speed_after_right_rad_s": wheel_after[1],
                        "wheel_speed_error_left_rad_s": info["wheel_target_rad_s"][0] - wheel_before[0],
                        "wheel_speed_error_right_rad_s": info["wheel_target_rad_s"][1] - wheel_before[1],
                        "wheel_current_left_a": wheel_i[0], "wheel_current_right_a": wheel_i[1],
                        "wheel_torque_left_nm": wheel_tau[0], "wheel_torque_right_nm": wheel_tau[1],
                        "wheel_current_saturated_left": bool(wheel_cur_sat[0]),
                        "wheel_current_saturated_right": bool(wheel_cur_sat[1]),
                        "wheel_torque_saturated_left": bool(wheel_tau_sat[0]),
                        "wheel_torque_saturated_right": bool(wheel_tau_sat[1]),
                        "wheel_saturated_left": bool(wheel_sat[0]),
                        "wheel_saturated_right": bool(wheel_sat[1]),
                        "wheel_saturation_run_left": int(sat_run[0]),
                        "wheel_saturation_run_right": int(sat_run[1]),
                        "left_wheel_contact": contact[0], "right_wheel_contact": contact[1],
                        "body_contact": info["body_contact"], "reward_total": reward,
                        "termination_reason": info["termination_reason"],
                    }
                    values.update({f"actor_mean_raw_{i}": actor_mean[i] for i in range(6)})
                    values.update({f"action_policy_clipped_{i}": action[i] for i in range(6)})
                    values.update({f"action_applied_{i}": info["action_applied"][i] for i in range(6)})
                    values.update({f"joint_target_{i}_rad": info["joint_target_rad"][i] for i in range(4)})
                    values.update({f"joint_position_before_{i}_rad": joint_pos_before[i] for i in range(4)})
                    values.update({f"joint_position_after_{i}_rad": joint_pos_after[i] for i in range(4)})
                    values.update({f"joint_velocity_before_{i}_rad_s": joint_vel_before[i] for i in range(4)})
                    values.update({f"joint_velocity_after_{i}_rad_s": joint_vel_after[i] for i in range(4)})
                    values.update({f"joint_torque_{i}_nm": leg_tau[i] for i in range(4)})
                    leg_sat = np.abs(leg_tau) >= 0.99 * env.params.joint.torque_limit
                    values.update({f"leg_torque_saturated_{i}": bool(leg_sat[i]) for i in range(4)})
                    values.update({f"reward_component_{k}": v for k, v in info["reward_components"].items()})
                    values.update({f"reward_raw_{k}": v for k, v in info["reward_raw"].items()})
                    writer.writerow(values)
                    drift_previous = float(info["planar_drift_m"])
                    if terminated or truncated:
                        done_metrics = info["episode_metrics"]
                        break
                if done_metrics is None:
                    raise RuntimeError(f"episode {case['label']} did not end within the task horizon")
                for side in range(2):
                    if open_start[side] is not None:
                        sat_runs[side].append({"start_step": open_start[side], "end_step": done_metrics["steps"]})
                passed, failures = episode_passes_acceptance(done_metrics)
                episode = {
                    "seed": int(case["seed"]), "label": case["label"],
                    "initial_pitch_rad": reset_info["initial_pitch_rad"],
                    "initial_pitch_rate_rad_s": reset_info["initial_pitch_rate_rad_s"],
                    "accepted_episode": passed, "acceptance_failures": failures,
                    **done_metrics,
                    "wheel_saturation_runs_by_side": sat_runs,
                }
                if case["seed"] in reference:
                    episode["reference_differences"] = reference_diff(episode, reference[case["seed"]])
                records.append(episode)
                stream.flush()
                print(
                    f"{case['label']}: {done_metrics['steps']} steps; "
                    f"failures={','.join(failures) if failures else 'none'}; "
                    f"maxwheelrun={max(done_metrics['wheel_saturation_max_consecutive_steps'])}",
                    flush=True,
                )
    finally:
        env.close()

    (output / "episode_metrics.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "diagnostic_only": True,
        "training_started": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "contract_sha256": sha256(checkpoint.with_name(f"{checkpoint.stem}_contract.json")),
        "task_id": args.task_id,
        "training_seed": args.training_seed,
        "leg_position_scale_rad": args.leg_scale,
        "reward_position_drift_coefficient": args.drift_weight,
        "model_xml_path": str(task.__dict__.get("MODEL_XML", contract.MODEL_XML)),
        "model_xml_sha256": verified["actual_model_sha256"],
        "saved_source_fingerprint_sha256": saved_contract["source_fingerprint_sha256"],
        "replay_source_fingerprint_sha256": current_source_fingerprint(),
        "source_note": "Current fingerprint is reported separately; training records preserve their original fingerprint. This replay does not rewrite either contract.",
        "seeds": [case["seed"] for case in cases],
        "episode_count": len(cases),
        "explicit_initial_states_json": str(args.initial_states_json.resolve()) if args.initial_states_json else None,
        "reference_episodes": str(args.reference_episodes.resolve()) if args.reference_episodes else None,
        "reference_comparison_mismatch_count": sum(
            bool(row.get("reference_differences")) for row in records
        ),
        "trace_rows": int(sum(row["steps"] for row in records)),
        "trace_csv_gzip": str(trace_path),
        "note": "Actor raw means, policy-clipped six actions, applied actions, actuator inputs/outputs, saturation flags/runs, contacts, drift, and reward terms are recorded at 8 ms policy steps.",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if manifest["reference_comparison_mismatch_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
