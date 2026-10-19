#!/usr/bin/env python3
"""Run the physical and observation gates before any formal PPO training."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from uz05.pure_ppo_contract import model_xml_sha256, source_fingerprint
from uz05.pure_ppo_stand import (
    ACTION_DIM,
    ACTION_ORDER,
    EPISODE_STEPS,
    MOTOR_TICKS_PER_POLICY_STEP,
    PurePPOStandActuator,
    PurePPOStandEnv,
    Q_REF,
    RESET_HEIGHT_M,
    map_stand_offset_action,
)
from uz05.spec import EnvParams


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "pure_ppo_stand_v1" / "preflight.json"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _one_step(action: np.ndarray) -> dict[str, Any]:
    env = PurePPOStandEnv(seed=11)
    env.reset(seed=11, options={"initial_pitch_rad": 0.0, "initial_pitch_rate_rad_s": 0.0})
    before = env._state()
    _, reward, terminated, truncated, info = env.step(action)
    after = env._state()
    result = {
        "action": np.asarray(action, dtype=float).tolist(),
        "wheel_target_rad_s": info["wheel_target_rad_s"],
        "wheel_current_a": info["wheel_current_a"],
        "wheel_torque_nm": info["wheel_torque_nm"],
        "wheel_joint_velocity_rad_s": env.sim.wheel_velocities().tolist(),
        "leg_target_rad": info["joint_target_rad"],
        "leg_torque_nm": info["joint_torque_nm"],
        "body_vx_delta_m_s": after["vx"] - before["vx"],
        "pitch_rate_delta_rad_s": after["pitch_rate"] - before["pitch_rate"],
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "finite": bool(info["sim_finite"]),
    }
    env.close()
    return result


def run() -> dict[str, Any]:
    started = time.perf_counter()
    checks: dict[str, bool] = {}
    env = None
    try:
        # A fresh model compile and closed-loop reset on every iteration.
        reset_rows = []
        for seed in range(100):
            sample = PurePPOStandEnv(seed=seed)
            obs, info = sample.reset(seed=seed)
            state = sample._state()
            residual = info["reset_equality_residual_max_m"]
            _assert(np.isfinite(obs).all(), f"seed {seed}: nonfinite observation")
            _assert(obs.shape == sample.observation_space.shape, f"seed {seed}: bad observation shape")
            _assert(residual <= 1e-8, f"seed {seed}: equality residual {residual}")
            _assert(np.all(state["wheel_hit"] > 0.5), f"seed {seed}: both wheels must contact")
            _assert(state["body_hit"] == 0.0, f"seed {seed}: chassis contact on reset")
            _assert(np.allclose(sample.sim.joint_positions(), Q_REF, atol=1e-12), "bad active reset pose")
            _assert(np.max(np.abs(sample.sim.leg_lengths() - 0.18413)) < 5e-4, "reset leg length mismatch")
            _assert(not sample.params.domain_randomization.enabled, "domain randomization enabled")
            _assert(not sample.params.noise.enabled, "observation noise enabled")
            _assert(not any(np.any(np.asarray(v) != 0.0) for v in sample.observation_encoder.noise.values()), "nonzero observation noise")
            reset_rows.append({
                "seed": seed,
                "pitch_rad": info["initial_pitch_rad"],
                "pitch_rate_rad_s": info["initial_pitch_rate_rad_s"],
                "height_m": state["height"],
                "leg_length_m": sample.sim.leg_lengths().tolist(),
                "left_right_wheel_contact": state["wheel_hit"].tolist(),
                "body_contact": state["body_hit"],
                "equality_residual_max_m": residual,
                "observation_noise_width_max": info["observation_noise_width_max"],
            })
            sample.close()
        checks["100_fresh_model_loads_and_resets"] = True

        env = PurePPOStandEnv(seed=123)
        model = env.sim.model
        data = env.sim.data
        wheel_joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ("left_J_link6_wheel", "right_J_link6_wheel")
        ]
        ground_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        physical = {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "neq": int(model.neq),
            "timestep_s": float(model.opt.timestep),
            "gravity_m_s2": model.opt.gravity.tolist(),
            "total_mass_kg": float(model.body_mass.sum()),
            "wheel_radius_m": env.sim.wheel_radius,
            "wheel_joint_axes_world": [data.xaxis[jid].tolist() for jid in wheel_joint_ids],
            "reset_height_m": RESET_HEIGHT_M,
            "active_joint_q_ref_rad": Q_REF.tolist(),
            "closed_loop_passive_qpos_rad": env._passive_reset_qpos.tolist(),
            "contact_friction_ground": model.geom_friction[ground_geom_id].tolist(),
        }
        checks["mjcf_compiles_and_dimensions"] = (
            physical["nq"] == 21 and physical["nv"] == 20 and physical["nu"] == 6 and physical["neq"] == 4
        )
        _assert(checks["mjcf_compiles_and_dimensions"], f"unexpected model dimensions: {physical}")
        _assert(abs(physical["timestep_s"] - 0.002) < 1e-12, "MuJoCo motor timestep must be 2 ms")
        _assert(MOTOR_TICKS_PER_POLICY_STEP == 4, "policy/motor decimation must be 4")

        # The qvel angular block is local. Its world conversion must round-trip
        # to local rates for nonzero roll, pitch, yaw, and all rate components.
        frame_cases = []
        for roll, pitch in ((0.13, np.deg2rad(5)), (-0.08, -np.deg2rad(5))):
            yaw = 0.71
            quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
            data.qpos[3:7] = [quat_xyzw[3], *quat_xyzw[:3]]
            data.qvel[3:6] = [0.04, 0.10 if pitch > 0 else -0.10, -0.03]
            mujoco.mj_forward(model, data)
            recovered = env.sim.body_frame(env.sim.base_ang_vel_world)
            _assert(np.allclose(recovered, data.qvel[3:6], atol=1e-12), "angular rate frame conversion mismatch")
            rpy = env._rpy(env.sim)
            _assert(np.sign(rpy[1]) == np.sign(pitch), "pitch sign mismatch")
            _assert(np.isclose(recovered[1], data.qvel[4], atol=1e-12), "pitch-rate sign mismatch")
            frame_cases.append({
                "rpy_rad": rpy.tolist(),
                "qvel_local_rad_s": data.qvel[3:6].tolist(),
                "omega_world_rad_s": env.sim.base_ang_vel_world.tolist(),
                "recovered_body_rate_rad_s": recovered.tolist(),
                "gravity_projection_body": env._get_observation()[:3].tolist(),
            })
        checks["free_joint_world_body_rate_frames"] = True

        # Zero action targets q_ref with no initial joint PD torque. The old
        # absolute-zero target would demand nearly 20 Nm per joint here.
        state_joint = env.sim.joint_positions()
        actuator = PurePPOStandActuator(EnvParams())
        zero_control = actuator.compute(
            np.zeros(ACTION_DIM), joint_pos=state_joint, joint_vel=np.zeros(4),
            wheel_vel=np.zeros(2),
        )
        legacy_zero_target_torque = np.clip(-state_joint * EnvParams().joint.kp,
                                            -EnvParams().joint.torque_limit,
                                            EnvParams().joint.torque_limit)
        _assert(np.allclose(zero_control["wheel_target"], 0.0), "zero action wheel target is nonzero")
        _assert(np.allclose(zero_control["joint_target"], Q_REF), "zero action leg target is not q_ref")
        _assert(np.max(np.abs(zero_control["joint_torque"])) < 1e-12, "new zero-action pose has PD error")

        # Coordinate-map tests are independent of current pose; basis actions
        # verify all six channels remain connected to exactly their target.
        basis_rows = []
        for index in range(ACTION_DIM):
            action = np.zeros(ACTION_DIM)
            action[index] = 0.1
            clipped, wheel_target, joint_target = map_stand_offset_action(action)
            altered = np.flatnonzero(np.abs(wheel_target) > 0) if index < 2 else np.flatnonzero(np.abs(joint_target - Q_REF) > 0)
            _assert(len(altered) == 1, f"action channel {index} did not map to one actuator")
            basis_rows.append({
                "channel": ACTION_ORDER[index],
                "target_change_index": int(altered[0]),
                "normalized_action": clipped.tolist(),
                "wheel_target_rad_s": wheel_target.tolist(),
                "joint_target_rad": joint_target.tolist(),
            })
        checks["six_independent_action_channels"] = True

        # Short physical probes: current, torque, wheel velocity, chassis vx,
        # and pitch-rate signs are read from MuJoCo, not inferred from XML names.
        probes = {
            "both_plus_0_1": np.array([0.1, 0.1, 0, 0, 0, 0]),
            "both_minus_0_1": np.array([-0.1, -0.1, 0, 0, 0, 0]),
            "both_plus_0_3": np.array([0.3, 0.3, 0, 0, 0, 0]),
            "both_minus_0_3": np.array([-0.3, -0.3, 0, 0, 0, 0]),
            "left_plus_0_1": np.array([0.1, 0, 0, 0, 0, 0]),
            "left_minus_0_1": np.array([-0.1, 0, 0, 0, 0, 0]),
            "right_plus_0_1": np.array([0, 0.1, 0, 0, 0, 0]),
            "right_minus_0_1": np.array([0, -0.1, 0, 0, 0, 0]),
        }
        physical_probes = {name: _one_step(action) for name, action in probes.items()}
        for name, sign in (("both_plus_0_1", 1), ("both_minus_0_1", -1),
                           ("both_plus_0_3", 1), ("both_minus_0_3", -1)):
            row = physical_probes[name]
            _assert(row["finite"] and not row["terminated"], f"{name}: numerical/physical failure")
            _assert(np.all(np.sign(row["wheel_current_a"]) == sign), f"{name}: current sign mismatch")
            _assert(np.all(np.sign(row["wheel_torque_nm"]) == sign), f"{name}: torque sign mismatch")
            _assert(np.sign(row["body_vx_delta_m_s"]) == -sign, f"{name}: observed body vx response sign mismatch")
        _assert(np.allclose(physical_probes["both_plus_0_1"]["wheel_current_a"], [1, 1], atol=1e-9), "0.1 action current mismatch")
        _assert(np.allclose(physical_probes["both_plus_0_3"]["wheel_current_a"], [3, 3], atol=1e-9), "0.3 action current mismatch")
        _assert(max(np.max(np.abs(row["wheel_current_a"])) for row in physical_probes.values()) < EnvParams().wheel.current_limit_a, "wheel action probes saturated")
        checks["wheel_action_to_current_torque_wheel_velocity_body_vx"] = True

        # Pure PPO step timing and all outputs remain finite on an actual step.
        timing_env = PurePPOStandEnv(seed=99)
        _, timing_reset_info = timing_env.reset(
            seed=99,
            options={"initial_pitch_rad": 0.0, "initial_pitch_rate_rad_s": 0.0},
        )
        t0 = float(timing_env.sim.data.time)
        obs, reward, terminated, truncated, info = timing_env.step(np.zeros(ACTION_DIM, dtype=np.float32))
        elapsed = float(timing_env.sim.data.time - t0)
        _assert(np.isclose(elapsed, 0.008, atol=1e-12), f"expected 8 ms policy step, got {elapsed}")
        _assert(np.isfinite(obs).all() and np.isfinite(reward), "nonfinite complete-step output")
        _assert(info["policy_steps"] == 1, "policy step counter mismatch")
        _assert(
            not timing_reset_info["domain_randomization_enabled"]
            and not timing_reset_info["observation_noise_enabled"],
            "noise/DR flags not disabled",
        )
        checks["complete_step_125hz_over_500hz_four_ticks"] = True
        timing_env.close()

        checks["zero_action_qref_and_no_new_pose_offset_torque"] = True
        return {
            "status": "PASS",
            "task_id": "pure_ppo_stand_v1",
            "model_xml_sha256": model_xml_sha256(),
            "source_fingerprint_sha256": source_fingerprint(),
            "checks": checks,
            "physical_model": physical,
            "reset_trials": reset_rows,
            "reset_summary": {
                "trials": len(reset_rows),
                "max_constraint_residual_m": max(row["equality_residual_max_m"] for row in reset_rows),
                "height_range_m": [min(row["height_m"] for row in reset_rows), max(row["height_m"] for row in reset_rows)],
                "min_left_right_contact": [min(row["left_right_wheel_contact"][i] for row in reset_rows) for i in range(2)],
                "body_contact_episodes": sum(row["body_contact"] > 0.5 for row in reset_rows),
                "observation_noise_width_max": max(row["observation_noise_width_max"] for row in reset_rows),
            },
            "angular_frame_cases": frame_cases,
            "zero_action": {
                "wheel_target_rad_s": zero_control["wheel_target"].tolist(),
                "joint_target_rad": zero_control["joint_target"].tolist(),
                "joint_torque_nm": zero_control["joint_torque"].tolist(),
                "legacy_absolute_zero_target_torque_nm": legacy_zero_target_torque.tolist(),
            },
            "action_basis_probes": basis_rows,
            "wheel_physical_probes": physical_probes,
            "policy_steps": EPISODE_STEPS,
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run()
    except Exception as exc:
        report = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "model_xml_sha256": model_xml_sha256(),
            "source_fingerprint_sha256": source_fingerprint(),
        }
        print(report["error"], file=sys.stderr)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
