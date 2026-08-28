"""Measure UZ-05 active-joint action sensitivity under gravity."""

import csv
import itertools
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import hydra
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from omni_drones import init_simulation_app


FILE_PATH = os.path.dirname(__file__)
JOINT_NAMES = ("LL", "Lcrank", "RL", "Rcrank")


def _build_candidates(single_amplitude: float, combination_amplitudes):
    candidates = [("zero", [0.0] * 4)]
    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        for sign, suffix in ((1.0, "pos"), (-1.0, "neg")):
            action = [0.0] * 4
            action[joint_idx] = sign * single_amplitude
            candidates.append((f"single_{joint_name}_{suffix}", action))

    for amplitude in combination_amplitudes:
        amplitude_label = f"a{amplitude:.2f}".replace(".", "p")
        for signs in itertools.product((-1.0, 1.0), repeat=4):
            sign_label = "".join("p" if value > 0.0 else "n" for value in signs)
            candidates.append(
                (
                    f"combo_{amplitude_label}_{sign_label}",
                    [amplitude * value for value in signs],
                )
            )
    return candidates


@hydra.main(config_path=FILE_PATH, config_name="stand_sweep", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    single_amplitude = float(cfg.get("sensitivity_single_amplitude", 0.20))
    combination_amplitudes = [
        float(value)
        for value in cfg.get("sensitivity_combination_amplitudes", [0.20, 0.40])
    ]
    candidates = _build_candidates(single_amplitude, combination_amplitudes)
    labels = [item[0] for item in candidates]
    candidate_values = [item[1] for item in candidates]

    cfg.task.env.num_envs = len(candidates)
    cfg.env.num_envs = len(candidates)
    cfg.task.robot.lock_leg_actions = False
    cfg.task.robot.lock_passive_joints = False
    cfg.task.wheel_balance_baseline_enabled = False
    cfg.task.max_init_roll = 0.0
    cfg.task.max_init_pitch = 0.0
    cfg.task.min_height = 0.0
    cfg.task.termination_roll = 3.2
    cfg.task.termination_pitch = 3.2
    cfg.task.max_xy = 100.0
    cfg.task.terminate_on_body_contact = False

    warmup_steps = int(cfg.get("sensitivity_warmup_steps", 180))
    ramp_steps = int(cfg.get("sensitivity_ramp_steps", 180))
    hold_steps = int(cfg.get("sensitivity_hold_steps", 240))
    average_steps = min(int(cfg.get("sensitivity_average_steps", 60)), hold_steps)
    required_steps = warmup_steps + ramp_steps + hold_steps + 1
    cfg.task.env.max_episode_length = max(int(cfg.task.env.max_episode_length), required_steps)
    cfg.env.max_episode_length = cfg.task.env.max_episode_length

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs import resolve_env_class

    env = resolve_env_class(cfg.task.name)(cfg, headless=cfg.headless).eval()
    env.set_seed(cfg.seed)
    env.reset()

    num_envs = len(candidates)
    balance_kp = float(cfg.get("sensitivity_balance_kp", 14.0))
    balance_kd = float(cfg.get("sensitivity_balance_kd", 2.0))
    balance_kv = float(cfg.get("sensitivity_balance_kv", 0.3))
    balance_kx = float(cfg.get("sensitivity_balance_kx", 0.02))
    candidate_tensor = torch.tensor(
        candidate_values, device=env.device, dtype=torch.float32
    ).reshape(num_envs, 1, 4)
    action = torch.zeros(
        num_envs, 1, env.robot.action_spec.shape[-1], device=env.device
    )
    action_td = TensorDict(
        {"agents": {"action": action}}, batch_size=[num_envs], device=env.device
    )
    env.robot.get_state()
    initial_forward_position = env.robot.pos[..., 1].clone()

    def apply_balance_feedback():
        env.robot.get_state()
        wheel_action = (
            balance_kp * env.robot.rpy[..., 0]
            + balance_kd * env.robot.vel_b[..., 3]
            + balance_kv * env.robot.vel_b[..., 1]
            + balance_kx * (env.robot.pos[..., 1] - initial_forward_position)
        ).clamp(-1.0, 1.0)
        action[..., 0] = wheel_action
        action[..., 1] = wheel_action
        return wheel_action

    with torch.no_grad():
        for _ in range(warmup_steps):
            action.zero_()
            apply_balance_feedback()
            env.step(action_td)

    baseline_height = env.robot.pos[:, 0, 2].clone()
    max_abs_roll = env.robot.rpy[:, 0, 0].abs().clone()
    max_abs_pitch = env.robot.rpy[:, 0, 1].abs().clone()
    max_abs_dof_vel = torch.zeros(num_envs, device=env.device)
    min_height = baseline_height.clone()
    max_height = baseline_height.clone()
    finite = torch.ones(num_envs, dtype=torch.bool, device=env.device)
    done_seen = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    max_abs_wheel_action = torch.zeros(num_envs, device=env.device)
    height_sum = torch.zeros(num_envs, device=env.device)
    roll_sum = torch.zeros(num_envs, device=env.device)
    pitch_sum = torch.zeros(num_envs, device=env.device)

    total_control_steps = ramp_steps + hold_steps
    with torch.no_grad():
        for step in range(total_control_steps):
            progress = min(1.0, (step + 1) / max(ramp_steps, 1))
            action.zero_()
            action[..., 2:6] = candidate_tensor * progress
            wheel_action = apply_balance_feedback()
            td = env.step(action_td)

            joint_pos = env.robot.get_joint_positions(clone=True).to(env.device)
            joint_vel = env.robot.get_joint_velocities(clone=True).to(env.device)
            height = env.robot.pos[:, 0, 2]
            roll = env.robot.rpy[:, 0, 0]
            pitch = env.robot.rpy[:, 0, 1]
            done = td[("next", "done")][:, 0]

            finite &= (
                torch.isfinite(joint_pos).all(dim=-1)[:, 0]
                & torch.isfinite(joint_vel).all(dim=-1)[:, 0]
                & torch.isfinite(height)
                & torch.isfinite(roll)
                & torch.isfinite(pitch)
            )
            done_seen |= done
            max_abs_wheel_action = torch.maximum(
                max_abs_wheel_action, wheel_action[:, 0].abs()
            )
            max_abs_dof_vel = torch.maximum(
                max_abs_dof_vel, joint_vel[:, 0].abs().amax(dim=-1)
            )
            max_abs_roll = torch.maximum(max_abs_roll, roll.abs())
            max_abs_pitch = torch.maximum(max_abs_pitch, pitch.abs())
            min_height = torch.minimum(min_height, height)
            max_height = torch.maximum(max_height, height)

            if step >= total_control_steps - average_steps:
                height_sum += height
                roll_sum += roll
                pitch_sum += pitch

    divisor = max(average_steps, 1)
    mean_height = height_sum / divisor
    mean_roll = roll_sum / divisor
    mean_pitch = pitch_sum / divisor
    final_leg_pos = env.robot.leg_pos[:, 0].detach()
    final_leg_target = env.robot.leg_position_targets[:, 0].detach()
    target_error = (final_leg_target - final_leg_pos).abs().amax(dim=-1)

    rows = []
    for idx, (label, candidate) in enumerate(candidates):
        row = {
            "index": idx,
            "label": label,
            "action_LL": candidate[0],
            "action_Lcrank": candidate[1],
            "action_RL": candidate[2],
            "action_Rcrank": candidate[3],
            "baseline_height": float(baseline_height[idx]),
            "mean_height": float(mean_height[idx]),
            "height_delta": float(mean_height[idx] - baseline_height[idx]),
            "min_height": float(min_height[idx]),
            "max_height": float(max_height[idx]),
            "mean_roll": float(mean_roll[idx]),
            "mean_pitch": float(mean_pitch[idx]),
            "max_abs_roll": float(max_abs_roll[idx]),
            "max_abs_pitch": float(max_abs_pitch[idx]),
            "max_abs_dof_velocity": float(max_abs_dof_vel[idx]),
            "max_target_error": float(target_error[idx]),
            "max_abs_wheel_action": float(max_abs_wheel_action[idx]),
            "finite": bool(finite[idx]),
            "done_seen": bool(done_seen[idx]),
        }
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            row[f"position_{joint_name}"] = float(final_leg_pos[idx, joint_idx])
            row[f"target_{joint_name}"] = float(final_leg_target[idx, joint_idx])
        rows.append(row)

    output_path = os.path.abspath(
        str(
            cfg.get(
                "sensitivity_output",
                os.path.join(PROJECT_ROOT, "diagnostics", "uz05_action_height_sensitivity.csv"),
            )
        )
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["finite"]]
    valid_rows.sort(key=lambda row: row["mean_height"], reverse=True)
    print("rank, label, action, mean_height, height_delta, mean_roll, max_abs_roll, target_error, max_wheel_action, done", flush=True)
    for rank, row in enumerate(valid_rows[:15], start=1):
        candidate = [row[f"action_{name}"] for name in JOINT_NAMES]
        print(
            f"{rank}, {row['label']}, {candidate}, {row['mean_height']:.5f}, "
            f"{row['height_delta']:+.5f}, {row['mean_roll']:+.5f}, "
            f"{row['max_abs_roll']:.5f}, {row['max_target_error']:.5f}, "
            f"{row['max_abs_wheel_action']:.5f}, "
            f"{row['done_seen']}",
            flush=True,
        )

    rows_by_label = {row["label"]: row for row in rows}
    position_scale = float(cfg.task.robot.leg_position_scale)
    print("joint, dheight_daction, dheight_dtarget_rad", flush=True)
    for joint_name in JOINT_NAMES:
        positive = rows_by_label[f"single_{joint_name}_pos"]["mean_height"]
        negative = rows_by_label[f"single_{joint_name}_neg"]["mean_height"]
        per_action = (positive - negative) / (2.0 * single_amplitude)
        per_target_rad = per_action / max(position_scale, 1e-6)
        print(
            f"{joint_name}, {per_action:+.6f}, {per_target_rad:+.6f}",
            flush=True,
        )

    print(f"csv: {output_path}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
