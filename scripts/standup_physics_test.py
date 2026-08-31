"""Pure-physics stand and pull-up sweep for the two-wheel leg robot.

This script does not use a policy. Legs are commanded with position targets and
zero velocity targets; wheels are commanded with a simple velocity PD balance
law. It is meant to validate the USD, joint signs, drive strengths, and neutral
stand pose before starting another RL run.
"""

import itertools
import csv
import math
import os
import sys

import hydra
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


FILE_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_PATH, ".."))
sys.path.insert(0, PROJECT_ROOT)


def _as_float_list(value, default):
    if value is None:
        value = default
    return [float(v) for v in value]


def _quat_from_axis_angle(angle: torch.Tensor, axis: str) -> torch.Tensor:
    quat = torch.zeros(*angle.shape, 4, device=angle.device)
    half = angle * 0.5
    quat[..., 0] = torch.cos(half)
    axis_to_index = {"roll": 1, "pitch": 2, "yaw": 3}
    quat[..., axis_to_index[axis]] = torch.sin(half)
    return quat


def _build_pose_candidates(values):
    patterns = (
        (1.0, 1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0, 1.0),
    )
    poses = [(0.0, 0.0, 0.0, 0.0)]
    for value in values:
        if abs(value) < 1e-9:
            continue
        for pattern in patterns:
            poses.append(tuple(value * sign for sign in pattern))
    seen = set()
    unique = []
    for pose in poses:
        key = tuple(round(v, 6) for v in pose)
        if key not in seen:
            seen.add(key)
            unique.append(pose)
    return unique


@hydra.main(config_path=os.path.join(FILE_PATH, "../cfg"), config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    test_cfg = cfg.get("standup_test", {})
    pose_values = _as_float_list(test_cfg.get("pose_values"), [0.04, 0.08, 0.12, 0.16, 0.20])
    tilt_values = _as_float_list(test_cfg.get("tilt_values"), [0.0, 0.08, -0.08, 0.14, -0.14])
    kp_values = _as_float_list(test_cfg.get("kp_values"), [-0.25, -0.45, -0.70, -1.00])
    kd_values = _as_float_list(test_cfg.get("kd_values"), [-0.04, -0.08, -0.14, -0.22])
    kv_values = _as_float_list(test_cfg.get("kv_values"), [0.0, -0.10])
    balance_axis = str(test_cfg.get("balance_axis", "roll"))
    if balance_axis not in ("roll", "pitch"):
        raise ValueError("standup_test.balance_axis must be 'roll' or 'pitch'")
    axis_idx = 0 if balance_axis == "roll" else 1
    rate_idx = 3 if balance_axis == "roll" else 4

    pose_candidates = _build_pose_candidates(pose_values)
    raw_cases = itertools.product(tilt_values, kp_values, kd_values, kv_values, pose_candidates)
    cases = [(pose, tilt, kp, kd, kv) for tilt, kp, kd, kv, pose in raw_cases]
    max_cases = int(test_cfg.get("max_cases", 0))
    if max_cases > 0 and max_cases < len(cases):
        sample_idx = torch.linspace(0, len(cases) - 1, max_cases).round().long().tolist()
        cases = [cases[idx] for idx in sample_idx]
    num_envs = len(cases)

    cfg.task.env.num_envs = num_envs
    cfg.env.num_envs = num_envs
    cfg.task.robot.wheel_effort_actions = False
    cfg.task.robot.leg_velocity_actions = True
    cfg.task.robot.lock_leg_actions = False
    cfg.task.robot.lock_passive_joints = bool(test_cfg.get("lock_passive_joints", True))
    cfg.task.robot.leg_action_smoothing = float(test_cfg.get("leg_action_smoothing", 1.0))
    cfg.task.robot.leg_action_rate_limit = float(test_cfg.get("leg_action_rate_limit", 0.0))
    cfg.task.robot.action_smoothing = float(test_cfg.get("wheel_action_smoothing", 1.0))
    cfg.task.robot.action_rate_limit = float(test_cfg.get("wheel_action_rate_limit", 0.0))
    cfg.task.robot.action_accel_limit = 0.0
    cfg.task.max_init_roll = 0.0
    cfg.task.max_init_pitch = 0.0
    cfg.task.min_height = 0.0
    cfg.task.terminate_on_body_contact = False
    cfg.task.termination_roll = 3.14
    cfg.task.termination_pitch = 3.14
    cfg.task.max_xy = 100.0

    steps = int(test_cfg.get("steps", 700))
    tail_start = int(test_cfg.get("tail_start", steps * 2 // 3))
    cfg.task.env.max_episode_length = max(int(cfg.task.env.max_episode_length), steps + 10)
    cfg.env.max_episode_length = cfg.task.env.max_episode_length

    from omni_drones import init_simulation_app

    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs import resolve_env_class

    env = resolve_env_class(cfg.task.name)(cfg, headless=cfg.headless).eval()
    env.set_seed(cfg.seed)
    env.reset()

    device = env.device
    poses = torch.tensor([case[0] for case in cases], dtype=torch.float32, device=device).reshape(num_envs, 1, 4)
    tilts = torch.tensor([case[1] for case in cases], dtype=torch.float32, device=device).reshape(num_envs, 1)
    kp = torch.tensor([case[2] for case in cases], dtype=torch.float32, device=device).reshape(num_envs, 1)
    kd = torch.tensor([case[3] for case in cases], dtype=torch.float32, device=device).reshape(num_envs, 1)
    kv = torch.tensor([case[4] for case in cases], dtype=torch.float32, device=device).reshape(num_envs, 1)

    root_pos, root_rot = env.robot.get_world_poses(clone=True)
    root_rot[:] = _quat_from_axis_angle(tilts, balance_axis)
    env.robot.set_world_poses(root_pos, root_rot)
    env.robot.set_velocities(torch.zeros_like(env.robot.get_velocities(clone=True)))
    env.robot.get_state()

    action = torch.zeros(num_envs, 1, env.robot.action_spec.shape[-1], device=device)
    action_td = TensorDict({"agents": {"action": action}}, batch_size=[num_envs], device=device)

    position_scale = max(float(cfg.task.robot.get("leg_position_scale", 0.1)), 1e-6)
    use_neutral_pose = bool(test_cfg.get("use_neutral_pose", True))
    if use_neutral_pose:
        env.robot.leg_neutral_pos[:] = poses
        leg_action = torch.zeros_like(poses)
    else:
        leg_action = (poses / position_scale).clamp(-1.0, 1.0)
    if bool(test_cfg.get("print_header", True)):
        print(
            "cases="
            f"{num_envs}, steps={steps}, position_scale={position_scale}, "
            f"use_neutral_pose={use_neutral_pose}, "
            f"leg_velocity_actions={cfg.task.robot.leg_velocity_actions}, "
            f"wheel_effort_actions={cfg.task.robot.wheel_effort_actions}",
            flush=True,
        )
        if bool(test_cfg.get("print_dof_debug", False)):
            joint_pos = env.robot.get_joint_positions(clone=True)[0, 0].detach().cpu().tolist()
            joint_vel = env.robot.get_joint_velocities(clone=True)[0, 0].detach().cpu().tolist()
            dof_names = list(env.robot._view._dof_names)
            print("dof_debug_before:", flush=True)
            for name, pos, vel in zip(dof_names, joint_pos, joint_vel):
                print(f"  {name}: pos={pos:+.6f}, vel={vel:+.6f}", flush=True)
            print(f"  leg_indices={env.robot.leg_joint_indices.detach().cpu().tolist()}", flush=True)
            print(f"  passive_indices={env.robot.passive_joint_indices.detach().cpu().tolist()}", flush=True)

    max_abs_roll = torch.zeros(num_envs, 1, device=device)
    max_abs_height_deficit = torch.zeros(num_envs, 1, device=device)
    tail_abs_roll = torch.zeros(num_envs, 1, device=device)
    tail_height = torch.zeros(num_envs, 1, device=device)
    tail_abs_v = torch.zeros(num_envs, 1, device=device)
    tail_leg_error = torch.zeros(num_envs, 1, device=device)
    saturated = torch.zeros(num_envs, 1, device=device)
    done_seen = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
    target_height = float(cfg.task.get("base_height_target", cfg.task.get("reset_height", 0.084)))

    with torch.no_grad():
        for step in range(steps):
            env.robot.get_state()
            forward_v = float(cfg.task.get("forward_axis_sign", 1.0)) * env.robot.vel_b[..., 1]
            command = (
                kp * env.robot.rpy[..., axis_idx]
                + kd * env.robot.vel_b[..., rate_idx]
                + kv * forward_v
            ).clamp(-1.0, 1.0)
            action.zero_()
            action[..., 0] = command
            action[..., 1] = command
            action[..., 2:6] = leg_action
            step_td = env.step(action_td)["next"]
            env.robot.get_state()

            roll_abs = env.robot.rpy[..., axis_idx].abs()
            max_abs_roll = torch.maximum(max_abs_roll, roll_abs)
            height_deficit = (target_height - env.robot.pos[..., 2]).clamp_min(0.0)
            max_abs_height_deficit = torch.maximum(max_abs_height_deficit, height_deficit)
            saturated += (command.abs() > 0.98).float()
            done_seen |= step_td["done"]
            if step >= tail_start:
                tail_abs_roll += roll_abs
                tail_height += env.robot.pos[..., 2]
                tail_abs_v += forward_v.abs()
                actual_leg = env.robot.get_joint_positions(clone=True)[:, 0, env.robot.leg_joint_indices].unsqueeze(1)
                tail_leg_error += (actual_leg - poses).abs().amax(dim=-1)

    tail_count = max(1, steps - tail_start)
    tail_abs_roll /= tail_count
    tail_height /= tail_count
    tail_abs_v /= tail_count
    tail_leg_error /= tail_count
    saturation_fraction = saturated / max(1, steps)
    score = (
        tail_abs_roll
        + 1.4 * max_abs_roll.clamp_min(0.10)
        + 20.0 * max_abs_height_deficit
        + 0.3 * tail_abs_v
        + 0.6 * saturation_fraction
        + 0.5 * tail_leg_error
        + done_seen.float()
    )
    order = score.flatten().argsort()
    top_k = min(int(test_cfg.get("print_top", 20)), num_envs)

    leg_names = list(env.robot.leg_joint_names)
    columns = (
        ["rank", "score"]
        + [f"pose_{name}" for name in leg_names]
        + [
            "init_roll",
            "kp",
            "kd",
            "kv",
            "tail_abs_roll",
            "max_abs_roll",
            "tail_height",
            "tail_abs_v",
            "tail_leg_error",
            "height_deficit",
            "saturation",
            "done",
            "passed",
        ]
    )
    print(",".join(columns), flush=True)
    csv_path = test_cfg.get("csv_path")
    csv_file = open(csv_path, "w", newline="") if csv_path else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(columns)
    pass_count = 0
    rows = []
    for rank, idx in enumerate(order.tolist(), start=1):
        passed = bool(
            tail_abs_roll[idx, 0] < 0.035
            and max_abs_roll[idx, 0] < 0.16
            and tail_height[idx, 0] > target_height - 0.006
            and tail_leg_error[idx, 0] < 0.035
            and saturation_fraction[idx, 0] < 0.25
            and not bool(done_seen[idx, 0])
        )
        pass_count += int(passed)
        pose = poses[idx, 0]
        row = (
            [rank, f"{score[idx, 0].item():.6f}"]
            + [f"{value.item():+.4f}" for value in pose]
            + [
                f"{tilts[idx, 0].item():+.4f}",
                f"{kp[idx, 0].item():+.4f}",
                f"{kd[idx, 0].item():+.4f}",
                f"{kv[idx, 0].item():+.4f}",
                f"{tail_abs_roll[idx, 0].item():.6f}",
                f"{max_abs_roll[idx, 0].item():.6f}",
                f"{tail_height[idx, 0].item():.6f}",
                f"{tail_abs_v[idx, 0].item():.6f}",
                f"{tail_leg_error[idx, 0].item():.6f}",
                f"{max_abs_height_deficit[idx, 0].item():.6f}",
                f"{saturation_fraction[idx, 0].item():.6f}",
                bool(done_seen[idx, 0]),
                passed,
            ]
        )
        rows.append(row)
        if rank <= top_k:
            print(",".join(str(value) for value in row), flush=True)
    if csv_writer:
        csv_writer.writerows(rows)
    if bool(test_cfg.get("print_dof_debug", False)):
        best_idx = int(order[0].item())
        joint_pos = env.robot.get_joint_positions(clone=True)[best_idx, 0].detach().cpu().tolist()
        joint_vel = env.robot.get_joint_velocities(clone=True)[best_idx, 0].detach().cpu().tolist()
        dof_names = list(env.robot._view._dof_names)
        print("dof_debug_best_after:", flush=True)
        for name, pos, vel in zip(dof_names, joint_pos, joint_vel):
            print(f"  {name}: pos={pos:+.6f}, vel={vel:+.6f}", flush=True)
    print(f"pass_in_top_{top_k}: {pass_count}", flush=True)
    if csv_file:
        csv_file.close()
        print(f"csv_path: {csv_path}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
