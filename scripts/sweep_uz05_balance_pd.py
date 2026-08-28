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


@hydra.main(config_path=FILE_PATH, config_name="stand_sweep", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    kp_values = [float(v) for v in cfg.get("diag_kp_values", [-0.1, -0.2, -0.4, -0.8, -1.2])]
    kd_values = [float(v) for v in cfg.get("diag_kd_values", [-0.02, -0.05, -0.1, -0.2, -0.4])]
    kv_values = [float(v) for v in cfg.get("diag_kv_values", [0.0])]
    kx_values = [float(v) for v in cfg.get("diag_kx_values", [0.0])]
    friction_values = [float(v) for v in cfg.get("diag_friction_values", [0.0])]
    yaw_kp_values = [float(v) for v in cfg.get("diag_yaw_kp_values", [0.0])]
    yaw_kd_values = [float(v) for v in cfg.get("diag_yaw_kd_values", [0.0])]
    command_balance_gain_values = [
        float(v) for v in cfg.get("diag_command_balance_gain_values", [0.0])
    ]
    feedforward_gain_values = [
        float(v) for v in cfg.get("diag_feedforward_gain_values", [0.0])
    ]
    integral_gain_values = [
        float(v) for v in cfg.get("diag_integral_gain_values", [0.0])
    ]
    startup_tilt_values = [
        float(v) for v in cfg.get("diag_startup_tilt_values", [0.0])
    ]
    startup_release_speed_values = [
        float(v) for v in cfg.get("diag_startup_release_speed_values", [0.0])
    ]
    startup_tilt_sign_values = [
        float(v) for v in cfg.get("diag_startup_tilt_sign_values", [-1.0])
    ]
    gains = list(
        itertools.product(
            kp_values,
            kd_values,
            kv_values,
            kx_values,
            friction_values,
            yaw_kp_values,
            yaw_kd_values,
            command_balance_gain_values,
            feedforward_gain_values,
            integral_gain_values,
            startup_tilt_values,
            startup_release_speed_values,
            startup_tilt_sign_values,
        )
    )
    copies = max(1, int(cfg.get("diag_copies", 1)))
    if copies > 1:
        gains = [gain for gain in gains for _ in range(copies)]
    cfg.task.env.num_envs = len(gains)
    cfg.env.num_envs = len(gains)
    cfg.task.robot.lock_leg_actions = True
    use_env_baseline = bool(cfg.get("diag_use_env_baseline", False))
    cfg.task.wheel_balance_baseline_enabled = use_env_baseline
    deterministic_disturbances = bool(cfg.get("diag_disturbances", False))
    cfg.task.disturbances_in_eval = False
    cfg.task.max_init_roll = 0.0
    cfg.task.max_init_pitch = 0.0
    cfg.task.min_height = 0.0
    cfg.task.terminate_on_body_contact = False
    cfg.task.termination_roll = 3.14
    cfg.task.termination_pitch = 3.14
    cfg.task.max_xy = 100.0

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs import resolve_env_class

    env = resolve_env_class(cfg.task.name)(cfg, headless=cfg.headless)
    if bool(cfg.get("diag_training_mode", False)):
        env.train()
    else:
        env.eval()
    env.reset()

    kp = torch.tensor([v[0] for v in gains], device=env.device).reshape(-1, 1)
    kd = torch.tensor([v[1] for v in gains], device=env.device).reshape(-1, 1)
    kv = torch.tensor([v[2] for v in gains], device=env.device).reshape(-1, 1)
    kx = torch.tensor([v[3] for v in gains], device=env.device).reshape(-1, 1)
    friction = torch.tensor([v[4] for v in gains], dtype=torch.float32).reshape(-1, 1)
    yaw_kp = torch.tensor([v[5] for v in gains], device=env.device).reshape(-1, 1)
    yaw_kd = torch.tensor([v[6] for v in gains], device=env.device).reshape(-1, 1)
    command_balance_gain = torch.tensor(
        [v[7] for v in gains], device=env.device
    ).reshape(-1, 1)
    feedforward_gain = torch.tensor(
        [v[8] for v in gains], device=env.device
    ).reshape(-1, 1)
    integral_gain = torch.tensor(
        [v[9] for v in gains], device=env.device
    ).reshape(-1, 1)
    startup_tilt = torch.tensor(
        [v[10] for v in gains], device=env.device
    ).reshape(-1, 1)
    startup_release_speed = torch.tensor(
        [v[11] for v in gains], device=env.device
    ).reshape(-1, 1)
    startup_tilt_sign = torch.tensor(
        [v[12] for v in gains], device=env.device
    ).reshape(-1, 1)
    command_vx = float(cfg.get("diag_command_vx", 0.0))
    forward_axis_sign = float(env.forward_axis_sign)
    wheel_radius = float(cfg.task.robot.get("wheel_radius", 0.06))
    wheel_speed_scale = max(env.robot.max_wheel_velocity * wheel_radius, 1e-6)
    wheel_velocity_feedforward = (
        feedforward_gain
        * command_vx
        / wheel_speed_scale
    )
    env.robot._view.set_friction_coefficients(
        friction.expand(-1, len(env.robot.passive_joint_indices)),
        joint_indices=env.robot.passive_joint_indices.cpu(),
    )
    action = torch.zeros(
        env.num_envs, 1, env.robot.action_spec.shape[-1], device=env.device
    )
    td = TensorDict(
        {"agents": {"action": action}},
        batch_size=[env.num_envs],
        device=env.device,
    )
    steps = int(cfg.get("contact_diag_steps", 400))
    tail_start = int(cfg.get("diag_tail_start", steps // 2))
    max_abs_roll = torch.zeros(env.num_envs, 1, device=env.device)
    max_abs_yaw = torch.zeros_like(max_abs_roll)
    tail_abs_roll = torch.zeros_like(max_abs_roll)
    tail_abs_yaw = torch.zeros_like(max_abs_roll)
    tail_height = torch.zeros_like(max_abs_roll)
    tail_forward_velocity = torch.zeros_like(max_abs_roll)
    saturation_steps = torch.zeros_like(max_abs_roll)
    initial_y = None
    velocity_error_integral = torch.zeros_like(kp)
    command_balance_limit = float(cfg.get("diag_command_balance_limit", 0.06))
    integral_error_limit = float(cfg.get("diag_integral_error_limit", 0.20))

    with torch.no_grad():
        for step in range(steps):
            env.robot.get_state()
            if deterministic_disturbances:
                disturbance_start = int(cfg.get("diag_disturbance_start", 200))
                disturbance_interval = int(cfg.get("diag_disturbance_interval", 250))
                disturbance_index = step - disturbance_start
                if disturbance_index >= 0 and disturbance_index % disturbance_interval == 0:
                    pulse_index = disturbance_index // disturbance_interval
                    pulse_sign = 1.0 if pulse_index % 2 == 0 else -1.0
                    velocities = env.robot.get_velocities(clone=True)
                    velocities[..., 1] += pulse_sign * float(
                        cfg.get("diag_disturbance_velocity", 0.08)
                    )
                    velocities[..., 3] -= pulse_sign * float(
                        cfg.get("diag_disturbance_roll_rate", 0.15)
                    )
                    env.robot.set_velocities(velocities)
                    env.robot.get_state()
            if initial_y is None:
                initial_y = env.robot.pos[..., 1].clone()
            forward_velocity = forward_axis_sign * env.robot.vel_b[..., 1]
            velocity_error = forward_velocity - command_vx
            velocity_error_integral.add_(-velocity_error * env.progress_dt).clamp_(
                -integral_error_limit, integral_error_limit
            )
            roll_target = (
                -command_balance_gain * velocity_error
                - integral_gain * velocity_error_integral
            ).clamp(-command_balance_limit, command_balance_limit)
            startup_active = (
                (startup_tilt > 0.0)
                & (forward_velocity.abs() < startup_release_speed)
                & (abs(command_vx) > 1e-6)
            )
            roll_target = torch.where(
                startup_active,
                startup_tilt_sign
                * torch.sign(torch.full_like(roll_target, command_vx))
                * startup_tilt,
                roll_target,
            )
            command = (
                kp * (env.robot.rpy[..., 0] - roll_target)
                + kd * env.robot.vel_b[..., 3]
                + kv * forward_axis_sign * velocity_error
                + kx * (env.robot.pos[..., 1] - initial_y) * (command_vx == 0.0)
                + wheel_velocity_feedforward
            ).clamp(-1.0, 1.0)
            yaw_command = (
                yaw_kp * torch.atan2(
                    torch.sin(env.robot.rpy[..., 2]),
                    torch.cos(env.robot.rpy[..., 2]),
                )
                + yaw_kd * env.robot.vel_b[..., 5]
            )
            if use_env_baseline:
                action[..., :2] = 0.0
            else:
                action[..., 0] = (command + yaw_command).clamp(-1.0, 1.0)
                action[..., 1] = (command - yaw_command).clamp(-1.0, 1.0)
            env.step(td)
            env.robot.get_state()
            max_abs_roll = torch.maximum(max_abs_roll, env.robot.rpy[..., 0].abs())
            max_abs_yaw = torch.maximum(max_abs_yaw, env.robot.rpy[..., 2].abs())
            saturation_steps += (action[..., :2].abs().amax(dim=-1) > 0.98).float()
            if step >= tail_start:
                tail_abs_roll += env.robot.rpy[..., 0].abs()
                tail_abs_yaw += env.robot.rpy[..., 2].abs()
                tail_height += env.robot.pos[..., 2]
                tail_forward_velocity += (
                    forward_axis_sign * env.robot.vel_b[..., 1]
                )

    tail_count = max(1, steps - tail_start)
    tail_abs_roll /= tail_count
    tail_abs_yaw /= tail_count
    tail_height /= tail_count
    tail_forward_velocity /= tail_count
    saturation_fraction = saturation_steps / max(1, steps)
    displacement = (env.robot.pos[..., 1] - initial_y).abs()
    tracking_error = (tail_forward_velocity - command_vx).abs()
    score = tail_abs_roll + 0.5 * tail_abs_yaw + tracking_error + 0.2 * saturation_fraction
    if command_vx == 0.0:
        score = score + 0.05 * displacement
    order = score.flatten().argsort()
    top_k = min(max(1, int(cfg.get("diag_top_k", len(gains)))), len(gains))

    print(
        "rank,kp,kd,kv,kx,friction,yaw_kp,yaw_kd,command_balance_gain,feedforward_gain,integral_gain,startup_tilt,startup_release_speed,startup_tilt_sign,score,final_roll,tail_abs_roll,"
        "max_abs_roll,tail_abs_yaw,max_abs_yaw,tail_height,tail_vy,tracking_error,"
        "displacement,saturation_fraction",
        flush=True,
    )
    for rank, idx in enumerate(order[:top_k].tolist()):
        print(
            f"{rank},{kp[idx, 0].item():.4f},{kd[idx, 0].item():.4f},"
            f"{kv[idx, 0].item():.4f},{kx[idx, 0].item():.4f},"
            f"{friction[idx, 0].item():.4f},"
            f"{yaw_kp[idx, 0].item():.4f},{yaw_kd[idx, 0].item():.4f},"
            f"{command_balance_gain[idx, 0].item():.4f},"
            f"{feedforward_gain[idx, 0].item():.4f},"
            f"{integral_gain[idx, 0].item():.4f},"
            f"{startup_tilt[idx, 0].item():.4f},"
            f"{startup_release_speed[idx, 0].item():.4f},"
            f"{startup_tilt_sign[idx, 0].item():.4f},"
            f"{score[idx, 0].item():.6f},{env.robot.rpy[idx, 0, 0].item():+.6f},"
            f"{tail_abs_roll[idx, 0].item():.6f},{max_abs_roll[idx, 0].item():.6f},"
            f"{tail_abs_yaw[idx, 0].item():.6f},{max_abs_yaw[idx, 0].item():.6f},"
            f"{tail_height[idx, 0].item():.6f},{tail_forward_velocity[idx, 0].item():.6f},"
            f"{tracking_error[idx, 0].item():.6f},{displacement[idx, 0].item():.6f},"
            f"{saturation_fraction[idx, 0].item():.6f}",
            flush=True,
        )

    simulation_app.close()


if __name__ == "__main__":
    main()
