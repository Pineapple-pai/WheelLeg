"""Slowly sweep the four UZ-05 input shafts without breaking loop closure."""

import ast
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
DEFAULT_TARGETS = [
    [0.06, 0.00, 0.00, 0.00],
    [0.00, 0.06, 0.00, 0.00],
    [0.00, 0.00, 0.06, 0.00],
    [0.00, 0.00, 0.00, 0.06],
    [0.05, -0.05, 0.05, -0.05],
    [-0.05, 0.05, -0.05, 0.05],
]


def _targets(value):
    if value is None:
        return DEFAULT_TARGETS
    if isinstance(value, str):
        return ast.literal_eval(value)
    return OmegaConf.to_container(value, resolve=True)


@hydra.main(config_path=FILE_PATH, config_name="stand_sweep", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    targets = _targets(cfg.get("closed_loop_targets"))
    if any(len(target) != 4 for target in targets):
        raise ValueError("Every closed-loop target must contain four input-shaft angles.")

    cfg.task.env.num_envs = len(targets)
    cfg.env.num_envs = len(targets)
    cfg.task.robot.lock_leg_actions = False
    cfg.task.robot.lock_passive_joints = False
    cfg.task.robot.max_wheel_velocity = 0.0
    cfg.task.robot.rigid_props = {"disable_gravity": True}
    cfg.task.robot.articulation_props = {
        "solver_position_iteration_count": 16,
        "solver_velocity_iteration_count": 4,
    }
    cfg.task.sim.gravity = [0.0, 0.0, 0.0]
    cfg.sim.gravity = [0.0, 0.0, 0.0]
    cfg.task.reset_height = 0.5
    cfg.task.max_init_roll = 0.0
    cfg.task.max_init_pitch = 0.0
    cfg.task.min_height = 0.0
    cfg.task.termination_roll = 3.2
    cfg.task.termination_pitch = 3.2
    cfg.task.max_xy = 100.0
    cfg.task.terminate_on_body_contact = False

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs import resolve_env_class

    env = resolve_env_class(cfg.task.name)(cfg, headless=cfg.headless).eval()
    env.set_seed(cfg.seed)
    env.reset()

    targets = torch.as_tensor(targets, device=env.device).float().reshape(len(targets), 1, 4)
    scale = float(cfg.task.robot.leg_position_scale)
    ramp_steps = int(cfg.get("closed_loop_ramp_steps", 180))
    hold_steps = int(cfg.get("closed_loop_hold_steps", 120))
    warmup_steps = int(cfg.get("closed_loop_warmup_steps", 120))
    print_every = int(cfg.get("closed_loop_print_every", 30))
    action = torch.zeros(len(targets), 1, env.robot.action_spec.shape[-1], device=env.device)
    action_td = TensorDict({"agents": {"action": action}}, batch_size=[len(targets)], device=env.device)
    dof_names = list(env.robot._view._dof_names)
    max_abs_velocity = torch.zeros(len(targets), len(dof_names), device=env.device)

    with torch.no_grad():
        for _ in range(warmup_steps):
            env.step(action_td)

    print("idx, step, target, inputs, max_abs_dof_vel, height, roll, pitch, done, finite", flush=True)
    with torch.no_grad():
        for step in range(ramp_steps + hold_steps):
            progress = min(1.0, (step + 1) / ramp_steps)
            action.zero_()
            action[..., 2:6] = (targets * progress / scale).clamp(-1.0, 1.0)
            td = env.step(action_td)
            joint_pos = env.robot.get_joint_positions(clone=True).to(env.device)
            joint_vel = env.robot.get_joint_velocities(clone=True).to(env.device)
            max_abs_velocity = torch.maximum(max_abs_velocity, joint_vel.abs()[:, 0])
            obs = td[("next", "agents", "observation")][:, 0]
            done = td[("next", "done")][:, 0]
            if step % print_every == 0 or step == ramp_steps + hold_steps - 1:
                inputs = joint_pos[..., env.robot.leg_joint_indices][:, 0]
                for idx, target in enumerate(targets[:, 0].tolist()):
                    finite = bool(torch.isfinite(joint_pos[idx]).all() and torch.isfinite(joint_vel[idx]).all())
                    values = [round(float(v), 5) for v in inputs[idx].tolist()]
                    print(
                        f"{idx}, {step}, {target}, {values}, "
                        f"{max_abs_velocity[idx].max().item():.5f}, {obs[idx, 3].item():.5f}, "
                        f"{obs[idx, 0].item():.5f}, {obs[idx, 1].item():.5f}, "
                        f"{bool(done[idx].item())}, {finite}",
                        flush=True,
                    )

    print("peak_dof_velocity_rad_s", flush=True)
    for idx, target in enumerate(targets[:, 0].tolist()):
        order = torch.argsort(max_abs_velocity[idx], descending=True)
        peaks = ", ".join(
            f"{dof_names[int(i)]}={max_abs_velocity[idx, i].item():.5f}"
            for i in order[:8]
        )
        print(f"{idx}, {target}: {peaks}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
