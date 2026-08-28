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


DEFAULT_CANDIDATES = [
    [0.0, 0.0, 0.0, 0.0],
    [-0.20, 0.20, -0.20, 0.20],
    [-0.40, 0.40, -0.40, 0.40],
    [-0.60, 0.60, -0.60, 0.60],
    [0.20, -0.20, 0.20, -0.20],
    [0.40, -0.40, 0.40, -0.40],
    [0.60, -0.60, 0.60, -0.60],
    [-0.40, -0.40, -0.40, -0.40],
    [0.40, 0.40, 0.40, 0.40],
]


def _parse_candidates(value):
    if value is None:
        return DEFAULT_CANDIDATES
    if isinstance(value, str):
        return ast.literal_eval(value)
    return OmegaConf.to_container(value, resolve=True)


@hydra.main(config_path=FILE_PATH, config_name="stand_sweep", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    candidates = _parse_candidates(cfg.get("stand_sweep", None))
    if not candidates:
        raise ValueError("stand_sweep must contain at least one joint target.")
    for candidate in candidates:
        if len(candidate) != 4:
            raise ValueError(f"Each stand_sweep candidate must have 4 values: {candidate}")

    cfg.task.env.num_envs = len(candidates)
    cfg.env.num_envs = len(candidates)
    cfg.task.robot.lock_leg_actions = True
    cfg.task.robot.lock_passive_joints = bool(cfg.task.robot.get("lock_passive_joints", False))
    cfg.task.robot.max_wheel_velocity = 0.0
    cfg.task.max_init_roll = 0.0
    cfg.task.max_init_pitch = 0.0
    cfg.task.min_height = 0.0
    cfg.task.terminate_on_body_contact = False

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs import resolve_env_class


    env_class = resolve_env_class(cfg.task.name)
    env = env_class(cfg, headless=cfg.headless).eval()
    env.set_seed(cfg.seed)

    physics_view = getattr(env.robot._view, "_physics_view", None)
    if physics_view is not None:
        dof_names = list(getattr(env.robot._view, "_dof_names", []))
        indices = env.robot.leg_joint_indices.detach().cpu().tolist()
        print("leg_drive_gains", flush=True)
        for getter_name, label in (
            ("get_dof_stiffnesses", "stiffness"),
            ("get_dof_dampings", "damping"),
            ("get_dof_max_forces", "max_force"),
            ("get_dof_max_velocities", "max_velocity"),
        ):
            getter = getattr(physics_view, getter_name, None)
            if getter is None:
                continue
            values = getter()
            print(
                f"  {label}: "
                + ", ".join(
                    f"{dof_names[i]}={float(values[0, i].item()):.6g}"
                    for i in indices
                ),
                flush=True,
            )

    candidate_tensor = torch.as_tensor(candidates, device=env.device).float().reshape(len(candidates), 1, 4)
    td = env.reset()

    joint_pos = env.robot.get_joint_positions(clone=True).to(env.device)
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[..., env.robot.leg_joint_indices] = candidate_tensor
    env.robot.set_joint_positions(joint_pos, torch.arange(env.num_envs, device=env.device))
    env.robot.set_joint_velocities(joint_vel, torch.arange(env.num_envs, device=env.device))
    env.robot.set_leg_neutral_positions(candidate_tensor)
    env.robot._view.set_joint_position_targets(candidate_tensor, joint_indices=env.robot.leg_joint_indices)
    env.robot._view.set_joint_velocity_targets(
        torch.zeros_like(candidate_tensor),
        joint_indices=env.robot.leg_joint_indices,
    )

    steps = int(cfg.get("sweep_steps", 240))
    print_every = int(cfg.get("sweep_print_every", 60))
    action = torch.zeros(env.num_envs, 1, env.robot.action_spec.shape[-1], device=env.device)
    action_td = TensorDict({"agents": {"action": action}}, batch_size=[env.num_envs], device=env.device)

    print("idx, target, step, height, roll, pitch, yaw, leg_pos, done, finite", flush=True)
    with torch.no_grad():
        for step in range(steps):
            td = env.step(action_td)
            obs = td[("next", "agents", "observation")][:, 0]
            done = td[("next", "done")][:, 0]
            if step % print_every == 0 or step == steps - 1:
                leg_pos = env.robot.get_joint_positions(clone=True).to(env.device)[
                    ..., env.robot.leg_joint_indices
                ][:, 0]
                for idx, candidate in enumerate(candidates):
                    row = obs[idx]
                    finite = bool(torch.isfinite(row[:4]).all().item())
                    leg = [round(float(v), 4) for v in leg_pos[idx].detach().cpu().tolist()]
                    print(
                        f"{idx}, {candidate}, {step}, "
                        f"{row[3].item():.4f}, {row[0].item():.4f}, "
                        f"{row[1].item():.4f}, {row[2].item():.4f}, "
                        f"{leg}, "
                        f"{bool(done[idx].item())}, {finite}",
                        flush=True,
                    )

    simulation_app.close()


if __name__ == "__main__":
    main()
