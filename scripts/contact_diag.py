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


def _rotate_xyzw(points: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """Rotate row-vector points by an xyzw quaternion."""
    q_xyz = quaternion[:3]
    q_w = quaternion[3]
    q_xyz = q_xyz.expand_as(points)
    return points + 2.0 * torch.cross(
        q_xyz, torch.cross(q_xyz, points, dim=-1) + q_w * points, dim=-1
    )


def _collision_points_by_body(usd_path: str, body_names, device):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    cache = UsdGeom.XformCache()
    points_by_body = {}
    for body_name in body_names:
        body = stage.GetPrimAtPath(f"/twowheel_uz05/{body_name}")
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        body_inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        body_points = []
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            mesh_to_body = cache.GetLocalToWorldTransform(prim) * body_inverse
            body_points.extend(mesh_to_body.Transform(Gf.Vec3d(point)) for point in mesh_points)
        if body_points:
            points_by_body[body_name] = torch.tensor(
                body_points, dtype=torch.float32, device=device
            )
    return points_by_body


def _loop_joint_frames(usd_path: str, body_indices, device):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    frames = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        if not prim.GetAttribute("physics:excludeFromArticulation").Get():
            continue
        body0 = stage.GetPrimAtPath(
            prim.GetRelationship("physics:body0").GetTargets()[0]
        ).GetName()
        body1 = stage.GetPrimAtPath(
            prim.GetRelationship("physics:body1").GetTargets()[0]
        ).GetName()
        frames.append(
            (
                prim.GetName(),
                body_indices[body0],
                body_indices[body1],
                torch.tensor(
                    prim.GetAttribute("physics:localPos0").Get(),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.tensor(
                    prim.GetAttribute("physics:localPos1").Get(),
                    dtype=torch.float32,
                    device=device,
                ),
            )
        )
    return frames


@hydra.main(config_path=FILE_PATH, config_name="stand_sweep", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.task.env.num_envs = 1
    cfg.env.num_envs = 1
    cfg.task.robot.lock_leg_actions = True
    cfg.task.max_init_roll = 0.0
    cfg.task.max_init_pitch = 0.0
    cfg.task.min_height = 0.0
    cfg.task.terminate_on_body_contact = False
    cfg.task.termination_roll = 3.14
    cfg.task.termination_pitch = 3.14
    cfg.task.max_xy = 100.0

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs import resolve_env_class


    env = resolve_env_class(cfg.task.name)(cfg, headless=cfg.headless).eval()
    env.reset()

    passive_friction = cfg.get("diag_passive_friction")
    if passive_friction is not None and len(env.robot.passive_joint_indices):
        friction = torch.full(
            (env.num_envs, len(env.robot.passive_joint_indices)),
            float(passive_friction),
            dtype=torch.float32,
        )
        env.robot._view.set_friction_coefficients(
            friction,
            joint_indices=env.robot.passive_joint_indices.cpu(),
        )

    diagnostic_joint_positions = cfg.get("diag_joint_positions")
    if diagnostic_joint_positions is not None:
        values = torch.as_tensor(
            OmegaConf.to_container(diagnostic_joint_positions, resolve=True),
            dtype=torch.float32,
            device=env.device,
        )
        if values.numel() != len(env.robot._view._dof_names):
            raise ValueError(
                "diag_joint_positions must follow the complete articulation DOF order: "
                f"{env.robot._view._dof_names}"
            )
        positions = values.reshape(1, 1, -1)
        velocities = torch.zeros_like(positions)
        env.robot.set_joint_positions(positions)
        env.robot.set_joint_velocities(velocities)
        env.robot._view.set_joint_position_targets(positions)
        env.robot._view.set_joint_velocity_targets(velocities)
        active_positions = positions[..., env.robot.leg_joint_indices]
        env.robot.set_leg_neutral_positions(active_positions)

    steps = int(cfg.get("contact_diag_steps", 240))
    roll_kp = float(cfg.get("diag_roll_kp", 0.0))
    roll_kd = float(cfg.get("diag_roll_kd", 0.0))
    forward_kd = float(cfg.get("diag_forward_kd", 0.0))
    position_kp = float(cfg.get("diag_position_kp", 0.0))
    action_bias = float(cfg.get("diag_action_bias", 0.0))
    left_action_bias = float(cfg.get("diag_left_action_bias", action_bias))
    right_action_bias = float(cfg.get("diag_right_action_bias", action_bias))
    pulse_start = int(cfg.get("diag_pulse_start", -1))
    pulse_end = int(cfg.get("diag_pulse_end", -1))
    pulse_left = float(cfg.get("diag_pulse_left", 0.0))
    pulse_right = float(cfg.get("diag_pulse_right", 0.0))
    action = torch.zeros(env.num_envs, 1, env.robot.action_spec.shape[-1], device=env.device)
    td = TensorDict({"agents": {"action": action}}, batch_size=[env.num_envs], device=env.device)
    print_interval = int(cfg.get("diag_print_interval", 25))
    max_abs_dof_velocity = torch.zeros(
        len(env.robot._view._dof_names), device=env.device
    )
    env.robot.get_state()
    initial_forward_position = env.robot.pos[..., 1].clone()
    with torch.no_grad():
        for step in range(steps):
            env.robot.get_state()
            feedback_action = (
                roll_kp * env.robot.rpy[..., 0]
                + roll_kd * env.robot.vel_b[..., 3]
                + forward_kd * env.robot.vel_b[..., 1]
                + position_kp * (
                    env.robot.pos[..., 1] - initial_forward_position
                )
            ).clamp(-1.0, 1.0)
            pulse_active = pulse_start <= step < pulse_end
            action[..., 0] = (
                left_action_bias + feedback_action + (pulse_left if pulse_active else 0.0)
            ).clamp(-1.0, 1.0)
            action[..., 1] = (
                right_action_bias + feedback_action + (pulse_right if pulse_active else 0.0)
            ).clamp(-1.0, 1.0)
            env.step(td)
            joint_velocity = env.robot.get_joint_velocities(clone=True)[0, 0]
            max_abs_dof_velocity = torch.maximum(
                max_abs_dof_velocity, joint_velocity.abs()
            )
            if step % print_interval == 0 or step == steps - 1:
                env.robot.get_state()
                print(
                    f"step={step:04d}, "
                    f"roll={env.robot.rpy[0, 0, 0].item():+.4f}, "
                    f"yaw={env.robot.rpy[0, 0, 2].item():+.4f}, "
                    f"height={env.robot.pos[0, 0, 2].item():.4f}, "
                    f"vy={env.robot.vel_b[0, 0, 1].item():+.4f}, "
                    f"roll_rate={env.robot.vel_b[0, 0, 3].item():+.4f}, "
                    f"yaw_rate={env.robot.vel_b[0, 0, 5].item():+.4f}, "
                    f"action_l={action[0, 0, 0].item():+.4f}, "
                    f"action_r={action[0, 0, 1].item():+.4f}",
                    flush=True,
                )

    env.robot.get_state()
    print(
        "pose, "
        f"roll={env.robot.rpy[0, 0, 0].item():.4f}, "
        f"pitch={env.robot.rpy[0, 0, 1].item():.4f}, "
        f"yaw={env.robot.rpy[0, 0, 2].item():.4f}, "
        f"height={env.robot.pos[0, 0, 2].item():.4f}, "
        f"vy={env.robot.vel_b[0, 0, 1].item():.4f}, "
        f"roll_rate={env.robot.vel_b[0, 0, 3].item():.4f}, "
        f"action_l={action[0, 0, 0].item():.4f}, "
        f"action_r={action[0, 0, 1].item():.4f}",
        flush=True,
    )

    physics_view = env.robot._view._physics_view
    link_transforms = physics_view.get_link_transforms()[0].detach().to(env.device)
    link_velocities = physics_view.get_link_velocities()[0].detach().to(env.device)
    body_names = list(env.robot._view._body_names)
    collision_points = _collision_points_by_body(
        str(cfg.task.robot.usd_path), body_names, env.device
    )
    print("body, x, y, z, collision_min_z, speed", flush=True)
    for body_idx, body_name in enumerate(body_names):
        transform = link_transforms[body_idx]
        min_z = float("nan")
        points = collision_points.get(body_name)
        if points is not None:
            world_points = _rotate_xyzw(points, transform[3:]) + transform[:3]
            min_z = float(world_points[:, 2].amin())
        speed = float(link_velocities[body_idx].norm())
        print(
            f"{body_name}, {transform[0].item():+.5f}, {transform[1].item():+.5f}, "
            f"{transform[2].item():+.5f}, {min_z:+.5f}, {speed:.5f}",
            flush=True,
        )

    joint_positions = env.robot.get_joint_positions(clone=True)[0, 0]
    joint_velocities = env.robot.get_joint_velocities(clone=True)[0, 0]
    print("joint, position, velocity, peak_abs_velocity", flush=True)
    for joint_idx, joint_name in enumerate(env.robot._view._dof_names):
        print(
            f"{joint_name}, {joint_positions[joint_idx].item():+.5f}, "
            f"{joint_velocities[joint_idx].item():+.5f}, "
            f"{max_abs_dof_velocity[joint_idx].item():.5f}",
            flush=True,
        )

    print("loop_joint, anchor_residual_m", flush=True)
    for name, body0, body1, local0, local1 in _loop_joint_frames(
        str(cfg.task.robot.usd_path), env.robot._view._body_indices, env.device
    ):
        transform0 = link_transforms[body0]
        transform1 = link_transforms[body1]
        anchor0 = _rotate_xyzw(local0, transform0[3:]) + transform0[:3]
        anchor1 = _rotate_xyzw(local1, transform1[3:]) + transform1[:3]
        print(f"{name}, {(anchor0 - anchor1).norm().item():.8f}", flush=True)

    get_contacts = getattr(env.robot._view, "get_net_contact_forces", None)
    if get_contacts is None:
        print("contact_forces_unavailable", flush=True)
        simulation_app.close()
        return

    forces = get_contacts(clone=True)[0, 0].detach().cpu()
    names = list(getattr(env.robot._view, "_body_names", []))
    print("contacts", flush=True)
    for idx, force in enumerate(forces):
        norm = float(force.norm())
        if norm <= 0.05:
            continue
        name = names[idx] if idx < len(names) else str(idx)
        values = ", ".join(f"{float(v):.4f}" for v in force.tolist())
        print(f"  {idx}: {name}: [{values}], norm={norm:.4f}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
