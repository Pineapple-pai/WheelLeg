"""Search symmetric UZ-05 closed-loop poses with the chassis held fixed."""

import csv
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
WHEEL_NAMES = ("Lwhl", "Rwhl")


def _rotate_xyzw(points: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    q_xyz = quaternion[..., :3].unsqueeze(-2).expand_as(points)
    q_w = quaternion[..., 3:].unsqueeze(-2)
    return points + 2.0 * torch.cross(
        q_xyz, torch.cross(q_xyz, points, dim=-1) + q_w * points, dim=-1
    )


def _collision_points_by_body(usd_path: str, body_names, device):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    cache = UsdGeom.XformCache()
    result = {}
    for body_name in body_names:
        body = stage.GetPrimAtPath(f"/twowheel_uz05/{body_name}")
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        body_inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        points = []
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mesh_to_body = cache.GetLocalToWorldTransform(prim) * body_inverse
            mesh_points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            points.extend(mesh_to_body.Transform(Gf.Vec3d(point)) for point in mesh_points)
        if points:
            result[body_name] = torch.tensor(points, dtype=torch.float32, device=device)
    return result


@hydra.main(config_path=FILE_PATH, config_name="stand_sweep", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    samples = int(cfg.get("stand_search_samples", 9))
    action_limit = float(cfg.get("stand_search_action_limit", 1.0))
    values = torch.linspace(-action_limit, action_limit, samples).tolist()
    candidates = [(ll, crank) for ll in values for crank in values]
    num_envs = len(candidates)

    cfg.task.env.num_envs = num_envs
    cfg.env.num_envs = num_envs
    cfg.task.robot.lock_leg_actions = False
    cfg.task.robot.lock_passive_joints = False
    cfg.task.robot.max_wheel_velocity = 0.0
    cfg.task.robot.leg_action_smoothing = 1.0
    cfg.task.robot.leg_action_rate_limit = 0.0
    cfg.task.robot.rigid_props = {"disable_gravity": True}
    cfg.task.robot.articulation_props = {
        "solver_position_iteration_count": 32,
        "solver_velocity_iteration_count": 8,
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

    ramp_steps = int(cfg.get("stand_search_ramp_steps", 300))
    hold_steps = int(cfg.get("stand_search_hold_steps", 300))
    cfg.task.env.max_episode_length = max(
        int(cfg.task.env.max_episode_length), ramp_steps + hold_steps + 10
    )
    cfg.env.max_episode_length = cfg.task.env.max_episode_length

    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs import resolve_env_class

    env = resolve_env_class(cfg.task.name)(cfg, headless=cfg.headless).eval()
    env.set_seed(cfg.seed)
    env.reset()

    anchor_pos, anchor_rot = env.robot.get_world_poses(clone=True)
    zero_root_velocity = torch.zeros_like(env.robot.get_velocities(clone=True))
    candidate_tensor = torch.tensor(
        [[ll, crank, ll, crank] for ll, crank in candidates],
        dtype=torch.float32,
        device=env.device,
    ).reshape(num_envs, 1, 4)
    action = torch.zeros(num_envs, 1, env.robot.action_spec.shape[-1], device=env.device)
    action_td = TensorDict(
        {"agents": {"action": action}}, batch_size=[num_envs], device=env.device
    )
    max_abs_velocity = torch.zeros(
        num_envs, len(env.robot._view._dof_names), device=env.device
    )
    done_seen = torch.zeros(num_envs, dtype=torch.bool, device=env.device)

    with torch.no_grad():
        for step in range(ramp_steps + hold_steps):
            progress = min(1.0, (step + 1) / max(ramp_steps, 1))
            action.zero_()
            action[..., 2:6] = candidate_tensor * progress
            td = env.step(action_td)
            env.robot.set_world_poses(anchor_pos, anchor_rot)
            env.robot.set_velocities(zero_root_velocity)
            velocity = env.robot.get_joint_velocities(clone=True)[:, 0]
            max_abs_velocity = torch.maximum(max_abs_velocity, velocity.abs())
            done_seen |= td[("next", "done")][:, 0]

    physics_view = env.robot._view._physics_view
    link_transforms = physics_view.get_link_transforms().detach().to(env.device)
    body_names = list(env.robot._view._body_names)
    body_indices = dict(env.robot._view._body_indices)
    points_by_body = _collision_points_by_body(
        str(cfg.task.robot.usd_path), body_names, env.device
    )
    min_z_by_body = {}
    for body_idx, body_name in enumerate(body_names):
        points = points_by_body.get(body_name)
        if points is None:
            continue
        transform = link_transforms[:, body_idx]
        expanded_points = points.unsqueeze(0).expand(num_envs, -1, -1)
        world_points = _rotate_xyzw(expanded_points, transform[:, 3:]) + transform[:, None, :3]
        min_z_by_body[body_name] = world_points[..., 2].amin(dim=-1)

    wheel_min = torch.stack([min_z_by_body[name] for name in WHEEL_NAMES], dim=-1)
    nonwheel_names = [name for name in min_z_by_body if name not in WHEEL_NAMES]
    nonwheel_min = torch.stack(
        [min_z_by_body[name] for name in nonwheel_names], dim=-1
    ).amin(dim=-1)
    wheel_low = wheel_min.amin(dim=-1)
    ground_clearance = nonwheel_min - wheel_low
    wheel_height_mismatch = (wheel_min[:, 0] - wheel_min[:, 1]).abs()

    joint_pos = env.robot.get_joint_positions(clone=True)[:, 0]
    joint_vel = env.robot.get_joint_velocities(clone=True)[:, 0]
    active_pos = joint_pos[:, env.robot.leg_joint_indices]
    active_target = env.robot.leg_position_targets[:, 0]
    active_error = (active_target - active_pos).abs().amax(dim=-1)
    passive_velocity = joint_vel[:, env.robot.passive_joint_indices].abs().amax(dim=-1)
    passive_peak = max_abs_velocity[:, env.robot.passive_joint_indices].amax(dim=-1)
    finite = torch.isfinite(joint_pos).all(dim=-1) & torch.isfinite(joint_vel).all(dim=-1)

    rows = []
    dof_names = list(env.robot._view._dof_names)
    for idx, (ll_action, crank_action) in enumerate(candidates):
        row = {
            "index": idx,
            "action_LL_RL": ll_action,
            "action_Lcrank_Rcrank": crank_action,
            "wheel_support_height": float(anchor_pos[idx, 0, 2] - wheel_low[idx]),
            "nonwheel_clearance": float(ground_clearance[idx]),
            "wheel_height_mismatch": float(wheel_height_mismatch[idx]),
            "active_target_error": float(active_error[idx]),
            "final_passive_velocity": float(passive_velocity[idx]),
            "peak_passive_velocity": float(passive_peak[idx]),
            "finite": bool(finite[idx]),
            "done_seen": bool(done_seen[idx]),
        }
        for joint_idx, joint_name in enumerate(dof_names):
            row[f"position_{joint_name}"] = float(joint_pos[idx, joint_idx])
        rows.append(row)

    output_path = os.path.abspath(
        str(cfg.get("stand_search_output", "diagnostics/uz05_stand_pose_search.csv"))
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [
        row for row in rows
        if row["finite"]
        and not row["done_seen"]
        and row["active_target_error"] < 0.02
        and row["final_passive_velocity"] < 0.05
    ]
    valid_rows.sort(
        key=lambda row: (
            row["nonwheel_clearance"] >= 0.005,
            -row["wheel_height_mismatch"],
            row["nonwheel_clearance"],
        ),
        reverse=True,
    )
    print(
        "rank, LL_RL, crank, support_height, clearance, wheel_mismatch, "
        "target_error, passive_velocity, passive_peak",
        flush=True,
    )
    for rank, row in enumerate(valid_rows[:20], start=1):
        print(
            f"{rank}, {row['action_LL_RL']:+.3f}, "
            f"{row['action_Lcrank_Rcrank']:+.3f}, "
            f"{row['wheel_support_height']:.5f}, "
            f"{row['nonwheel_clearance']:+.5f}, "
            f"{row['wheel_height_mismatch']:.5f}, "
            f"{row['active_target_error']:.5f}, "
            f"{row['final_passive_velocity']:.5f}, "
            f"{row['peak_passive_velocity']:.5f}",
            flush=True,
        )
    print(f"valid_candidates: {len(valid_rows)}/{len(rows)}", flush=True)
    print(f"csv: {output_path}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
