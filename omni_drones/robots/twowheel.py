import os.path as osp
import logging
from dataclasses import dataclass
from typing import Sequence

import omni.isaac.core.utils.prims as prim_utils
import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots.config import RobotCfg
from omni_drones.robots.robot import ASSET_PATH, RobotBase, TEMPLATE_PRIM_PATH
from omni.isaac.core.simulation_context import SimulationContext

from omni_drones.views import ArticulationView
from omni_drones.utils.torch import quat_axis, quat_rotate_inverse, quaternion_to_euler


logger = logging.getLogger(__name__)


@dataclass
class TwoWheelRobotCfg(RobotCfg):
    usd_path: str = osp.join(ASSET_PATH, "twowheel_uz05/usd/twowheel_uz05.all_joints.usd")
    left_wheel_joint: str = "Lwhl"
    right_wheel_joint: str = "Rwhl"
    leg_joint_names: Sequence[str] = ("L12", "L45", "R12", "R45")
    max_wheel_velocity: float = 25.0
    action_smoothing: float = 1.0
    action_rate_limit: float = 0.0
    action_accel_limit: float = 0.0
    wheel_action_decimation: int = 1
    wheel_action_deadband: float = 0.0
    wheel_action_hold_threshold: float = 0.0
    wheel_velocity_sign: float = 1.0
    leg_position_scale: float = 0.25
    max_leg_velocity: float = 1.5
    leg_action_smoothing: float = 0.25
    leg_action_rate_limit: float = 0.10
    lock_leg_actions: bool = False
    wheel_drive_stiffness: float = -1.0
    wheel_drive_damping: float = -1.0
    wheel_drive_max_force: float = -1.0
    wheel_drive_max_velocity: float = -1.0
    leg_drive_stiffness: float = -1.0
    leg_drive_damping: float = -1.0
    leg_drive_max_force: float = -1.0
    leg_drive_max_velocity: float = -1.0
    lock_passive_joints: bool = False
    passive_drive_stiffness: float = -1.0
    passive_drive_damping: float = -1.0
    passive_drive_max_force: float = -1.0
    passive_drive_max_velocity: float = -1.0


class TwoWheelRobot(RobotBase):
    cfg_cls = TwoWheelRobotCfg

    def __init__(self, name: str = None, cfg: TwoWheelRobotCfg = None):
        super().__init__(name or "TwoWheelRobot", cfg, is_articulation=True)
        self.usd_path = self.cfg.usd_path
        self.left_wheel_joint = self.cfg.left_wheel_joint
        self.right_wheel_joint = self.cfg.right_wheel_joint
        self.leg_joint_names = tuple(self.cfg.leg_joint_names)
        self.max_wheel_velocity = float(self.cfg.max_wheel_velocity)
        self.action_smoothing = float(self.cfg.action_smoothing)
        self.action_rate_limit = float(self.cfg.action_rate_limit)
        self.action_accel_limit = float(self.cfg.action_accel_limit)
        self.wheel_action_decimation = max(1, int(self.cfg.wheel_action_decimation))
        self.wheel_action_deadband = float(self.cfg.wheel_action_deadband)
        self.wheel_action_hold_threshold = float(self.cfg.wheel_action_hold_threshold)
        self.wheel_velocity_sign = float(self.cfg.wheel_velocity_sign)
        self.leg_position_scale = float(self.cfg.leg_position_scale)
        self.max_leg_velocity = float(self.cfg.max_leg_velocity)
        self.leg_action_smoothing = float(self.cfg.leg_action_smoothing)
        self.leg_action_rate_limit = float(self.cfg.leg_action_rate_limit)
        self.lock_leg_actions = bool(self.cfg.lock_leg_actions)
        self.wheel_drive_stiffness = float(self.cfg.wheel_drive_stiffness)
        self.wheel_drive_damping = float(self.cfg.wheel_drive_damping)
        self.wheel_drive_max_force = float(self.cfg.wheel_drive_max_force)
        self.wheel_drive_max_velocity = float(self.cfg.wheel_drive_max_velocity)
        self.leg_drive_stiffness = float(self.cfg.leg_drive_stiffness)
        self.leg_drive_damping = float(self.cfg.leg_drive_damping)
        self.leg_drive_max_force = float(self.cfg.leg_drive_max_force)
        self.leg_drive_max_velocity = float(self.cfg.leg_drive_max_velocity)
        self.lock_passive_joints = bool(self.cfg.lock_passive_joints)
        self.passive_drive_stiffness = float(self.cfg.passive_drive_stiffness)
        self.passive_drive_damping = float(self.cfg.passive_drive_damping)
        self.passive_drive_max_force = float(self.cfg.passive_drive_max_force)
        self.passive_drive_max_velocity = float(self.cfg.passive_drive_max_velocity)
        self._action_step = 0

        self.num_leg_joints = len(self.leg_joint_names)
        self.state_spec = UnboundedContinuousTensorSpec(
            3 + 4 + 6 + 2 + 2 * self.num_leg_joints + 2 + 2 * self.num_leg_joints,
            device=self.device,
        )
        self._action_spec = BoundedTensorSpec(
            -1.0,
            1.0,
            2 + 2 * self.num_leg_joints,
            device=self.device,
        )

    @property
    def action_spec(self):
        return self._action_spec

    def _create_prim(self, prim_path, translation, orientation):
        if not osp.exists(self.usd_path):
            raise FileNotFoundError(
                f"TwoWheelRobot USD not found: {self.usd_path}\n"
                "先运行 tools/convert_urdf_to_usd.py 把 twowheel_uz05.urdf 转成 USD。"
            )
        return super()._create_prim(prim_path, translation, orientation)

    def initialize(self, prim_paths_expr: str = None):
        if SimulationContext.instance()._physics_sim_view is None:
            raise RuntimeError(
                f"Cannot initialize {self.__class__.__name__} before the simulation context resets."
                "Call simulation_context.reset() first."
            )
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/{self.name}_.*/base_link"
        self.prim_paths_expr = prim_paths_expr
        self._view = ArticulationView(
            self.prim_paths_expr,
            reset_xform_properties=False,
            shape=(-1, self.n),
        )
        self.articulation = self
        self._view.initialize()
        self.shape = torch.arange(self._view.count).reshape(-1, self.n).shape
        self.prim_paths = self._view.prim_paths
        self.initialized = True

        dof_indices = self._view._dof_indices
        requested_dofs = (self.left_wheel_joint, self.right_wheel_joint, *self.leg_joint_names)
        missing = [name for name in requested_dofs if name not in dof_indices]
        if missing:
            raise RuntimeError(
                f"Missing two-wheel robot joints {missing}. Available DOFs: {self._view._dof_names}"
            )

        self.wheel_joint_indices = torch.tensor(
            [dof_indices[self.left_wheel_joint], dof_indices[self.right_wheel_joint]],
            dtype=torch.long,
            device=self.device,
        )
        self.leg_joint_indices = torch.tensor(
            [dof_indices[name] for name in self.leg_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.num_dofs = len(self._view._dof_names)
        controlled_joint_indices = torch.cat([self.wheel_joint_indices, self.leg_joint_indices]).tolist()
        passive_joint_indices = [
            i for i in range(self.num_dofs)
            if i not in controlled_joint_indices
        ]
        self.passive_joint_indices = torch.tensor(
            passive_joint_indices,
            dtype=torch.long,
            device=self.device,
        )
        self._apply_drive_overrides()

        self.pos, self.rot = self.get_world_poses(True)
        self.vel_w = torch.zeros(*self.shape, 6, device=self.device)
        self.vel_b = torch.zeros_like(self.vel_w)
        self.rpy = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.joint_pos = torch.zeros(*self.shape, self.num_dofs, device=self.device)
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.passive_joint_targets = torch.zeros(*self.shape, len(passive_joint_indices), device=self.device)
        self.wheel_vel = torch.zeros(*self.shape, 2, device=self.device)
        self.leg_pos = torch.zeros(*self.shape, self.num_leg_joints, device=self.device)
        self.leg_vel = torch.zeros_like(self.leg_pos)
        self.leg_neutral_pos = torch.zeros_like(self.leg_pos)
        self.leg_position_targets = torch.zeros_like(self.leg_pos)
        self.leg_velocity_targets = torch.zeros_like(self.leg_pos)
        self.prev_action = torch.zeros(*self.shape, self.action_spec.shape[-1], device=self.device)
        self.last_action = torch.zeros(*self.shape, self.action_spec.shape[-1], device=self.device)
        self.smoothed_action = torch.zeros(*self.shape, self.action_spec.shape[-1], device=self.device)
        self.action_difference = torch.zeros(*self.shape, device=self.device)
        self.action_acceleration = torch.zeros(*self.shape, device=self.device)
        self.action_magnitude = torch.zeros(*self.shape, device=self.device)
        self.wheel_action_difference = torch.zeros(*self.shape, device=self.device)
        self.wheel_action_acceleration = torch.zeros(*self.shape, device=self.device)
        self.wheel_action_magnitude = torch.zeros(*self.shape, device=self.device)
        self.leg_action_difference = torch.zeros(*self.shape, device=self.device)
        self.leg_action_magnitude = torch.zeros(*self.shape, device=self.device)

    def set_leg_neutral_positions(self, positions: torch.Tensor, env_ids: torch.Tensor = None):
        if self.num_leg_joints == 0:
            return
        if env_ids is None:
            self.leg_neutral_pos[:] = positions
            self.leg_position_targets[:] = positions
        else:
            self.leg_neutral_pos[env_ids] = positions
            self.leg_position_targets[env_ids] = positions

    def spawn(
        self,
        translations=[(0.0, 0.0, 0.35)],
        orientations=None,
        prim_paths: Sequence[str] = None,
    ):
        translations = torch.atleast_2d(torch.as_tensor(translations, device=self.device))
        n = translations.shape[0]
        if orientations is None:
            orientations = [None for _ in range(n)]
        if prim_paths is None:
            prim_paths = [f"{TEMPLATE_PRIM_PATH}/{self.name}_{i}" for i in range(n)]
        prims = []
        for prim_path, translation, orientation in zip(prim_paths, translations, orientations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim = self._create_prim(prim_path, translation, orientation)
            prims.append(prim)
        self.n += n
        return prims

    def _apply_drive_overrides(self):
        physics_view = getattr(self._view, "_physics_view", None)
        if physics_view is None:
            logger.warning("Cannot override drive gains before the physics view is initialized.")
            return
        env_indices = torch.arange(self._view.count, dtype=torch.long)

        def _set_dof_values(dof_indices, getter_name, setter_name, value, label):
            if value < 0.0:
                return
            getter = getattr(physics_view, getter_name, None)
            setter = getattr(physics_view, setter_name, None)
            if getter is None or setter is None:
                logger.warning(
                    "Physics view does not expose %s/%s; skipping %s override.",
                    getter_name,
                    setter_name,
                    label,
                )
                return
            values = getter()
            old = values[:, dof_indices].detach().clone()
            values[:, dof_indices] = values.new_tensor(value)
            try:
                setter(values, env_indices)
            except TypeError:
                setter(values)
            logger.info(
                "%s override: %s -> %.6g",
                label,
                old[0].detach().cpu().tolist() if old.numel() else [],
                value,
            )

        def _apply_group(prefix, dof_indices, stiffness, damping, max_force, max_velocity):
            if (
                stiffness < 0.0
                and damping < 0.0
                and max_force < 0.0
                and max_velocity < 0.0
            ):
                return
            dof_indices = dof_indices.to("cpu")
            _set_dof_values(dof_indices, "get_dof_stiffnesses", "set_dof_stiffnesses", stiffness, f"{prefix} stiffness")
            _set_dof_values(dof_indices, "get_dof_dampings", "set_dof_dampings", damping, f"{prefix} damping")
            _set_dof_values(dof_indices, "get_dof_max_forces", "set_dof_max_forces", max_force, f"{prefix} max force")
            _set_dof_values(dof_indices, "get_dof_max_velocities", "set_dof_max_velocities", max_velocity, f"{prefix} max velocity")

        _apply_group(
            "wheel",
            self.wheel_joint_indices,
            self.wheel_drive_stiffness,
            self.wheel_drive_damping,
            self.wheel_drive_max_force,
            self.wheel_drive_max_velocity,
        )
        if self.num_leg_joints:
            _apply_group(
                "leg",
                self.leg_joint_indices,
                self.leg_drive_stiffness,
                self.leg_drive_damping,
                self.leg_drive_max_force,
                self.leg_drive_max_velocity,
            )
        if self.passive_joint_indices.numel():
            _apply_group(
                "passive",
                self.passive_joint_indices,
                self.passive_drive_stiffness,
                self.passive_drive_damping,
                self.passive_drive_max_force,
                self.passive_drive_max_velocity,
            )

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clamp(-1.0, 1.0).expand(*self.shape, self.action_spec.shape[-1])
        wheel_actions = actions[..., :2]
        leg_pos_actions = actions[..., 2:2 + self.num_leg_joints]
        leg_vel_actions = actions[..., 2 + self.num_leg_joints:]

        smoothed_wheel_actions = self.smoothed_action[..., :2]
        update_wheel_action = self._action_step % self.wheel_action_decimation == 0
        if update_wheel_action:
            if self.action_smoothing < 1.0:
                wheel_actions = smoothed_wheel_actions + self.action_smoothing * (
                    wheel_actions - smoothed_wheel_actions
                )
            if self.action_rate_limit > 0.0:
                delta = (wheel_actions - smoothed_wheel_actions).clamp(
                    -self.action_rate_limit,
                    self.action_rate_limit,
                )
                wheel_actions = smoothed_wheel_actions + delta
            if self.action_accel_limit > 0.0:
                prev_delta = self.last_action[..., :2] - self.prev_action[..., :2]
                next_delta = wheel_actions - self.last_action[..., :2]
                delta_change = (next_delta - prev_delta).clamp(
                    -self.action_accel_limit,
                    self.action_accel_limit,
                )
                wheel_actions = (self.last_action[..., :2] + prev_delta + delta_change).clamp(-1.0, 1.0)
            if self.wheel_action_deadband > 0.0:
                wheel_actions = torch.where(
                    wheel_actions.abs() < self.wheel_action_deadband,
                    torch.zeros_like(wheel_actions),
                    wheel_actions,
                )
            if self.wheel_action_hold_threshold > 0.0:
                last_wheel_actions = self.last_action[..., :2]
                wheel_actions = torch.where(
                    (wheel_actions - last_wheel_actions).abs() < self.wheel_action_hold_threshold,
                    last_wheel_actions,
                    wheel_actions,
                )
        else:
            wheel_actions = smoothed_wheel_actions

        if self.num_leg_joints:
            if self.lock_leg_actions:
                leg_pos_actions = torch.zeros_like(leg_pos_actions)
                leg_vel_actions = torch.zeros_like(leg_vel_actions)
            smoothed_leg_pos_actions = self.smoothed_action[..., 2:2 + self.num_leg_joints]
            smoothed_leg_vel_actions = self.smoothed_action[..., 2 + self.num_leg_joints:]
            if self.leg_action_smoothing < 1.0:
                leg_pos_actions = smoothed_leg_pos_actions + self.leg_action_smoothing * (
                    leg_pos_actions - smoothed_leg_pos_actions
                )
                leg_vel_actions = smoothed_leg_vel_actions + self.leg_action_smoothing * (
                    leg_vel_actions - smoothed_leg_vel_actions
                )
            if self.leg_action_rate_limit > 0.0:
                leg_delta = (leg_pos_actions - smoothed_leg_pos_actions).clamp(
                    -self.leg_action_rate_limit,
                    self.leg_action_rate_limit,
                )
                leg_pos_actions = smoothed_leg_pos_actions + leg_delta
                leg_vel_delta = (leg_vel_actions - smoothed_leg_vel_actions).clamp(
                    -self.leg_action_rate_limit,
                    self.leg_action_rate_limit,
                )
                leg_vel_actions = smoothed_leg_vel_actions + leg_vel_delta
            self.leg_position_targets[:] = (
                self.leg_neutral_pos + leg_pos_actions * self.leg_position_scale
            )
            self.leg_velocity_targets[:] = leg_vel_actions * self.max_leg_velocity

        filtered_actions = torch.cat([wheel_actions, leg_pos_actions, leg_vel_actions], dim=-1)
        self.smoothed_action[:] = filtered_actions
        target_vel = wheel_actions * self.max_wheel_velocity * self.wheel_velocity_sign
        self._view.set_joint_velocity_targets(
            target_vel,
            joint_indices=self.wheel_joint_indices,
        )
        if self.num_leg_joints:
            self._view.set_joint_position_targets(
                self.leg_position_targets,
                joint_indices=self.leg_joint_indices,
            )
            self._view.set_joint_velocity_targets(
                self.leg_velocity_targets,
                joint_indices=self.leg_joint_indices,
            )
        if self.lock_passive_joints and len(self.passive_joint_indices):
            self._view.set_joint_position_targets(
                self.passive_joint_targets,
                joint_indices=self.passive_joint_indices,
            )
        self.action_difference[:] = torch.norm(filtered_actions - self.last_action, dim=-1)
        self.action_acceleration[:] = torch.norm(filtered_actions + self.prev_action - 2.0 * self.last_action, dim=-1)
        self.action_magnitude[:] = torch.norm(filtered_actions, dim=-1)
        self.wheel_action_difference[:] = torch.norm(
            filtered_actions[..., :2] - self.last_action[..., :2],
            dim=-1,
        )
        self.wheel_action_acceleration[:] = torch.norm(
            filtered_actions[..., :2] + self.prev_action[..., :2] - 2.0 * self.last_action[..., :2],
            dim=-1,
        )
        self.wheel_action_magnitude[:] = torch.norm(filtered_actions[..., :2], dim=-1)
        if self.num_leg_joints:
            self.leg_action_difference[:] = torch.norm(
                filtered_actions[..., 2:] - self.last_action[..., 2:],
                dim=-1,
            )
            self.leg_action_magnitude[:] = torch.norm(filtered_actions[..., 2:], dim=-1)
        else:
            self.leg_action_difference[:] = 0.0
            self.leg_action_magnitude[:] = 0.0
        self.prev_action[:] = self.last_action
        self.last_action[:] = filtered_actions
        self._action_step += 1
        return torch.norm(filtered_actions, dim=-1)

    def get_state(self, env_frame: bool = True):
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        if env_frame and hasattr(self, "_envs_positions"):
            self.pos.sub_(self._envs_positions)

        self.vel_w[:] = self.get_velocities(True)
        self.vel_b[:] = torch.cat(
            [
                quat_rotate_inverse(self.rot, self.vel_w[..., :3]),
                quat_rotate_inverse(self.rot, self.vel_w[..., 3:]),
            ],
            dim=-1,
        )
        self.rpy[:] = quaternion_to_euler(self.rot)
        self.up[:] = quat_axis(self.rot, axis=2)
        self.heading[:] = quat_axis(self.rot, axis=0)
        self.joint_pos[:] = self.get_joint_positions(True)
        self.joint_vel[:] = self.get_joint_velocities(True)
        self.wheel_vel[:] = self.joint_vel[..., self.wheel_joint_indices]
        if self.num_leg_joints:
            self.leg_pos[:] = self.joint_pos[..., self.leg_joint_indices]
            self.leg_vel[:] = self.joint_vel[..., self.leg_joint_indices]

        return torch.cat(
            [
                self.pos,
                self.rot,
                self.vel_b,
                self.wheel_vel,
                self.leg_pos,
                self.leg_vel,
                self.last_action,
            ],
            dim=-1,
        )

    def _reset_idx(self, env_ids: torch.Tensor, train: bool = True):
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        self.prev_action[env_ids] = 0.0
        self.last_action[env_ids] = 0.0
        self.smoothed_action[env_ids] = 0.0
        self.action_difference[env_ids] = 0.0
        self.action_acceleration[env_ids] = 0.0
        self.action_magnitude[env_ids] = 0.0
        self.wheel_action_difference[env_ids] = 0.0
        self.wheel_action_acceleration[env_ids] = 0.0
        self.wheel_action_magnitude[env_ids] = 0.0
        self.leg_action_difference[env_ids] = 0.0
        self.leg_action_magnitude[env_ids] = 0.0
        return env_ids
