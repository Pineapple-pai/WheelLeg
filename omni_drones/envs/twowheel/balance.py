import torch
import torch.distributions as D

import omni_drones.utils.kit as kit_utils
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.twowheel import TwoWheelRobot, TwoWheelRobotCfg
from omni_drones.utils.torch import euler_to_quaternion


class TwoWheelBalance(IsaacEnv):
    def __init__(self, cfg, headless):
        self.time_encoding = cfg.task.time_encoding
        self.time_encoding_dim = 4
        self.alpha = cfg.task.stats_alpha

        self.reward_alive_weight = cfg.task.reward_alive_weight
        self.reward_stand_weight = cfg.task.reward_stand_weight
        self.reward_upright_weight = cfg.task.reward_upright_weight
        self.reward_height_weight = cfg.task.reward_height_weight
        self.reward_velocity_weight = cfg.task.reward_velocity_weight
        self.reward_direction_weight = cfg.task.reward_direction_weight
        self.reward_forward_progress_weight = cfg.task.reward_forward_progress_weight
        self.reward_displacement_progress_weight = cfg.task.reward_displacement_progress_weight
        self.penalty_forward_deficit_weight = cfg.task.penalty_forward_deficit_weight
        self.penalty_speed_shortfall_weight = cfg.task.penalty_speed_shortfall_weight
        self.reward_position_weight = cfg.task.reward_position_weight
        self.reward_yaw_weight = cfg.task.reward_yaw_weight
        self.reward_yaw_rate_weight = cfg.task.reward_yaw_rate_weight
        self.reward_settle_weight = cfg.task.reward_settle_weight
        self.penalty_roll_weight = cfg.task.penalty_roll_weight
        self.penalty_roll_rate_weight = cfg.task.penalty_roll_rate_weight
        self.penalty_pitch_weight = cfg.task.penalty_pitch_weight
        self.penalty_pitch_rate_weight = cfg.task.penalty_pitch_rate_weight
        self.penalty_yaw_weight = cfg.task.penalty_yaw_weight
        self.penalty_yaw_rate_weight = cfg.task.penalty_yaw_rate_weight
        self.penalty_position_weight = cfg.task.penalty_position_weight
        self.penalty_lateral_velocity_weight = cfg.task.penalty_lateral_velocity_weight
        self.penalty_lin_vel_z_weight = cfg.task.penalty_lin_vel_z_weight
        self.penalty_action_diff_weight = cfg.task.penalty_action_diff_weight
        self.penalty_wheel_vel_diff_weight = cfg.task.penalty_wheel_vel_diff_weight
        self.penalty_wheel_speed_weight = cfg.task.penalty_wheel_speed_weight
        self.penalty_wheel_delta_weight = float(cfg.task.get("penalty_wheel_delta_weight", 0.0))
        self.penalty_wheel_common_delta_weight = float(cfg.task.get("penalty_wheel_common_delta_weight", 0.0))
        self.penalty_wheel_diff_delta_weight = float(cfg.task.get("penalty_wheel_diff_delta_weight", 0.0))
        self.penalty_wheel_sign_flip_weight = float(cfg.task.get("penalty_wheel_sign_flip_weight", 0.0))
        self.penalty_wheel_action_diff_weight = float(cfg.task.get("penalty_wheel_action_diff_weight", 0.0))
        self.penalty_wheel_action_accel_weight = float(cfg.task.get("penalty_wheel_action_accel_weight", 0.0))
        self.penalty_wheel_action_magnitude_weight = float(cfg.task.get("penalty_wheel_action_magnitude_weight", 0.0))
        self.penalty_quiet_wheel_weight = float(cfg.task.get("penalty_quiet_wheel_weight", 0.0))
        self.penalty_action_weight = cfg.task.penalty_action_weight
        self.penalty_smoothness_weight = cfg.task.penalty_smoothness_weight
        self.penalty_action_accel_weight = cfg.task.penalty_action_accel_weight
        self.reward_leg_neutral_weight = cfg.task.reward_leg_neutral_weight
        self.penalty_leg_pos_weight = cfg.task.penalty_leg_pos_weight
        self.penalty_leg_vel_weight = cfg.task.penalty_leg_vel_weight
        self.penalty_leg_action_weight = cfg.task.penalty_leg_action_weight
        self.penalty_leg_action_diff_weight = cfg.task.penalty_leg_action_diff_weight
        self.penalty_fall_weight = cfg.task.penalty_fall_weight
        self.wl_tracking_sigma = cfg.task.wl_tracking_sigma
        self.wl_tracking_ang_sigma = cfg.task.wl_tracking_ang_sigma
        self.wl_height_sigma = cfg.task.wl_height_sigma
        self.wl_tracking_lin_vel_weight = cfg.task.wl_tracking_lin_vel_weight
        self.wl_tracking_lin_vel_enhance_weight = cfg.task.wl_tracking_lin_vel_enhance_weight
        self.wl_tracking_ang_vel_weight = cfg.task.wl_tracking_ang_vel_weight
        self.wl_base_height_weight = cfg.task.wl_base_height_weight
        self.wl_orientation_weight = cfg.task.wl_orientation_weight
        self.wl_lin_vel_z_weight = cfg.task.wl_lin_vel_z_weight
        self.wl_ang_vel_xy_weight = cfg.task.wl_ang_vel_xy_weight
        self.wl_dof_vel_weight = cfg.task.wl_dof_vel_weight
        self.wl_dof_acc_weight = cfg.task.wl_dof_acc_weight
        self.wl_action_rate_weight = cfg.task.wl_action_rate_weight
        self.wl_action_smooth_weight = cfg.task.wl_action_smooth_weight
        self.wl_nominal_state_weight = cfg.task.wl_nominal_state_weight
        self.reward_tracking_lin_vel_weight = float(cfg.task.get("reward_tracking_lin_vel_weight", 1.0))
        self.reward_tracking_lin_vel_enhance_weight = float(cfg.task.get("reward_tracking_lin_vel_enhance_weight", 1.0))
        self.reward_tracking_ang_vel_weight = float(cfg.task.get("reward_tracking_ang_vel_weight", 3.0))
        self.reward_tracking_ang_vel_enhance_weight = float(cfg.task.get("reward_tracking_ang_vel_enhance_weight", 3.0))
        self.reward_base_height_weight = float(cfg.task.get("reward_base_height_weight", 1.0))
        self.reward_nominal_state_weight = float(cfg.task.get("reward_nominal_state_weight", -1.0))
        self.reward_lin_vel_z_weight = float(cfg.task.get("reward_lin_vel_z_weight", -1.0))
        self.reward_ang_vel_xy_weight = float(cfg.task.get("reward_ang_vel_xy_weight", -0.2))
        self.reward_orientation_weight = float(cfg.task.get("reward_orientation_weight", -30.0))
        self.reward_dof_vel_weight = float(cfg.task.get("reward_dof_vel_weight", -5e-5))
        self.reward_dof_acc_weight = float(cfg.task.get("reward_dof_acc_weight", -2.5e-7))
        self.reward_torques_weight = float(cfg.task.get("reward_torques_weight", -1e-4))
        self.reward_action_rate_weight = float(cfg.task.get("reward_action_rate_weight", -0.05))
        self.reward_action_smooth_weight = float(cfg.task.get("reward_action_smooth_weight", -0.05))
        self.reward_collision_weight = float(cfg.task.get("reward_collision_weight", -1.0))
        self.reward_dof_pos_limits_weight = float(cfg.task.get("reward_dof_pos_limits_weight", -1.0))
        self.tracking_sigma = float(cfg.task.get("tracking_sigma", 0.25))
        self.base_height_target = float(cfg.task.get("base_height_target", 0.18))
        self.height_floor = float(cfg.task.get("height_floor", self.base_height_target - 0.01))
        self.penalty_low_height_weight = float(cfg.task.get("penalty_low_height_weight", 0.0))
        self.soft_dof_pos_limit = float(cfg.task.get("soft_dof_pos_limit", 0.97))
        self.max_contact_force = float(cfg.task.get("max_contact_force", 100.0))
        self.terminate_on_body_contact = bool(cfg.task.get("terminate_on_body_contact", True))
        self.contact_terminate_force_threshold = float(
            cfg.task.get("contact_terminate_force_threshold", self.max_contact_force)
        )
        self.upright_roll_rate_scale = float(cfg.task.get("upright_roll_rate_scale", 1.5))
        self.upright_pitch_rate_scale = float(cfg.task.get("upright_pitch_rate_scale", 0.5))
        self.settle_roll_rate_scale = float(cfg.task.get("settle_roll_rate_scale", 1.2))
        self.settle_pitch_rate_scale = float(cfg.task.get("settle_pitch_rate_scale", 0.8))
        self.settle_wheel_delta_scale = float(cfg.task.get("settle_wheel_delta_scale", 0.75))
        self.settle_wheel_common_delta_scale = float(cfg.task.get("settle_wheel_common_delta_scale", 0.65))
        self.settle_wheel_diff_delta_scale = float(cfg.task.get("settle_wheel_diff_delta_scale", 0.30))

        self.max_init_roll = cfg.task.max_init_roll
        self.max_init_pitch = cfg.task.max_init_pitch
        self.min_command_vx = cfg.task.min_command_vx
        self.max_command_vx = cfg.task.max_command_vx
        self.forward_axis_sign = cfg.task.forward_axis_sign
        self.command_vx_sign = cfg.task.command_vx_sign
        self.command_bidirectional = cfg.task.command_bidirectional
        self.command_zero_prob = float(cfg.task.get("command_zero_prob", 0.0))
        self.command_active_vx = float(cfg.task.get("command_active_vx", 0.015))
        self.command_pitch_gain = cfg.task.command_pitch_gain
        self.command_pitch_limit = cfg.task.command_pitch_limit
        self.command_pitch_sign = cfg.task.command_pitch_sign
        self.wheel_command_bias = cfg.task.wheel_command_bias
        self.wheel_command_bias_sign = cfg.task.wheel_command_bias_sign
        self.max_command_yaw_rate = cfg.task.max_command_yaw_rate
        self.eval_command_vx = cfg.task.eval_command_vx
        self.eval_command_yaw_rate = cfg.task.eval_command_yaw_rate
        self.reset_height = cfg.task.reset_height
        self.progress_dt = cfg.task.sim.dt
        self.min_height = cfg.task.min_height
        self.termination_roll = cfg.task.termination_roll
        self.termination_pitch = cfg.task.termination_pitch
        self.max_xy = cfg.task.max_xy
        stand_joint_pos = cfg.task.get("stand_joint_pos", [0.0, 0.0, 0.0, 0.0])
        self.stand_joint_pos = torch.as_tensor(stand_joint_pos, device=self.device).float().reshape(1, -1)
        self.push_interval = float(cfg.task.get("push_interval", 0.0))
        self.kick_velocity_min = float(cfg.task.get("kick_velocity_min", cfg.task.get("push_force_min", 0.0)))
        self.kick_velocity_max = float(cfg.task.get("kick_velocity_max", cfg.task.get("push_force_max", 0.0)))
        self.kick_pitch_rate_min = float(cfg.task.get("kick_pitch_rate_min", 0.0))
        self.kick_pitch_rate_max = float(cfg.task.get("kick_pitch_rate_max", 0.0))
        self.push_warmup_steps = int(cfg.task.get("push_warmup_steps", 0))
        self.drop_reset_prob = float(cfg.task.get("drop_reset_prob", 0.0))
        self.drop_height_min = float(cfg.task.get("drop_height_min", self.reset_height))
        self.drop_height_max = float(cfg.task.get("drop_height_max", self.reset_height))
        self.reset_lin_vel_x = float(cfg.task.get("reset_lin_vel_x", 0.0))
        self.reset_pitch_rate = float(cfg.task.get("reset_pitch_rate", 0.0))

        super().__init__(cfg, headless)

        self.robot.initialize()
        self.dof_limits = self.robot._view.get_dof_limits().to(self.device)
        body_indices = getattr(self.robot._view, "_body_indices", {})
        self.collision_body_indices = torch.tensor(
            [idx for name, idx in body_indices.items() if name not in {"Lwhl", "Rwhl"}],
            device=self.device,
            dtype=torch.long,
        )
        init_pos, init_rot = self.robot.get_world_poses(clone=True)
        self.init_poses = (init_pos.to(self.device), init_rot.to(self.device))
        self.init_vels = torch.zeros_like(self.robot.get_velocities()).to(self.device)
        self.init_joint_pos = self.robot.get_joint_positions(clone=True).to(self.device)
        self.init_joint_vel = torch.zeros_like(self.robot.get_joint_velocities()).to(self.device)

        self.command = torch.zeros(self.num_envs, 1, 2, device=self.device)
        self.prev_x = torch.zeros(self.num_envs, 1, device=self.device)
        self.prev_leg_vel = torch.zeros_like(self.robot.leg_vel)
        self.prev_wheel_vel = torch.zeros_like(self.robot.wheel_vel)
        self.push_prob = 0.0
        if self.push_interval > 0.0 and self.kick_velocity_max > 0.0:
            self.push_prob = float(1.0 - torch.exp(
                torch.tensor(-self.progress_dt / self.push_interval)
            ).item())
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-self.max_init_roll, -self.max_init_pitch, 0.0], device=self.device),
            torch.tensor([self.max_init_roll, self.max_init_pitch, 0.0], device=self.device),
        )
        self.yaw_command_dist = D.Uniform(
            torch.tensor([-self.max_command_yaw_rate], device=self.device),
            torch.tensor([self.max_command_yaw_rate], device=self.device),
        )

    def _get_contact_penalty(self, threshold: float = None):
        if self.collision_body_indices.numel() == 0:
            return torch.zeros(self.num_envs, 1, device=self.device)
        get_contacts = getattr(self.robot._view, "get_net_contact_forces", None)
        if get_contacts is None:
            return torch.zeros(self.num_envs, 1, device=self.device)
        try:
            contact_forces = get_contacts(clone=True)
        except TypeError:
            contact_forces = get_contacts()
        except Exception:
            return torch.zeros(self.num_envs, 1, device=self.device)
        if contact_forces is None or contact_forces.numel() == 0:
            return torch.zeros(self.num_envs, 1, device=self.device)
        if threshold is None:
            threshold = self.max_contact_force
        contact_norm = contact_forces[..., self.collision_body_indices, :].norm(dim=-1)
        return (contact_norm > threshold).float().sum(dim=-1, keepdim=True)

    def _get_dof_pos_limit_penalty(self):
        if self.dof_limits is None or self.dof_limits.numel() == 0:
            return torch.zeros(self.num_envs, 1, device=self.device)
        limits = self.dof_limits
        if limits.dim() == 2:
            limits = limits.unsqueeze(0).expand(self.num_envs, -1, -1)
        pos = self.robot.joint_pos
        lower = limits[..., 0]
        upper = limits[..., 1]
        finite = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower)
        if not finite.any():
            return torch.zeros(self.num_envs, 1, device=self.device)
        center = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower) * self.soft_dof_pos_limit
        soft_lower = center - half_range
        soft_upper = center + half_range
        violation = torch.clamp(soft_lower - pos, min=0.0) + torch.clamp(pos - soft_upper, min=0.0)
        violation = violation * finite.float()
        return torch.sum(violation.square(), dim=-1, keepdim=True)

    def _design_scene(self):
        robot_cfg = self.cfg.task.robot
        default_robot_cfg = TwoWheelRobotCfg()
        self.robot = TwoWheelRobot(
            cfg=TwoWheelRobotCfg(
                usd_path=robot_cfg.get("usd_path", default_robot_cfg.usd_path),
                left_wheel_joint=robot_cfg.get("left_wheel_joint", default_robot_cfg.left_wheel_joint),
                right_wheel_joint=robot_cfg.get("right_wheel_joint", default_robot_cfg.right_wheel_joint),
                leg_joint_names=robot_cfg.get("leg_joint_names", default_robot_cfg.leg_joint_names),
                max_wheel_velocity=robot_cfg.get("max_wheel_velocity", default_robot_cfg.max_wheel_velocity),
                action_smoothing=robot_cfg.get("action_smoothing", default_robot_cfg.action_smoothing),
                action_rate_limit=robot_cfg.get("action_rate_limit", default_robot_cfg.action_rate_limit),
                action_accel_limit=robot_cfg.get("action_accel_limit", default_robot_cfg.action_accel_limit),
                wheel_action_decimation=robot_cfg.get("wheel_action_decimation", 1),
                wheel_action_deadband=robot_cfg.get("wheel_action_deadband", 0.0),
                wheel_action_hold_threshold=robot_cfg.get("wheel_action_hold_threshold", 0.0),
                wheel_velocity_sign=robot_cfg.get("wheel_velocity_sign", default_robot_cfg.wheel_velocity_sign),
                leg_position_scale=robot_cfg.get("leg_position_scale", default_robot_cfg.leg_position_scale),
                max_leg_velocity=robot_cfg.get("max_leg_velocity", default_robot_cfg.max_leg_velocity),
                leg_action_smoothing=robot_cfg.get("leg_action_smoothing", default_robot_cfg.leg_action_smoothing),
                leg_action_rate_limit=robot_cfg.get("leg_action_rate_limit", default_robot_cfg.leg_action_rate_limit),
                lock_leg_actions=robot_cfg.get("lock_leg_actions", default_robot_cfg.lock_leg_actions),
                wheel_drive_stiffness=robot_cfg.get("wheel_drive_stiffness", -1.0),
                wheel_drive_damping=robot_cfg.get("wheel_drive_damping", -1.0),
                wheel_drive_max_force=robot_cfg.get("wheel_drive_max_force", -1.0),
                wheel_drive_max_velocity=robot_cfg.get("wheel_drive_max_velocity", -1.0),
                leg_drive_stiffness=robot_cfg.get("leg_drive_stiffness", -1.0),
                leg_drive_damping=robot_cfg.get("leg_drive_damping", -1.0),
                leg_drive_max_force=robot_cfg.get("leg_drive_max_force", -1.0),
                leg_drive_max_velocity=robot_cfg.get("leg_drive_max_velocity", -1.0),
                lock_passive_joints=robot_cfg.get(
                    "lock_passive_joints",
                    default_robot_cfg.lock_passive_joints,
                ),
                passive_drive_stiffness=robot_cfg.get("passive_drive_stiffness", -1.0),
                passive_drive_damping=robot_cfg.get("passive_drive_damping", -1.0),
                passive_drive_max_force=robot_cfg.get("passive_drive_max_force", -1.0),
                passive_drive_max_velocity=robot_cfg.get("passive_drive_max_velocity", -1.0),
            )
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=self.cfg.task.ground.static_friction,
            dynamic_friction=self.cfg.task.ground.dynamic_friction,
            restitution=self.cfg.task.ground.restitution,
        )
        self.robot.spawn(translations=[(0.0, 0.0, self.reset_height)])
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        action_dim = self.robot.action_spec.shape[-1]
        leg_obs_dim = 2 * self.robot.num_leg_joints
        observation_dim = 3 + 1 + 6 + 2 + 2 + 2 + leg_obs_dim + action_dim
        if self.time_encoding:
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": UnboundedContinuousTensorSpec((1, 1), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.robot.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["robot"] = AgentSpec(
            "robot",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1, device=self.device),
            "episode_len": UnboundedContinuousTensorSpec(1, device=self.device),
            "uprightness": UnboundedContinuousTensorSpec(1, device=self.device),
            "roll_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "roll_rate_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "pitch_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "pitch_target": UnboundedContinuousTensorSpec(1, device=self.device),
            "pitch_tracking_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "pitch_rate_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "height": UnboundedContinuousTensorSpec(1, device=self.device),
            "height_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "terminated_roll": UnboundedContinuousTensorSpec(1, device=self.device),
            "terminated_pitch": UnboundedContinuousTensorSpec(1, device=self.device),
            "terminated_height": UnboundedContinuousTensorSpec(1, device=self.device),
            "terminated_xy": UnboundedContinuousTensorSpec(1, device=self.device),
            "fall_penalty": UnboundedContinuousTensorSpec(1, device=self.device),
            "stand_reward": UnboundedContinuousTensorSpec(1, device=self.device),
            "settle_reward": UnboundedContinuousTensorSpec(1, device=self.device),
            "velocity_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "forward_velocity": UnboundedContinuousTensorSpec(1, device=self.device),
            "velocity_alignment": UnboundedContinuousTensorSpec(1, device=self.device),
            "forward_progress": UnboundedContinuousTensorSpec(1, device=self.device),
            "displacement_progress": UnboundedContinuousTensorSpec(1, device=self.device),
            "forward_displacement": UnboundedContinuousTensorSpec(1, device=self.device),
            "forward_deficit": UnboundedContinuousTensorSpec(1, device=self.device),
            "speed_shortfall": UnboundedContinuousTensorSpec(1, device=self.device),
            "position_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "lateral_velocity_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "vertical_velocity_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "yaw_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "yaw_rate_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "action_diff": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_vel_diff": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_speed": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_delta": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_common_delta": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_diff_delta": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_sign_flip": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_action_diff": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_action_acceleration": UnboundedContinuousTensorSpec(1, device=self.device),
            "wheel_action_magnitude": UnboundedContinuousTensorSpec(1, device=self.device),
            "quiet_state": UnboundedContinuousTensorSpec(1, device=self.device),
            "quiet_wheel_penalty": UnboundedContinuousTensorSpec(1, device=self.device),
            "action_smoothness": UnboundedContinuousTensorSpec(1, device=self.device),
            "action_acceleration": UnboundedContinuousTensorSpec(1, device=self.device),
            "leg_pos_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "leg_vel_error": UnboundedContinuousTensorSpec(1, device=self.device),
            "leg_neutral_reward": UnboundedContinuousTensorSpec(1, device=self.device),
            "leg_action_magnitude": UnboundedContinuousTensorSpec(1, device=self.device),
            "leg_action_diff": UnboundedContinuousTensorSpec(1, device=self.device),
            "command_vx": UnboundedContinuousTensorSpec(1, device=self.device),
            "command_yaw_rate": UnboundedContinuousTensorSpec(1, device=self.device),
            "wl_tracking_lin_vel": UnboundedContinuousTensorSpec(1, device=self.device),
            "wl_tracking_ang_vel": UnboundedContinuousTensorSpec(1, device=self.device),
            "wl_base_height": UnboundedContinuousTensorSpec(1, device=self.device),
            "wl_orientation": UnboundedContinuousTensorSpec(1, device=self.device),
            "wl_action_rate": UnboundedContinuousTensorSpec(1, device=self.device),
            "wl_action_smooth": UnboundedContinuousTensorSpec(1, device=self.device),
            "wl_dof_vel": UnboundedContinuousTensorSpec(1, device=self.device),
            "wl_dof_acc": UnboundedContinuousTensorSpec(1, device=self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.robot._reset_idx(env_ids, self.training)

        if self.training and (self.max_init_roll > 0.0 or self.max_init_pitch > 0.0):
            rpy = self.init_rpy_dist.sample(env_ids.shape).unsqueeze(1)
        else:
            rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        rot = euler_to_quaternion(rpy)
        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[..., 2] = self.reset_height
        if self.training and self.drop_reset_prob > 0.0 and self.drop_height_max > self.reset_height:
            drop_mask = torch.rand(len(env_ids), 1, device=self.device) < self.drop_reset_prob
            drop_height = torch.empty(len(env_ids), 1, device=self.device).uniform_(
                max(self.drop_height_min, self.reset_height),
                self.drop_height_max,
            )
            pos[..., 2] = torch.where(drop_mask, drop_height, pos[..., 2])
        self.robot.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1),
            rot,
            env_ids,
        )
        init_vels = self.init_vels[env_ids].clone()
        if self.training:
            if self.reset_lin_vel_x > 0.0:
                init_vels[..., 0] = torch.empty(len(env_ids), 1, device=self.device).uniform_(
                    -self.reset_lin_vel_x,
                    self.reset_lin_vel_x,
                )
            if self.reset_pitch_rate > 0.0:
                init_vels[..., 4] = torch.empty(len(env_ids), 1, device=self.device).uniform_(
                    -self.reset_pitch_rate,
                    self.reset_pitch_rate,
                )
        self.robot.set_velocities(init_vels, env_ids)
        joint_pos = self.init_joint_pos[env_ids].clone()
        if self.robot.num_leg_joints:
            stand_joint_pos = self.stand_joint_pos.to(joint_pos.device).unsqueeze(1).expand(len(env_ids), 1, -1)
            joint_pos[..., self.robot.leg_joint_indices] = stand_joint_pos
        self.robot.set_joint_positions(joint_pos, env_ids)
        self.robot.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)
        self.robot._view.set_joint_velocity_targets(self.init_joint_vel[env_ids], env_indices=env_ids)
        if self.robot.num_leg_joints:
            leg_neutral_pos = self.stand_joint_pos.to(joint_pos.device).unsqueeze(1).expand(len(env_ids), 1, -1)
            leg_zero_vel = self.init_joint_vel[env_ids][..., self.robot.leg_joint_indices]
            self.robot.set_leg_neutral_positions(leg_neutral_pos, env_ids)
            self.robot._view.set_joint_position_targets(
                leg_neutral_pos,
                env_indices=env_ids,
                joint_indices=self.robot.leg_joint_indices,
            )
            self.robot._view.set_joint_velocity_targets(
                leg_zero_vel,
                env_indices=env_ids,
                joint_indices=self.robot.leg_joint_indices,
            )
        if self.robot.lock_passive_joints and len(self.robot.passive_joint_indices):
            self.robot.passive_joint_targets[env_ids] = self.init_joint_pos[
                env_ids
            ][..., self.robot.passive_joint_indices]
            self.robot._view.set_joint_position_targets(
                self.robot.passive_joint_targets[env_ids],
                env_indices=env_ids,
                joint_indices=self.robot.passive_joint_indices,
            )

        if self.training and (self.max_command_vx > 0.0 or self.max_command_yaw_rate > 0.0):
            command = torch.zeros(len(env_ids), 1, 2, device=self.device)
            if self.max_command_vx > 0.0:
                vx = torch.empty(len(env_ids), 1, device=self.device).uniform_(
                    self.min_command_vx,
                    self.max_command_vx,
                )
                if self.command_bidirectional:
                    signs = torch.where(
                        torch.rand(len(env_ids), 1, device=self.device) < 0.5,
                        -1.0,
                        1.0,
                    )
                    vx = vx * signs
                else:
                    vx = vx * self.command_vx_sign
                if self.command_zero_prob > 0.0:
                    moving = torch.rand(len(env_ids), 1, device=self.device) >= self.command_zero_prob
                    vx = torch.where(moving, vx, torch.zeros_like(vx))
                command[..., 0] = vx
            if self.max_command_yaw_rate > 0.0:
                command[..., 1] = self.yaw_command_dist.sample(env_ids.shape)
            self.command[env_ids] = command
        else:
            command = torch.zeros(len(env_ids), 1, 2, device=self.device)
            command[..., 0] = self.eval_command_vx
            command[..., 1] = self.eval_command_yaw_rate
            self.command[env_ids] = command

        self.prev_x[env_ids] = 0.0
        self.prev_wheel_vel[env_ids] = self.init_joint_vel[env_ids][..., self.robot.wheel_joint_indices]
        if self.robot.num_leg_joints:
            self.prev_leg_vel[env_ids] = self.init_joint_vel[env_ids][..., self.robot.leg_joint_indices]
        self.stats[env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")].clone()
        if self.max_command_vx > 0.0 and self.wheel_command_bias > 0.0:
            command_ratio = (
                self.command[..., 0] / max(self.max_command_vx, 1e-6)
            ).clamp(-1.0, 1.0)
            wheel_bias = (
                command_ratio
                * self.wheel_command_bias
                * self.wheel_command_bias_sign
            ).unsqueeze(-1)
            actions[..., :2] = (actions[..., :2] + wheel_bias).clamp(-1.0, 1.0)
        self.effort = self.robot.apply_action(actions)
        self._apply_disturbances()

    def _apply_disturbances(self):
        if not self.training or self.push_prob <= 0.0:
            return
        can_push = self.progress_buf > self.push_warmup_steps
        new_push = (torch.rand(self.num_envs, 1, device=self.device) < self.push_prob) & can_push.unsqueeze(-1)
        if new_push.any():
            kick_mag = torch.empty(self.num_envs, 1, device=self.device).uniform_(
                self.kick_velocity_min,
                self.kick_velocity_max,
            )
            kick_sign = torch.where(
                torch.rand(self.num_envs, 1, device=self.device) < 0.5,
                -1.0,
                1.0,
            )
            push_ids = new_push.squeeze(-1)
            env_ids = push_ids.nonzero(as_tuple=False).squeeze(-1)
            vels = self.robot.get_velocities(clone=True)
            vels[push_ids, 0, 0] += kick_mag[push_ids, 0] * kick_sign[push_ids, 0]
            if self.kick_pitch_rate_max > 0.0:
                pitch_mag = torch.empty(self.num_envs, 1, device=self.device).uniform_(
                    self.kick_pitch_rate_min,
                    self.kick_pitch_rate_max,
                )
                vels[push_ids, 0, 4] += -kick_mag.new_tensor(1.0) * (
                    pitch_mag[push_ids, 0] * kick_sign[push_ids, 0]
                )
            self.robot.set_velocities(
                vels[push_ids],
                env_ids,
            )

    def _compute_state_and_obs(self):
        self.robot_state = self.robot.get_state()
        rpy = self.robot.rpy
        pos_xy = self.robot.pos[..., :2].clamp(-1.0, 1.0)
        height = self.robot.pos[..., 2:3]
        vel = self.robot.vel_b
        wheel_vel = self.robot.wheel_vel / self.robot.max_wheel_velocity
        leg_pos = (self.robot.leg_pos - self.robot.leg_neutral_pos) / self.robot.leg_position_scale
        leg_vel = self.robot.leg_vel / self.robot.max_leg_velocity
        action = self.robot.last_action

        obs = [
            rpy,
            height,
            vel,
            wheel_vel,
            pos_xy,
            self.command,
            leg_pos.clamp(-2.0, 2.0),
            leg_vel.clamp(-2.0, 2.0),
            action,
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": torch.ones(self.num_envs, 1, 1, device=self.device),
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        def _as_column(x: torch.Tensor):
            return x.reshape(self.num_envs, -1)[..., :1]

        roll = self.robot.rpy[..., 0]
        pitch = self.robot.rpy[..., 1]
        yaw = self.robot.rpy[..., 2]
        height = self.robot.pos[..., 2]
        pos_xy = self.robot.pos[..., :2]
        vx = self.robot.vel_b[..., 0]
        vy = self.robot.vel_b[..., 1]
        vz = self.robot.vel_b[..., 2]
        roll_rate = self.robot.vel_b[..., 3]
        pitch_rate = self.robot.vel_b[..., 4]
        yaw_rate = self.robot.vel_b[..., 5]
        action_left = self.robot.last_action[..., 0]
        action_right = self.robot.last_action[..., 1]
        wheel_left = self.robot.wheel_vel[..., 0] / self.robot.max_wheel_velocity
        wheel_right = self.robot.wheel_vel[..., 1] / self.robot.max_wheel_velocity
        prev_wheel_left = self.prev_wheel_vel[..., 0] / self.robot.max_wheel_velocity
        prev_wheel_right = self.prev_wheel_vel[..., 1] / self.robot.max_wheel_velocity
        leg_pos_delta = (self.robot.leg_pos - self.robot.leg_neutral_pos) / self.robot.leg_position_scale
        leg_vel_scaled = self.robot.leg_vel / self.robot.max_leg_velocity

        roll_error = roll.abs()
        roll_rate_error = roll_rate.abs()
        pitch_target = (
            self.command[..., 0]
            * self.command_pitch_gain
            * self.command_pitch_sign
        ).clamp(-self.command_pitch_limit, self.command_pitch_limit)
        pitch_error = pitch.abs()
        pitch_tracking_error = (pitch - pitch_target).abs()
        pitch_rate_error = pitch_rate.abs()
        height_error = (height - self.base_height_target).abs()
        tilt_error = roll.square() + pitch.square()
        uprightness = (torch.cos(roll) * torch.cos(pitch)).clamp(-1.0, 1.0)
        forward_vx = self.forward_axis_sign * vx
        velocity_error = (forward_vx - self.command[..., 0]).abs()
        command_vx = self.command[..., 0]
        command_active = command_vx.abs() > self.command_active_vx
        forward_displacement = self.forward_axis_sign * self.robot.pos[..., 0]
        delta_x = forward_displacement - self.prev_x
        if self.max_command_vx > 0.0:
            position_error = pos_xy[..., 1].abs()
        else:
            position_error = pos_xy.norm(dim=-1)
        lateral_velocity_error = vy.abs()
        vertical_velocity_error = vz.abs()
        yaw_error = torch.atan2(torch.sin(yaw), torch.cos(yaw)).abs()
        yaw_rate_error = (yaw_rate - self.command[..., 1]).abs()
        action_diff = (action_left - action_right).abs()
        wheel_vel_diff = (wheel_left - wheel_right).abs()
        wheel_speed = torch.sqrt(wheel_left.square() + wheel_right.square())
        wheel_delta_left = wheel_left - prev_wheel_left
        wheel_delta_right = wheel_right - prev_wheel_right
        wheel_delta = torch.sqrt(wheel_delta_left.square() + wheel_delta_right.square())
        wheel_common_delta = (
            0.5 * (wheel_left + wheel_right)
            - 0.5 * (prev_wheel_left + prev_wheel_right)
        ).abs()
        wheel_diff_delta = (
            (wheel_left - wheel_right)
            - (prev_wheel_left - prev_wheel_right)
        ).abs()
        wheel_sign_flip = (
            ((wheel_left * prev_wheel_left) < 0.0) & (wheel_left.abs() + prev_wheel_left.abs() > 0.035)
        ).float() + (
            ((wheel_right * prev_wheel_right) < 0.0) & (wheel_right.abs() + prev_wheel_right.abs() > 0.035)
        ).float()
        action_diff_limit = 0.10 if self.max_command_vx > 0.0 else 0.05
        wheel_vel_diff_limit = 0.14 if self.max_command_vx > 0.0 else 0.08
        action_diff_excess = (action_diff - action_diff_limit).clamp_min(0.0)
        wheel_vel_diff_excess = (wheel_vel_diff - wheel_vel_diff_limit).clamp_min(0.0)
        wheel_speed_limit = 0.55 if self.max_command_vx > 0.0 else 0.12
        wheel_speed_excess = (wheel_speed - wheel_speed_limit).clamp_min(0.0)
        if self.robot.num_leg_joints:
            leg_pos_error = torch.linalg.vector_norm(leg_pos_delta, dim=-1) / (self.robot.num_leg_joints ** 0.5)
            leg_vel_error = torch.linalg.vector_norm(leg_vel_scaled, dim=-1) / (self.robot.num_leg_joints ** 0.5)
            leg_dof_vel_sq = torch.sum(self.robot.leg_vel.square(), dim=-1)
            leg_dof_acc = (self.robot.leg_vel - self.prev_leg_vel) / self.progress_dt
            leg_dof_acc_sq = torch.sum(leg_dof_acc.square(), dim=-1)
        else:
            leg_pos_error = torch.zeros_like(height)
            leg_vel_error = torch.zeros_like(height)
            leg_dof_vel_sq = torch.zeros_like(height)
            leg_dof_acc_sq = torch.zeros_like(height)

        reward_alive = torch.ones_like(height)
        wl_tracking_lin_vel = torch.exp(
            -velocity_error.square() / max(self.wl_tracking_sigma, 1e-6)
        )
        wl_tracking_lin_vel_enhance = torch.exp(
            -velocity_error.square() / max(self.wl_tracking_sigma * 10.0, 1e-6)
        ) - 1.0
        wl_tracking_ang_vel = torch.exp(
            -yaw_rate_error.square() / max(self.wl_tracking_ang_sigma, 1e-6)
        )
        wl_base_height = torch.exp(
            -height_error.square() / max(self.wl_height_sigma, 1e-6)
        )
        wl_orientation = roll.square() + pitch.square()
        wl_lin_vel_z = vz.square()
        wl_ang_vel_xy = roll_rate.square() + pitch_rate.square()
        wl_action_rate = self.robot.action_difference.square()
        wl_action_smooth = self.robot.action_acceleration.square()
        wl_nominal_state = leg_pos_error.square()
        reward_upright = torch.exp(
            -30.0 * roll.square()
            -12.0 * pitch_tracking_error.square()
            -self.upright_roll_rate_scale * roll_rate.square()
            -self.upright_pitch_rate_scale * pitch_rate.square()
        )
        reward_height = torch.exp(-400.0 * height_error.square())
        reward_velocity = torch.exp(-600.0 * velocity_error.square())
        reward_direction = torch.where(
            command_active,
            (forward_vx * torch.sign(command_vx) / max(self.max_command_vx, 1e-6)).clamp(-1.0, 1.0),
            torch.ones_like(vx),
        )
        forward_progress = torch.where(
            command_active,
            (forward_vx * torch.sign(command_vx) / command_vx.abs().clamp_min(1e-6)).clamp(-1.0, 1.5),
            torch.ones_like(vx),
        )
        displacement_progress = torch.where(
            command_active,
            (delta_x * torch.sign(command_vx) / (command_vx.abs().clamp_min(1e-6) * self.progress_dt)).clamp(-1.0, 1.5),
            torch.ones_like(vx),
        )
        forward_deficit = torch.where(
            command_active,
            (0.85 - forward_progress).clamp_min(0.0),
            torch.zeros_like(vx),
        )
        speed_shortfall = torch.where(
            command_active,
            (0.75 - forward_progress).clamp_min(0.0),
            torch.zeros_like(vx),
        )
        reward_position = torch.exp(-18.0 * position_error.square())
        reward_yaw = torch.exp(-20.0 * yaw_error.square())
        reward_yaw_rate = torch.exp(-2.0 * yaw_rate_error.square())
        reward_leg_neutral = torch.exp(-3.0 * leg_pos_error.square() - 0.4 * leg_vel_error.square())
        stand_reward = reward_upright * reward_height
        settle_error = (
            35.0 * velocity_error.square()
            + 7.0 * position_error.square()
            + 2.0 * lateral_velocity_error.square()
            + 16.0 * yaw_error.square()
            + 12.0 * yaw_rate_error.square()
            + self.settle_roll_rate_scale * roll_rate.square()
            + self.settle_pitch_rate_scale * pitch_rate.square()
            + 0.10 * wheel_speed_excess.square()
            + 0.22 * wheel_vel_diff_excess.square()
            + self.settle_wheel_delta_scale * wheel_delta.square()
            + self.settle_wheel_common_delta_scale * wheel_common_delta.square()
            + self.settle_wheel_diff_delta_scale * wheel_diff_delta.square()
            + 0.26 * self.robot.action_difference.square()
            + 0.14 * self.robot.action_acceleration.square()
            + 0.8 * leg_pos_error.square()
            + 0.2 * leg_vel_error.square()
        )
        settle_reward = stand_reward * torch.exp(-settle_error)
        quiet_state = stand_reward * torch.exp(
            -16.0 * velocity_error.square()
            - 6.0 * roll_rate.square()
            - 4.0 * pitch_rate.square()
            - 4.0 * yaw_rate_error.square()
        )
        quiet_wheel_penalty = -quiet_state * (
            wheel_delta.square()
            + 1.5 * wheel_common_delta.square()
            + 0.75 * wheel_diff_delta.square()
            + 0.8 * wheel_sign_flip
            + 2.5 * self.robot.wheel_action_acceleration.square()
        )

        roll_penalty = -roll.square()
        roll_rate_penalty = -roll_rate.square()
        pitch_penalty = -pitch_tracking_error.square()
        pitch_rate_penalty = -pitch_rate.square()
        yaw_penalty = -yaw_error.square()
        yaw_rate_penalty = -yaw_rate.square()
        position_penalty = -(position_error - 0.02).clamp_min(0.0).square()
        lateral_velocity_penalty = -vy.square()
        lin_vel_z_penalty = -vz.square()
        action_diff_penalty = -action_diff_excess.square()
        wheel_vel_diff_penalty = -wheel_vel_diff_excess.square()
        wheel_speed_penalty = -wheel_speed_excess.square()
        wheel_delta_penalty = -wheel_delta.square()
        wheel_common_delta_penalty = -wheel_common_delta.square()
        wheel_diff_delta_penalty = -wheel_diff_delta.square()
        wheel_sign_flip_penalty = -wheel_sign_flip
        wheel_action_diff_penalty = -self.robot.wheel_action_difference.square()
        wheel_action_accel_penalty = -self.robot.wheel_action_acceleration.square()
        wheel_action_magnitude_penalty = -self.robot.wheel_action_magnitude.square()
        action_penalty = -self.effort.square()
        smoothness_penalty = -(
            self.robot.action_difference.square()
            + 0.05 * self.robot.action_magnitude
        )
        action_accel_penalty = -self.robot.action_acceleration.square()
        leg_pos_penalty = -leg_pos_error.square()
        leg_vel_penalty = -leg_vel_error.square()
        leg_action_penalty = -self.robot.leg_action_magnitude.square()
        leg_action_diff_penalty = -self.robot.leg_action_difference.square()
        forward_deficit_penalty = -forward_deficit.square()
        speed_shortfall_penalty = -speed_shortfall.square()
        base_height_target_error = height - self.base_height_target
        low_height_penalty = -(
            (self.height_floor - height).clamp_min(0.0)
            / max(self.base_height_target - self.height_floor, 1e-6)
        ).square()
        target_tracking_lin_vel = torch.exp(
            -velocity_error.square() / max(self.tracking_sigma, 1e-6)
        )
        target_tracking_lin_vel_enhance = torch.exp(
            -velocity_error.square() / max(self.tracking_sigma * 10.0, 1e-6)
        ) - 1.0
        safe_target_tracking_lin_vel = stand_reward * target_tracking_lin_vel
        safe_target_tracking_lin_vel_enhance = stand_reward * target_tracking_lin_vel_enhance
        target_tracking_ang_vel = torch.exp(
            -yaw_rate_error.square() / max(self.tracking_sigma, 1e-6)
        )
        target_tracking_ang_vel_enhance = torch.exp(
            -yaw_rate_error.square() / max(self.tracking_sigma * 10.0, 1e-6)
        ) - 1.0
        target_base_height = torch.exp(-base_height_target_error.square() / 0.001)
        if self.robot.num_leg_joints >= 4:
            left_nominal = self.robot.leg_pos[..., : self.robot.num_leg_joints // 2].mean(-1)
            right_nominal = self.robot.leg_pos[..., self.robot.num_leg_joints // 2 :].mean(-1)
            target_nominal_state = torch.square(left_nominal - right_nominal)
        else:
            target_nominal_state = leg_pos_error.square()
        target_lin_vel_z = vz.square()
        target_ang_vel_xy = roll_rate.square() + pitch_rate.square()
        target_orientation = self.robot.up[..., :2].square().sum(-1)
        target_dof_vel = self.robot.wheel_vel.square().sum(-1)
        if self.robot.num_leg_joints:
            target_dof_vel = target_dof_vel + self.robot.leg_vel.square().sum(-1)
        target_dof_acc = ((self.robot.wheel_vel - self.prev_wheel_vel) / self.progress_dt).square().sum(-1)
        if self.robot.num_leg_joints:
            target_dof_acc = target_dof_acc + ((self.robot.leg_vel - self.prev_leg_vel) / self.progress_dt).square().sum(-1)
        target_torques = self.effort.square()
        target_action_rate = self.robot.action_difference.square()
        target_action_smooth = self.robot.action_acceleration.square()
        target_collision = self._get_contact_penalty()
        terminal_collision = self._get_contact_penalty(self.contact_terminate_force_threshold)
        target_dof_pos_limits = self._get_dof_pos_limit_penalty()

        terminated_roll = roll_error > self.termination_roll
        terminated_pitch = pitch_error > self.termination_pitch
        terminated_height = height < self.min_height
        terminated_xy = self.robot.pos[..., :2].norm(dim=-1) > self.max_xy
        misbehave = terminated_roll | terminated_pitch | terminated_height | terminated_xy
        if self.terminate_on_body_contact:
            misbehave = misbehave | (terminal_collision > 0.0)
        fall_penalty = -misbehave.float()

        reward = (
            self.reward_alive_weight * reward_alive
            + self.reward_stand_weight * _as_column(stand_reward)
            + self.reward_upright_weight * _as_column(reward_upright)
            + self.reward_height_weight * _as_column(reward_height)
            + self.reward_settle_weight * _as_column(settle_reward)
            + self.reward_leg_neutral_weight * _as_column(reward_leg_neutral)
            + self.penalty_fall_weight * _as_column(fall_penalty)
            + self.penalty_roll_weight * _as_column(roll_penalty)
            + self.penalty_roll_rate_weight * _as_column(roll_rate_penalty)
            + self.penalty_pitch_weight * _as_column(pitch_penalty)
            + self.penalty_pitch_rate_weight * _as_column(pitch_rate_penalty)
            + self.penalty_yaw_weight * _as_column(yaw_penalty)
            + self.penalty_yaw_rate_weight * _as_column(yaw_rate_penalty)
            + self.penalty_position_weight * _as_column(position_penalty)
            + self.penalty_lateral_velocity_weight * _as_column(lateral_velocity_penalty)
            + self.penalty_lin_vel_z_weight * _as_column(lin_vel_z_penalty)
            + self.penalty_action_diff_weight * _as_column(action_diff_penalty)
            + self.penalty_wheel_vel_diff_weight * _as_column(wheel_vel_diff_penalty)
            + self.penalty_wheel_speed_weight * _as_column(wheel_speed_penalty)
            + self.penalty_wheel_delta_weight * _as_column(wheel_delta_penalty)
            + self.penalty_wheel_common_delta_weight * _as_column(wheel_common_delta_penalty)
            + self.penalty_wheel_diff_delta_weight * _as_column(wheel_diff_delta_penalty)
            + self.penalty_wheel_sign_flip_weight * _as_column(wheel_sign_flip_penalty)
            + self.penalty_wheel_action_diff_weight * _as_column(wheel_action_diff_penalty)
            + self.penalty_wheel_action_accel_weight * _as_column(wheel_action_accel_penalty)
            + self.penalty_wheel_action_magnitude_weight * _as_column(wheel_action_magnitude_penalty)
            + self.penalty_quiet_wheel_weight * _as_column(quiet_wheel_penalty)
            + self.penalty_action_weight * _as_column(action_penalty)
            + self.penalty_smoothness_weight * _as_column(smoothness_penalty)
            + self.penalty_action_accel_weight * _as_column(action_accel_penalty)
            + self.penalty_leg_pos_weight * _as_column(leg_pos_penalty)
            + self.penalty_leg_vel_weight * _as_column(leg_vel_penalty)
            + self.penalty_leg_action_weight * _as_column(leg_action_penalty)
            + self.penalty_leg_action_diff_weight * _as_column(leg_action_diff_penalty)
            + self.penalty_forward_deficit_weight * _as_column(forward_deficit_penalty)
            + self.penalty_speed_shortfall_weight * _as_column(speed_shortfall_penalty)
            + self.penalty_low_height_weight * _as_column(low_height_penalty)
            + self.reward_tracking_lin_vel_weight * _as_column(safe_target_tracking_lin_vel)
            + self.reward_tracking_lin_vel_enhance_weight * _as_column(safe_target_tracking_lin_vel_enhance)
            + self.reward_tracking_ang_vel_weight * _as_column(target_tracking_ang_vel)
            + self.reward_tracking_ang_vel_enhance_weight * _as_column(target_tracking_ang_vel_enhance)
            + self.reward_base_height_weight * _as_column(target_base_height)
            + self.reward_nominal_state_weight * _as_column(target_nominal_state)
            + self.reward_lin_vel_z_weight * _as_column(target_lin_vel_z)
            + self.reward_ang_vel_xy_weight * _as_column(target_ang_vel_xy)
            + self.reward_orientation_weight * _as_column(target_orientation)
            + self.reward_dof_vel_weight * _as_column(target_dof_vel)
            + self.reward_dof_acc_weight * _as_column(target_dof_acc)
            + self.reward_torques_weight * _as_column(target_torques)
            + self.reward_action_rate_weight * _as_column(target_action_rate)
            + self.reward_action_smooth_weight * _as_column(target_action_smooth)
            + self.reward_collision_weight * _as_column(target_collision)
            + self.reward_dof_pos_limits_weight * _as_column(target_dof_pos_limits)
        )
        reward = _as_column(reward)
        hasnan = torch.isnan(self.robot_state).any(-1)
        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["uprightness"].lerp_(uprightness, 1 - self.alpha)
        self.stats["roll_error"].lerp_(roll_error, 1 - self.alpha)
        self.stats["roll_rate_error"].lerp_(roll_rate_error, 1 - self.alpha)
        self.stats["pitch_error"].lerp_(pitch_error, 1 - self.alpha)
        self.stats["pitch_target"].lerp_(pitch_target.abs(), 1 - self.alpha)
        self.stats["pitch_tracking_error"].lerp_(pitch_tracking_error, 1 - self.alpha)
        self.stats["pitch_rate_error"].lerp_(pitch_rate_error, 1 - self.alpha)
        self.stats["height"].lerp_(height, 1 - self.alpha)
        self.stats["height_error"].lerp_(height_error, 1 - self.alpha)
        self.stats["terminated_roll"].lerp_(terminated_roll.float(), 1 - self.alpha)
        self.stats["terminated_pitch"].lerp_(terminated_pitch.float(), 1 - self.alpha)
        self.stats["terminated_height"].lerp_(terminated_height.float(), 1 - self.alpha)
        self.stats["terminated_xy"].lerp_(terminated_xy.float(), 1 - self.alpha)
        self.stats["fall_penalty"].lerp_(fall_penalty, 1 - self.alpha)
        self.stats["stand_reward"].lerp_(stand_reward, 1 - self.alpha)
        self.stats["settle_reward"].lerp_(settle_reward, 1 - self.alpha)
        self.stats["quiet_state"].lerp_(quiet_state, 1 - self.alpha)
        self.stats["quiet_wheel_penalty"].lerp_(-quiet_wheel_penalty, 1 - self.alpha)
        self.stats["velocity_error"].lerp_(velocity_error, 1 - self.alpha)
        self.stats["forward_velocity"].lerp_(forward_vx, 1 - self.alpha)
        self.stats["velocity_alignment"].lerp_(reward_direction, 1 - self.alpha)
        self.stats["forward_progress"].lerp_(forward_progress, 1 - self.alpha)
        self.stats["displacement_progress"].lerp_(displacement_progress, 1 - self.alpha)
        self.stats["forward_displacement"].lerp_(forward_displacement, 1 - self.alpha)
        self.stats["forward_deficit"].lerp_(forward_deficit, 1 - self.alpha)
        self.stats["speed_shortfall"].lerp_(speed_shortfall, 1 - self.alpha)
        self.stats["position_error"].lerp_(position_error, 1 - self.alpha)
        self.stats["lateral_velocity_error"].lerp_(lateral_velocity_error, 1 - self.alpha)
        self.stats["vertical_velocity_error"].lerp_(vertical_velocity_error, 1 - self.alpha)
        self.stats["yaw_error"].lerp_(yaw_error, 1 - self.alpha)
        self.stats["yaw_rate_error"].lerp_(yaw_rate_error, 1 - self.alpha)
        self.stats["action_diff"].lerp_(action_diff, 1 - self.alpha)
        self.stats["wheel_vel_diff"].lerp_(wheel_vel_diff, 1 - self.alpha)
        self.stats["wheel_speed"].lerp_(wheel_speed, 1 - self.alpha)
        self.stats["wheel_delta"].lerp_(wheel_delta, 1 - self.alpha)
        self.stats["wheel_common_delta"].lerp_(wheel_common_delta, 1 - self.alpha)
        self.stats["wheel_diff_delta"].lerp_(wheel_diff_delta, 1 - self.alpha)
        self.stats["wheel_sign_flip"].lerp_(wheel_sign_flip, 1 - self.alpha)
        self.stats["wheel_action_diff"].lerp_(self.robot.wheel_action_difference, 1 - self.alpha)
        self.stats["wheel_action_acceleration"].lerp_(-self.robot.wheel_action_acceleration, 1 - self.alpha)
        self.stats["wheel_action_magnitude"].lerp_(self.robot.wheel_action_magnitude, 1 - self.alpha)
        self.stats["action_smoothness"].lerp_(-self.robot.action_difference, 1 - self.alpha)
        self.stats["action_acceleration"].lerp_(-self.robot.action_acceleration, 1 - self.alpha)
        self.stats["leg_pos_error"].lerp_(leg_pos_error, 1 - self.alpha)
        self.stats["leg_vel_error"].lerp_(leg_vel_error, 1 - self.alpha)
        self.stats["leg_neutral_reward"].lerp_(reward_leg_neutral, 1 - self.alpha)
        self.stats["leg_action_magnitude"].lerp_(self.robot.leg_action_magnitude, 1 - self.alpha)
        self.stats["leg_action_diff"].lerp_(self.robot.leg_action_difference, 1 - self.alpha)
        self.stats["command_vx"].lerp_(self.command[..., 0].abs(), 1 - self.alpha)
        self.stats["command_yaw_rate"].lerp_(self.command[..., 1].abs(), 1 - self.alpha)
        self.stats["wl_tracking_lin_vel"].lerp_(wl_tracking_lin_vel, 1 - self.alpha)
        self.stats["wl_tracking_ang_vel"].lerp_(wl_tracking_ang_vel, 1 - self.alpha)
        self.stats["wl_base_height"].lerp_(wl_base_height, 1 - self.alpha)
        self.stats["wl_orientation"].lerp_(wl_orientation, 1 - self.alpha)
        self.stats["wl_action_rate"].lerp_(wl_action_rate, 1 - self.alpha)
        self.stats["wl_action_smooth"].lerp_(wl_action_smooth, 1 - self.alpha)
        self.stats["wl_dof_vel"].lerp_(leg_dof_vel_sq, 1 - self.alpha)
        self.stats["wl_dof_acc"].lerp_(leg_dof_acc_sq, 1 - self.alpha)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.prev_x[:] = forward_displacement
        self.prev_wheel_vel[:] = self.robot.wheel_vel
        if self.robot.num_leg_joints:
            self.prev_leg_vel[:] = self.robot.leg_vel

        return TensorDict(
            {
                "agents": {"reward": reward.unsqueeze(-1)},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
