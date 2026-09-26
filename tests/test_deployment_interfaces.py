from __future__ import annotations

import struct
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from uz05.actuators import quantize_linear
from uz05.deployment import (
    DmMitCommand,
    action_to_physical_targets,
    c620_current_to_raw,
    encode_c620_group,
)
from uz05.env import UZ05Env
from uz05.spec import ACTOR_OBS_DIM, MOTOR_CONTROL_RATE_HZ, MOTOR_TICKS_PER_POLICY, POLICY_RATE_HZ


class DeploymentInterfaceTest(unittest.TestCase):
    def test_frozen_rate_ratio(self):
        self.assertEqual(MOTOR_CONTROL_RATE_HZ, 500)
        self.assertEqual(POLICY_RATE_HZ, 125)
        self.assertEqual(MOTOR_TICKS_PER_POLICY, 4)

    def test_dm_mit_payload_is_eight_bytes(self):
        payload = DmMitCommand(p_des=0.1, kp=100.0, kd=4.0).encode()
        self.assertEqual(len(payload), 8)

    def test_linear_quantization_clips_endpoints(self):
        decoded, encoded = quantize_linear(np.array([-99.0, 99.0]), -12.5, 12.5, 16)
        np.testing.assert_allclose(decoded, [-12.5, 12.5])
        np.testing.assert_array_equal(encoded, [0, 65535])

    def test_c620_current_mapping_and_group_order(self):
        self.assertEqual(c620_current_to_raw(20.0), 16384)
        self.assertEqual(c620_current_to_raw(-20.0), -16384)
        payload = encode_c620_group({1: 20.0, 2: -20.0}, group=0x200)
        self.assertEqual(struct.unpack(">hhhh", payload), (16384, -16384, 0, 0))

    def test_action_layout_is_stable(self):
        wheel, leg = action_to_physical_targets(np.zeros(6))
        np.testing.assert_allclose(wheel, np.zeros(2))
        self.assertEqual(leg.shape, (4,))

    def test_positive_wheel_action_uses_forward_body_direction(self):
        action = np.array([0.25, 0.25, 0.0, 0.0, 0.0, 0.0])
        wheel, _ = action_to_physical_targets(action)
        np.testing.assert_allclose(wheel, [-2.0, -2.0])

        env = UZ05Env(stage="stand", seed=8, init_scale=0.0, stand_level=0)
        env.reset(seed=8)
        result = env.actuators.compute(
            action,
            state={
                "joint_pos": env.sim.joint_positions(),
                "joint_vel": env.sim.joint_velocities(),
                "wheel_vel": env.sim.wheel_velocities(),
            },
            dt=env.params.motor_control_dt,
        )
        env.close()
        np.testing.assert_allclose(result["wheel_target"], wheel)

    def test_environment_runs_four_motor_ticks_per_policy_step(self):
        env = UZ05Env(stage="stand", seed=1, init_scale=0.0, stand_level=0)
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs[:ACTOR_OBS_DIM].shape, (ACTOR_OBS_DIM,))
        calls = 0
        original = env.actuators.compute

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        env.actuators.compute = counted
        env.step(np.zeros(6, dtype=np.float32))
        env.close()
        self.assertEqual(calls, MOTOR_TICKS_PER_POLICY)

    def test_low_speed_zero_command_uses_stand_smoothing_reward(self):
        env = UZ05Env(
            stage="low_speed", seed=2, init_scale=0.0, stand_level=2,
            zero_command_prob_override=1.0,
        )
        env.reset(seed=2)
        env.params.rewards.stand_leg_target_rate = 10.0
        _, _, _, _, info = env.step(np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5]))
        env.close()
        self.assertEqual(info["command_target_vx"], 0.0)
        self.assertLess(info["reward_terms"]["stand_leg_target_rate"], 0.0)
        self.assertEqual(info["reward_terms"]["low_speed_startup_progress"], 0.0)
        self.assertEqual(info["reward_terms"]["low_speed_cruise_progress"], 0.0)
        self.assertEqual(info["reward_terms"]["low_speed_stable_motion"], 0.0)

    def test_low_speed_moving_reward_does_not_duplicate_velocity_error(self):
        env = UZ05Env(
            stage="low_speed", seed=3, init_scale=0.0, stand_level=2,
            vx_range_override=(0.04, 0.04), zero_command_prob_override=0.0,
            reverse_prob_override=0.0,
        )
        env.reset(seed=3)
        env.step(np.zeros(6, dtype=np.float32))
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        env.close()
        terms = info["reward_terms"]
        self.assertGreater(info["command_vx"], 0.01)
        self.assertEqual(terms["track_vx_error"], 0.0)
        self.assertEqual(terms["track_vx_forward_error"], 0.0)
        self.assertEqual(terms["track_vx_reverse_error"], 0.0)

    def test_low_speed_startup_signal_is_dense_and_signed(self):
        env = UZ05Env(stage="low_speed", seed=4, init_scale=0.0, stand_level=2)
        env.reset(seed=4)
        env.command[0] = 0.06
        env.command_target[0] = 0.06
        env._translation_episode_active = True
        env.steps = 16
        state = {
            "joint_pos": env.sim.joint_positions(),
            "joint_vel": env.sim.joint_velocities(),
        }
        control = {
            "leg_torque": np.zeros(4),
            "wheel_current": np.zeros(2),
        }

        def reward_terms(vx):
            env.sim.data.qvel[:3] = (vx, 0.0, 0.0)
            env.sim.forward()
            return env._reward(
                control, state, np.zeros(6, dtype=np.float32), False
            )[1]

        stopped = reward_terms(0.0)
        forward = reward_terms(0.03)
        backward = reward_terms(-0.03)
        env.close()

        self.assertEqual(stopped["low_speed_startup_progress"], 0.0)
        self.assertGreater(forward["low_speed_startup_progress"], 0.0)
        self.assertLess(backward["low_speed_startup_progress"], 0.0)
        self.assertEqual(forward["track_vx_progress"], 0.0)
        self.assertGreater(forward["track_vx"], stopped["track_vx"])

    def test_low_speed_cruise_penalty_only_when_posture_exceeds_envelope(self):
        env = UZ05Env(stage="low_speed", seed=5, init_scale=0.0, stand_level=2)
        env.reset(seed=5)
        env.command[0] = 0.10
        env.command_target[0] = 0.10
        env._translation_episode_active = True
        env.steps = 128
        state = {
            "joint_pos": env.sim.joint_positions(),
            "joint_vel": env.sim.joint_velocities(),
        }
        control = {"leg_torque": np.zeros(4), "wheel_current": np.zeros(2)}

        def terms_for(vx, pitch=0.0, pitch_rate=0.0):
            env._motion_vx_window.clear()
            env._motion_vx_window.extend([vx] * env._motion_vx_window.maxlen)
            env.sim.data.qvel[:3] = (vx, 0.0, 0.0)
            env.sim.data.qvel[4] = pitch_rate
            env.sim.data.qpos[3:7] = (
                np.cos(pitch / 2.0), 0.0, np.sin(pitch / 2.0), 0.0
            )
            env.sim.forward()
            return env._reward(control, state, np.zeros(6, dtype=np.float32), False)[1]

        stopped = terms_for(0.0)
        moving = terms_for(0.10)
        excessive_posture = terms_for(0.10, pitch=0.24, pitch_rate=1.6)
        env.params.rewards.low_speed_posture_gate_gain = 2.0
        stronger_gate = terms_for(0.10, pitch=0.24, pitch_rate=1.6)
        env.close()
        self.assertEqual(stopped["low_speed_stable_motion"], 0.0)
        self.assertEqual(moving["low_speed_stable_motion"], 0.0)
        self.assertAlmostEqual(excessive_posture["low_speed_stable_motion"], -6.0)
        self.assertAlmostEqual(
            excessive_posture["track_vx"] / moving["track_vx"],
            1.0 / 3.0,
            places=4,
        )
        self.assertAlmostEqual(
            excessive_posture["low_speed_cruise_progress"]
            / moving["low_speed_cruise_progress"],
            1.0 / 3.0,
            places=3,
        )
        self.assertAlmostEqual(
            stronger_gate["low_speed_cruise_progress"]
            / moving["low_speed_cruise_progress"],
            1.0 / 5.0,
            places=3,
        )
        self.assertGreater(moving["low_speed_cruise_progress"], stopped["low_speed_cruise_progress"])
        self.assertEqual(moving["low_speed_startup_progress"], 0.0)


if __name__ == "__main__":
    unittest.main()
