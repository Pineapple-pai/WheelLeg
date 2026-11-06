"""Behavioral checks for command coverage, circular actions and tracking rewards."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from uz05.env import UZ05Env
from uz05.spec import EnvParams


class JointTrackingTests(unittest.TestCase):
    def setUp(self):
        params = EnvParams()
        params.domain_randomization.enabled = False
        params.noise.enabled = False
        self.env = UZ05Env("joint_tracking", params=params, seed=5)
        self.env.reset(seed=5)

    def tearDown(self):
        self.env.close()

    def test_first_reset_has_full_command_distribution(self):
        commands = []
        for seed in range(80):
            self.env.reset(seed=seed)
            commands.append(self.env.command.copy())
        cmds = np.asarray(commands)
        for axis in (0, 2):
            self.assertLess(cmds[:, axis].min(), -0.1)
            self.assertGreater(cmds[:, axis].max(), 0.1)
        self.assertGreater(np.ptp(cmds[:, 3]), 0.04)
        self.assertTrue(np.any(np.all(cmds[:, [0, 2]] == 0, axis=1)))
        self.assertTrue(np.any(np.all(cmds[:, [0, 2]] != 0, axis=1)))

    def test_circular_positions_and_independent_wheel_velocities(self):
        bank = self.env.actuators
        q = np.full(4, np.pi - 0.01)
        state = {"joint_pos": q, "joint_vel": np.zeros(4), "wheel_vel": np.zeros(2)}
        action = np.array([0.25, -0.5] + [(-np.pi + 0.01) / np.pi] * 4)
        result = bank.compute(action, state=state, dt=0.008)
        np.testing.assert_allclose(result["wheel_target"], [4, -8])
        np.testing.assert_allclose(bank.diagnostics["joint_target"], q + 0.02)
        state["joint_pos"] = q + 4 * np.pi
        result2 = bank.compute(action, state=state, dt=0.008)
        np.testing.assert_allclose(result["leg_torque"], result2["leg_torque"], atol=1e-10)
        # Zero action remains an active motor position target, not zero effort.
        state["joint_pos"] = np.full(4, 0.1)
        result = bank.compute(np.zeros(6), state=state, dt=0.008)
        self.assertTrue(np.all(result["leg_torque"] < 0))

    def test_no_joint_angle_clamp_or_movement_drift_termination(self):
        model = self.env.sim.model
        for addr in self.env.sim.hip_qpos_adr:
            idx = int(np.flatnonzero(model.jnt_qposadr == addr)[0])
            self.assertEqual(model.jnt_limited[idx], 0)
        self.env.sim.data.qpos[0] += 2
        self.env.sim.forward()
        _, _, reason = self.env._termination()
        self.assertNotIn("station_limit", reason)
        self.assertNotIn("drift_limit", reason)

    def test_zero_command_uses_same_tracking_reward(self):
        self.env.command[:] = [0, 0, 0, 0.265, 0]
        _, reward, _, _, info = self.env.step(np.zeros(6))
        self.assertTrue(np.isfinite(reward))
        self.assertGreater(info["reward_terms"]["track_vx"], 0)
        self.assertGreater(info["reward_terms"]["track_yaw"], 0)
        self.assertNotIn("joint_neutral", info["reward_terms"])
        self.assertNotIn("stand_balance_action", info["reward_terms"])

    def test_stance_measurement_invariant_to_world_yaw(self):
        from scipy.spatial.transform import Rotation
        sim = self.env.sim
        original = sim.wheel_stance()
        # Rotate the entire configuration in world coordinates.
        old = sim.data.qpos[3:7].copy()
        rot = Rotation.from_euler("z", 1.2) * Rotation.from_quat([old[1], old[2], old[3], old[0]])
        q = rot.as_quat()
        sim.data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        sim.forward()
        for key, value in original.items():
            self.assertAlmostEqual(sim.wheel_stance()[key], value, places=9)

    def test_v2_rejects_persistent_diagonal_stance(self):
        from unittest.mock import patch
        env = UZ05Env("joint_tracking_v2", params=self.env.params)
        try:
            env.reset(seed=5)
            env.steps = env.stage.stance_reset_grace_steps + 1
            bad = {"wheel_stagger_m": 0.50, "wheel_offset_max_m": 0.25}
            with patch.object(env.sim, "wheel_stance", return_value=bad):
                for _ in range(env.stage.stance_invalid_hold_steps - 1):
                    self.assertNotIn("wheel_stance_limit", env._termination()[2])
                self.assertIn("wheel_stance_limit", env._termination()[2])
            with patch.object(env.sim, "wheel_stance", return_value={"wheel_stagger_m": .01, "wheel_offset_max_m": .02}):
                self.assertNotIn("wheel_stance_limit", env._termination()[2])
                self.assertEqual(env.stance_invalid_steps, 0)
        finally:
            env.close()

    def test_v2_does_not_reward_diagonal_prop_as_normal_tracking(self):
        from unittest.mock import patch
        env = UZ05Env("joint_tracking_v2", params=self.env.params)
        try:
            env.reset(seed=5)
            state = {"nominal_joint_pos": np.zeros(4), "joint_pos": env.sim.joint_positions(),
                     "joint_vel": env.sim.joint_velocities(), "wheel_vel": env.sim.wheel_velocities()}
            control = env.actuators.compute(np.zeros(6), state=state, dt=.008)
            with patch.object(env.sim, "wheel_stance", return_value={"wheel_stagger_m": .01, "wheel_offset_max_m": .02}):
                good, good_terms = env._reward(control, state, np.zeros(6), False)
            with patch.object(env.sim, "wheel_stance", return_value={"wheel_stagger_m": .50, "wheel_offset_max_m": .25}):
                bad, bad_terms = env._reward(control, state, np.zeros(6), False)
            self.assertGreater(good, bad)
            self.assertGreater(good_terms["track_height"], bad_terms["track_height"])
            self.assertLess(bad_terms["stance_geometry"], 0)
        finally:
            env.close()

    def test_commands_resample_without_training_schedule(self):
        old = self.env.command.copy()
        self.env.command_steps_left = 0
        self.env._advance_command()
        self.assertFalse(np.array_equal(self.env.command, old))
        np.testing.assert_array_equal(self.env.command, self.env.command_target)


if __name__ == "__main__":
    unittest.main()
