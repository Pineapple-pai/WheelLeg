from __future__ import annotations

import inspect
import unittest

import numpy as np

from uz05.pure_ppo_stand import (
    PITCH_INIT_LIMIT_RAD,
    PITCH_RATE_INIT_LIMIT_RAD_S,
    Q_REF,
    PurePPOStandEnv,
    compute_stand_reward,
)


class PurePPOResetAndRewardTests(unittest.TestCase):
    def test_reset_is_closed_loop_contact_valid_and_has_no_noise_or_dr(self):
        env = PurePPOStandEnv(seed=21)
        signature = (
            env.sim.model.body_mass.copy(),
            env.sim.model.body_inertia.copy(),
            env.sim.model.geom_friction.copy(),
        )
        for seed in range(30):
            obs, info = env.reset(seed=seed)
            state = env._state()
            self.assertTrue(np.isfinite(obs).all())
            self.assertLessEqual(info["reset_equality_residual_max_m"], 1e-8)
            np.testing.assert_allclose(env.sim.joint_positions(), Q_REF, atol=1e-12)
            self.assertTrue(np.all(state["wheel_hit"] > 0.5))
            self.assertEqual(state["body_hit"], 0.0)
            self.assertFalse(info["domain_randomization_enabled"])
            self.assertFalse(info["observation_noise_enabled"])
            self.assertEqual(info["observation_noise_width_max"], 0.0)
            self.assertLessEqual(abs(info["initial_pitch_rad"]), PITCH_INIT_LIMIT_RAD)
            self.assertLessEqual(abs(info["initial_pitch_rate_rad_s"]), PITCH_RATE_INIT_LIMIT_RAD_S)
            np.testing.assert_array_equal(env.sim.model.body_mass, signature[0])
            np.testing.assert_array_equal(env.sim.model.body_inertia, signature[1])
            np.testing.assert_array_equal(env.sim.model.geom_friction, signature[2])
        env.close()

    def test_reward_prefers_upright_and_lower_pitch_rate(self):
        base = dict(
            roll_rad=0.0,
            pitch_rad=0.0,
            pitch_rate_rad_s=0.0,
            horizontal_velocity_m_s=0.0,
            position_drift_m=0.0,
            height_error_m=0.0,
            leg_length_error_m=0.0,
            action=np.zeros(6),
            previous_action=np.zeros(6),
            normalized_effort=0.0,
            terminated=False,
        )
        upright, _, _ = compute_stand_reward(**base)
        tilted, _, _ = compute_stand_reward(**(base | {"pitch_rad": np.deg2rad(8)}))
        moving_rate, _, _ = compute_stand_reward(**(base | {"pitch_rate_rad_s": 0.1}))
        self.assertGreater(upright, tilted)
        self.assertGreater(upright, moving_rate)

    def test_terminal_failure_is_finite_and_penalized(self):
        base = dict(
            roll_rad=0.0,
            pitch_rad=0.0,
            pitch_rate_rad_s=0.0,
            horizontal_velocity_m_s=0.0,
            position_drift_m=0.0,
            height_error_m=0.0,
            leg_length_error_m=0.0,
            action=np.zeros(6),
            previous_action=np.zeros(6),
            normalized_effort=0.0,
            terminated=False,
        )
        alive, _, _ = compute_stand_reward(**base)
        failed, raw, weighted = compute_stand_reward(**(base | {"terminated": True}))
        self.assertTrue(np.isfinite(failed))
        self.assertLess(failed, alive)
        self.assertEqual(raw["terminal_failure"], 1.0)
        self.assertLess(weighted["termination"], 0.0)

    def test_reward_signature_has_state_and_action_costs_but_no_teacher_output(self):
        names = set(inspect.signature(compute_stand_reward).parameters)
        self.assertIn("action", names)
        self.assertIn("previous_action", names)
        self.assertIn("normalized_effort", names)
        self.assertNotIn("target_action", names)
        self.assertNotIn("expert_action", names)
        self.assertNotIn("pitch_action", names)

    def test_zero_action_maps_to_qref_and_zero_wheel_target(self):
        env = PurePPOStandEnv(seed=22)
        env.reset(seed=22, options={"initial_pitch_rad": 0.0, "initial_pitch_rate_rad_s": 0.0})
        _, _, terminated, _, info = env.step(np.zeros(6, dtype=np.float32))
        self.assertFalse(terminated)
        np.testing.assert_array_equal(info["wheel_target_rad_s"], [0.0, 0.0])
        np.testing.assert_allclose(info["joint_target_rad"], Q_REF, atol=1e-12)
        env.close()


if __name__ == "__main__":
    unittest.main()
