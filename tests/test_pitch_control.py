"""v6 signed posture shaping and physical wheel smoothness, without action overrides."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage, observation_noise_vector, precision_rewards_for_stage


class PitchControlTests(unittest.TestCase):
    def setUp(self):
        p = env_params_for_stage('joint_tracking_v6')
        p.domain_randomization.enabled = False
        p.robust.pushes_enabled = False
        self.env = UZ05Env('joint_tracking_v6', params=p)
        self.env.reset(seed=5)

    def tearDown(self):
        self.env.close()

    def test_near_zero_pitch_positive_large_pitch_negative_and_symmetric(self):
        e = self.env
        values = [e._pitch_terms(0, np.radians(x)) for x in (0, 3, 5, 10, 12, 20)]
        self.assertAlmostEqual(sum(values[0].values()), 3)
        self.assertEqual(values[1]['pitch_error'], 0)
        net = [v['pitch_alignment'] + v['pitch_error'] for v in values]
        self.assertTrue(all(a > b for a, b in zip(net, net[1:])))
        self.assertGreater(net[2], 0)
        self.assertLess(net[3], 0)
        for x in (3, 5, 10, 12, 20, 45):
            self.assertEqual(e._pitch_terms(.01, np.radians(x)), e._pitch_terms(.01, np.radians(-x)))
            self.assertTrue(np.isfinite(list(e._pitch_terms(0, np.radians(x)).values())).all())

    def test_noise_separates_terrain_and_height_without_disabling_randomization(self):
        p = env_params_for_stage('joint_tracking_v6')
        noise = observation_noise_vector(p.noise)
        self.assertAlmostEqual(noise['terrain_preview']/5, .002)
        self.assertAlmostEqual(noise['base_height_rel']/5, .005)
        self.assertAlmostEqual(noise['wheel_joint_vel']/.05, .10)
        self.assertTrue(p.domain_randomization.enabled and p.robust.pushes_enabled)
        old = observation_noise_vector(env_params_for_stage('joint_tracking_v5').noise)
        self.assertAlmostEqual(old['terrain_preview']/5, .05)

    def test_physical_acceleration_includes_all_four_substeps(self):
        e = self.env
        initial = e.sim.wheel_velocities().copy()
        samples = [initial]
        real_step = e.sim.step
        def record():
            real_step()
            samples.append(e.sim.wheel_velocities().copy())
        e.sim.step = record
        _, _, _, _, info = e.step(np.zeros(6))
        acceleration = np.diff(samples, axis=0)/.002
        np.testing.assert_allclose(e._previous_control['wheel_acceleration_samples'], acceleration)
        self.assertAlmostEqual(info['wheel_acceleration_sq'], np.square(acceleration).mean())
        self.assertTrue(np.isfinite(info['reward_terms']['wheel_acceleration']))
        self.assertIn('pitch_alignment', info['reward_terms'])
        self.assertIn('pitch_error', info['reward_terms'])

    def test_alternating_wheel_acceleration_cannot_cancel_its_cost(self):
        e = self.env
        e.step(np.zeros(6))
        c = e._previous_control.copy()
        c['wheel_acceleration_samples'] = np.array([[150,150],[-150,-150],[150,150],[-150,-150]])
        self.assertAlmostEqual(e._precision_terms(c)['wheel_acceleration'], -.02)
        c['wheel_acceleration_samples'] *= 2
        self.assertLess(e._precision_terms(c)['wheel_acceleration'], -.02)

    def test_v6_retains_task_budget_and_does_not_change_actions(self):
        e = self.env
        action = np.array([.1,-.1,.01,-.01,.02,-.02])
        e.step(action)
        c=e._previous_control
        np.testing.assert_allclose(c['wheel_target'], action[:2]*16)
        np.testing.assert_allclose(np.arctan2(np.sin(c['joint_target']),np.cos(c['joint_target'])),action[2:]*np.pi)
        p=precision_rewards_for_stage(e.stage,e.params)
        self.assertLess(p.wheel_target_change,.12)
        self.assertEqual(e.stage.tracking_vx_weight,2.5)
        self.assertEqual(e.stage.tracking_vx_sigma,.15)


if __name__ == '__main__':
    unittest.main()
