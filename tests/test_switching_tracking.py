"""Regression coverage for high-speed command switches and v13 reproducibility."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage, precision_rewards_for_stage


class SwitchingTrackingTest(unittest.TestCase):
    def env(self, stage='joint_tracking_v14'):
        p = env_params_for_stage(stage)
        p.domain_randomization.enabled = False
        p.robust.pushes_enabled = False
        e = UZ05Env(stage, params=p)
        e.reset(seed=8)
        self.addCleanup(e.close)
        return e

    def test_high_speed_events_are_distinct_and_preserve_unmodified_commands(self):
        e = self.env()
        e.steps = 100
        counts = dict(stop=0, reverse=0, height=0, turn=0)
        mid_reversals = 0
        for _ in range(4000):
            e.command[:] = [1.9, 0, 0, .29, 0]
            e.command_target[:] = e.command
            e._sample_command()
            vx, _, yaw, height, _ = e.command
            if vx == 0 and yaw == 0 and height == .29:
                counts['stop'] += 1
            elif (vx == -1.9 or .8 <= -vx <= 1.2) and height == .29 and yaw == 0:
                counts['reverse'] += 1
                if .8 <= abs(vx) <= 1.2:
                    mid_reversals += 1
                else:
                    self.assertEqual(vx, -1.9)
                self.assertEqual(height, .29)
                self.assertEqual(yaw, 0)
            elif vx == 1.9 and height in (.22, .32):
                counts['height'] += 1
                self.assertEqual(yaw, 0)
            elif vx == 0 and yaw != 0 and height == .29:
                counts['turn'] += 1
            np.testing.assert_array_equal(e.command, e.command_target)
            self.assertLess(e.command_steps_left, 1250)
        for name, probability in zip(counts, (.20, .40, .15, .10)):
            self.assertAlmostEqual(counts[name] / 4000, probability, delta=.025)
        self.assertAlmostEqual(mid_reversals / counts['reverse'], .5, delta=.05)

    def test_reversal_progress_has_stronger_incentive_over_the_full_error_range(self):
        e, old = self.env(), self.env('joint_tracking_v13')
        p = precision_rewards_for_stage(e.stage, e.params)
        q = precision_rewards_for_stage(old.stage, old.params)
        self.assertAlmostEqual(p.vx_error_dense / p.vx_error_scale_m_s,
                               2 * q.vx_error_dense / q.vx_error_scale_m_s)
        for error in (-.05, .05):
            self.assertAlmostEqual(e._transition_terms(0, error, 0)['vx_error_dense'],
                                   4 * old._transition_terms(0, error, 0)['vx_error_dense'])
        self.assertEqual(e._transition_terms(0, .05, 0), e._transition_terms(0, -.05, 0))
        for sign in (-1, 1):
            def progress(env):
                return (env._transition_terms(0, sign*1.9, 0)['vx_error_dense']
                        - env._transition_terms(0, sign*2., 0)['vx_error_dense'])
            self.assertAlmostEqual(progress(e), 2 * progress(old))

    def test_actual_geometry_is_symmetric_and_small_persistent_errors_cost_more(self):
        e, old = self.env(), self.env('joint_tracking_v13')
        a = e._symmetry_terms([.28, .275], .01, 0, 0)
        b = e._symmetry_terms([.275, .28], 0, .01, 0)
        self.assertEqual(a, b)
        self.assertLess(a['leg_length_difference'],
                        old._symmetry_terms([.28, .275], .01, 0, 0)['leg_length_difference'])

    def test_old_checkpoint_physics_actions_and_failure_boundaries_are_preserved(self):
        from train_uz05 import write_contract, require_compatible_contract
        e, old = self.env(), self.env('joint_tracking_v13')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'policy'
            write_contract(path, 'joint_tracking_v13', 1)
            require_compatible_contract(path, 'joint_tracking_v14')
        action = np.array([.1, -.2, .9, -.8, .7, -.6])
        for _ in range(10):
            e.step(action)
            old.step(action)
            np.testing.assert_array_equal(e.sim.data.qpos, old.sim.data.qpos)
        self.assertEqual(e.stage.tilt_limit, old.stage.tilt_limit)
        self.assertEqual(e.stage.wheel_stagger_limit_m, old.stage.wheel_stagger_limit_m)
        self.assertEqual(e.stage.wheel_offset_limit_m, old.stage.wheel_offset_limit_m)
        self.assertFalse(old.stage.switching_tracking)

    def test_height_key_does_not_accidentally_brake_high_speed_motion(self):
        from uz05.replay_controls import KeyboardCommandState
        for mode in ('forward', 'reverse', 'yaw', 'yaw_right'):
            state = KeyboardCommandState(mode=mode, speed=2, height=.27)
            before = state.command()
            state.handle('height', -.01, (-2, 2), (-1.5, 1.5), (.22, .32))
            self.assertEqual(state.command()[:3], before[:3])
            self.assertAlmostEqual(state.command()[3], .26)


if __name__ == '__main__':
    unittest.main()
