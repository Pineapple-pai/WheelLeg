"""Common-mode leg swing cannot hide behind a symmetric support stance."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage, precision_rewards_for_stage
from eval_joint_tracking_transitions import support_response


class CenteredSupportTest(unittest.TestCase):
    def env(self, stage='joint_tracking_v15'):
        p = env_params_for_stage(stage)
        p.domain_randomization.enabled = False
        p.robust.pushes_enabled = False
        e = UZ05Env(stage, params=p)
        e.reset(seed=8)
        self.addCleanup(e.close)
        return e

    def test_symmetric_common_swing_and_opposite_swing_cannot_cancel(self):
        e = self.env()
        def cost(left, right):
            return e._centered_support_terms(dict(wheel_fore_aft_left_m=left,
                                                 wheel_fore_aft_right_m=right))['wheel_fore_aft_alignment']
        for offset in (.06, .15, .228):
            self.assertLess(cost(offset, offset), 0)
            self.assertEqual(cost(offset, offset), cost(-offset, -offset))
            self.assertEqual(cost(offset, offset), cost(offset, -offset))
            self.assertAlmostEqual(cost(offset, offset), 2*cost(offset, 0))
        self.assertEqual(cost(.03, -.03), 0)
        self.assertLess(cost(.228, .228), cost(.15, .15))
        # The previous symmetry objective sees no difference in either pose.
        self.assertEqual(e._symmetry_terms([.25,.25], .15,.15,0),
                         e._symmetry_terms([.25,.25], 0,0,0))

    def test_reward_is_applied_during_motion_and_stand_without_overriding_actions(self):
        e = self.env()
        action = np.array([.1,-.2,.9,-.8,.7,-.6])
        for vx in (0.,1.,-1.):
            e.command[:] = [vx,0,0,.27,0]
            e.command_target[:] = e.command
            e.command_steps_left = 100
            _,_,_,_,info = e.step(action)
            expected = e._centered_support_terms(info)['wheel_fore_aft_alignment']
            self.assertEqual(info['reward_terms']['wheel_fore_aft_alignment'], expected)
            np.testing.assert_allclose(e._previous_control['wheel_target'], [4.8,-9.6])
            np.testing.assert_allclose(e._previous_control['joint_target'], action[2:]*np.pi)

    def test_v14_checkpoint_physics_speed_objective_and_limits_are_preserved(self):
        from train_uz05 import write_contract, require_compatible_contract
        old, e = self.env('joint_tracking_v14'), self.env()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'policy'
            write_contract(path,'joint_tracking_v14',1)
            require_compatible_contract(path,'joint_tracking_v15')
        action = np.array([.1,-.2,.9,-.8,.7,-.6])
        for _ in range(10):
            a = old.step(action)
            b = e.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(old.sim.data.qpos, e.sim.data.qpos)
            self.assertEqual(a[2:4], b[2:4])
        self.assertNotIn('wheel_fore_aft_alignment',a[4]['reward_terms'])
        p = precision_rewards_for_stage(e.stage, e.params)
        q = precision_rewards_for_stage(old.stage, old.params)
        self.assertEqual(p.vx_error_dense, q.vx_error_dense)
        self.assertEqual(p.vx_error_scale_m_s, q.vx_error_scale_m_s)
        self.assertEqual(e.stage.tilt_limit, old.stage.tilt_limit)
        self.assertEqual(e.stage.wheel_offset_limit_m, old.stage.wheel_offset_limit_m)

    def test_duration_and_recovery_require_both_legs_to_return(self):
        offsets = np.r_[np.full((50,2), .2), np.zeros((50,2))]
        r = support_response(offsets, np.zeros((100,4)), .008)
        self.assertEqual(r['wheel_fore_aft_outside_band_fraction'], .5)
        self.assertAlmostEqual(r['wheel_fore_aft_longest_outside_band_s'], .4)
        self.assertAlmostEqual(r['wheel_fore_aft_recovery_after_peak_s'], .408)
        # Alternating which leg is outside must never count as recovery.
        offsets[50:,0] = np.where(np.arange(50)%2, .2, 0)
        offsets[50:,1] = np.where(np.arange(50)%2, 0, .2)
        r = support_response(offsets, np.zeros((100,4)), .008)
        self.assertIsNone(r['wheel_fore_aft_recovery_after_peak_s'])


if __name__ == '__main__':
    unittest.main()
