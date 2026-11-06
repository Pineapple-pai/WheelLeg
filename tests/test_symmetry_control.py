"""v7 geometry soft costs must not constrain the six independent actions."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage, precision_rewards_for_stage, STAGE_BY_NAME


class SymmetryControlTests(unittest.TestCase):
    def setUp(self):
        params = env_params_for_stage('joint_tracking_v7')
        params.domain_randomization.enabled = False
        params.robust.pushes_enabled = False
        self.env = UZ05Env('joint_tracking_v7', params=params)
        self.env.reset(seed=5)

    def tearDown(self):
        self.env.close()

    def test_actual_length_and_fore_aft_are_separate_costs(self):
        e = self.env
        x = e._symmetry_terms([.23,.25],-.02,-.02,0)
        self.assertLess(x['leg_length_difference'],0)
        self.assertEqual(x['fore_aft_difference'],0)
        x = e._symmetry_terms([.24,.24],-.04,0,0)
        self.assertEqual(x['leg_length_difference'],0)
        self.assertLess(x['fore_aft_difference'],0)

    def test_small_differences_have_tolerance(self):
        x=self.env._symmetry_terms([.240,.242],-.002,.002,np.radians(.8))
        self.assertTrue(all(v==0 for v in x.values()))

    def test_swapping_sides_and_roll_sign_does_not_change_cost(self):
        e=self.env
        a=e._symmetry_terms([.23,.25],-.04,-.005,np.radians(3))
        b=e._symmetry_terms([.25,.23],-.005,-.04,np.radians(-3))
        self.assertEqual(a,b)

    def test_large_errors_remain_finite_and_cost_grows_linearly(self):
        e=self.env
        costs=[e._symmetry_terms([.24+d,.24],0,0,0)['leg_length_difference'] for d in (.033,.043,.053)]
        self.assertAlmostEqual(costs[1]-costs[0],costs[2]-costs[1])
        self.assertLess(costs[2],costs[1])
        self.assertTrue(np.isfinite(list(e._symmetry_terms([.1,.4],-.3,.3,np.radians(45)).values())).all())

    def test_full_reward_uses_geometry_for_stand_motion_and_yaw(self):
        e=self.env
        for command in ([0,0,0,.27,0],[.3,0,0,.27,0],[0,0,1,.27,0]):
            e.reset(seed=5)
            e.command[:]=command;e.command_target[:]=command;e.command_steps_left=3001
            _,_,_,_,info=e.step(np.zeros(6))
            expected=e._symmetry_terms(e.sim.leg_lengths(),info['wheel_fore_aft_left_m'],info['wheel_fore_aft_right_m'],info['roll'])
            for k,v in expected.items():self.assertAlmostEqual(info['reward_terms'][k],v)

    def test_independent_positions_and_velocity_targets_are_not_overridden(self):
        e=self.env
        action=np.array([.13,-.21,.04,-.03,.01,-.05])
        e.step(action)
        c=e._previous_control
        np.testing.assert_allclose(c['wheel_target'],action[:2]*16)
        np.testing.assert_allclose(np.arctan2(np.sin(c['joint_target']),np.cos(c['joint_target'])),action[2:]*np.pi)

    def test_v6_profile_is_preserved_and_pitch_budget_unchanged(self):
        old=precision_rewards_for_stage(STAGE_BY_NAME['joint_tracking_v6'],env_params_for_stage('joint_tracking_v6'))
        new=precision_rewards_for_stage(self.env.stage,self.env.params)
        self.assertAlmostEqual(old.roll_sigma_rad,np.radians(12))
        self.assertAlmostEqual(old.leg_drop_deadband_m,.01)
        self.assertEqual(old.pitch_sigma_rad,new.pitch_sigma_rad)
        self.assertAlmostEqual(sum(self.env._pitch_terms(0,0).values()),3)
        self.assertTrue(self.env.stage.long_command_prob==.5)


if __name__=='__main__':
    unittest.main()
