"""Stopping reward remains useful after finite braking distance; no action helper."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage, precision_rewards_for_stage

class BrakingTrackingTest(unittest.TestCase):
    def make_env(self, stage='joint_tracking_v13'):
        p=env_params_for_stage(stage);p.domain_randomization.enabled=False;p.robust.pushes_enabled=False
        e=UZ05Env(stage,params=p);e.reset(seed=8);self.addCleanup(e.close)
        return e

    def test_stopping_keeps_speed_reward_after_braking_displacement(self):
        old=self.make_env('joint_tracking_v12');new=self.make_env()
        for e in (old,new):
            e.command[:]=[0,0,0,.27,0];e.command_target[:]=e.command
            e.command_steps_left=100;e._obs()
            e.hold_xy=e.sim.base_pos[:2].copy()-[.7,0]
            e.step(np.zeros(6))
        np.testing.assert_array_equal(old.sim.data.qpos,new.sim.data.qpos)
        self.assertLess(old._last_reward_terms['track_vx'],1e-10)
        self.assertGreater(new._last_reward_terms['track_vx'],1)
        self.assertGreaterEqual(new._last_reward_terms['hold_position'],-4)
        self.assertLess(new._last_reward_terms['hold_position'],0)
        self.assertLess(old._last_reward_terms['hold_position'],-10)
        np.testing.assert_array_equal(new.hold_xy,old.hold_xy)

    def test_stop_and_reversal_are_explicit_full_steps_with_same_height(self):
        e=self.make_env();e.steps=100
        counts=[0,0]
        for _ in range(2000):
            e.command[:]=[1.9,0,0,.29,0];e.command_target[:]=e.command
            e._sample_command()
            if e.command[0]==0 and e.command[2]==0 and e.command[3]==.29:counts[0]+=1
            if e.command[0]==-1.9:
                counts[1]+=1
                np.testing.assert_array_equal(e.command,[-1.9,0,0,.29,0])
        self.assertTrue(.23<counts[0]/2000<.37)
        self.assertTrue(.31<counts[1]/2000<.39)

    def test_soft_geometry_and_speed_costs_are_symmetric(self):
        e=self.make_env();old=self.make_env('joint_tracking_v12')
        a=e._symmetry_terms([.28,.27],.01,0,0)
        b=e._symmetry_terms([.27,.28],0,.01,0)
        self.assertEqual(a,b)
        self.assertLess(a['leg_length_difference'],old._symmetry_terms([.28,.27],.01,0,0)['leg_length_difference'])
        costs=[e._transition_terms(0,v,0)['vx_error_dense'] for v in (-.3,.3)]
        self.assertEqual(costs[0],costs[1]);self.assertLess(costs[0],-.5)
        p=precision_rewards_for_stage(e.stage,e.params)
        self.assertEqual(p.hold_position_cost_cap,4)
        self.assertEqual(p.pitch_error,old.params.precision_responsive.pitch_error)

    def test_policy_contract_and_motor_targets_are_unchanged(self):
        from train_uz05 import write_contract,require_compatible_contract
        e=self.make_env();old=self.make_env('joint_tracking_v12')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'policy';write_contract(p,'joint_tracking_v12',1)
            require_compatible_contract(p,'joint_tracking_v13')
        action=np.array([.1,-.2,.9,-.8,.7,-.6])
        for _ in range(10):
            e.step(action);old.step(action)
            np.testing.assert_array_equal(e.sim.data.qpos,old.sim.data.qpos)
        np.testing.assert_allclose(e._previous_control['wheel_target'],[4.8,-9.6])
        self.assertEqual(e.stage.vx_range,(-2,2))
        self.assertEqual(e.params.joint.torque_limit,20)
        self.assertFalse(old.stage.braking_tracking)

if __name__=='__main__':unittest.main()
