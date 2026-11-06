"""v12 fixed moving states, reversal coverage and unchanged six-action control."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.env import UZ05Env
from uz05.spec import STAGE_BY_NAME,env_params_for_stage


class ResponsiveTrackingTest(unittest.TestCase):
    def setUp(self):
        p=env_params_for_stage('joint_tracking_v12')
        p.domain_randomization.enabled=False;p.robust.pushes_enabled=False
        self.env=UZ05Env('joint_tracking_v12',params=p)

    def tearDown(self):self.env.close()

    def test_moving_starts_are_fixed_and_have_consistent_body_and_wheel_velocities(self):
        moving=[]
        for seed in range(500):
            self.env.reset(seed=seed)
            vx=self.env._initial_rolling_vx
            if vx:
                moving.append(vx)
                self.assertAlmostEqual(self.env.sim.body_frame(self.env.sim.base_lin_vel_world)[0],vx)
                np.testing.assert_allclose(self.env.sim.wheel_velocities(),-vx/.055)
                self.assertTrue(.4<=abs(vx)<=2)
        self.assertTrue(.28<len(moving)/500<.42)
        self.assertTrue(.48<np.mean(np.array(moving)<0)<.72)
        self.assertGreater(max(moving),1.9)
        self.assertLess(min(moving),-1.9)

    def test_stationary_evaluation_can_disable_only_the_moving_start_mixture(self):
        self.env.stage=replace(self.env.stage,moving_start_probability=0)
        for seed in range(20):
            self.env.reset(seed=seed)
            self.assertEqual(self.env._initial_rolling_vx,0)
            self.assertLess(np.linalg.norm(self.env.sim.base_lin_vel_world),.14)
        self.assertEqual(self.env.stage.vx_range,(-2,2))

    def test_high_speed_and_reverse_commands_are_present_from_initial_sampling(self):
        self.env.reset(seed=123)
        self.env.steps=0;vx=[]
        for _ in range(3000):
            self.env._sample_command();vx.append(self.env.command[0])
        moving=np.array([x for x in vx if x!=0])
        self.assertTrue(.54<np.mean(moving<0)<.66)
        self.assertTrue(.24<np.mean(np.abs(moving)>1.5)<.36)
        self.assertGreater(moving.max(),1.95)
        self.assertLess(moving.min(),-1.95)

    def test_direct_reversal_holds_allow_complete_braking_without_ramping_command(self):
        e=self.env;e.reset(seed=4);e.steps=100;reversals=0
        for _ in range(1000):
            e.command_target[:]=[1.7,0,0,.27,0]
            e.command[:]=e.command_target
            e._sample_command()
            if e.command[0]==-1.7:
                reversals+=1
                np.testing.assert_array_equal(e.command,[-1.7,0,0,.27,0])
                self.assertTrue(250<=e.command_steps_left<625)
        self.assertTrue(.29<reversals/1000<.41)

    def test_straight_heading_cost_is_stronger_but_intentional_turns_are_free(self):
        e=self.env;e.reset(seed=2)
        e.command[:]=[.5,0,0,.27,0];e._obs()
        e.straight_yaw-=np.radians(5)
        cost=e._straight_terms(0,0,0,0)['straight_heading']
        self.assertLess(cost,-1)
        e.command[0]=-.5
        self.assertAlmostEqual(e._straight_terms(0,0,0,0)['straight_heading'],cost)
        e.command[2]=1
        self.assertEqual(e._straight_terms(0,0,0,0)['straight_heading'],0)

    def test_old_stage_and_action_contract_are_preserved(self):
        from train_uz05 import write_contract,require_compatible_contract
        old=STAGE_BY_NAME['joint_tracking_v11']
        self.assertEqual(old.moving_start_probability,0)
        self.assertEqual(old.velocity_band_probabilities,(.25,.4,.35))
        self.assertEqual(old.command_hold_steps,(125,500))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'policy';write_contract(p,'joint_tracking_v11',1)
            require_compatible_contract(p,'joint_tracking_v12')
        e=self.env;e.reset(seed=12)
        e.step(np.array([.1,-.2,.9,-.8,.7,-.6]))
        np.testing.assert_allclose(e._previous_control['wheel_target'],[4.8,-9.6])
        self.assertEqual(e.params.joint.torque_limit,20)


if __name__=='__main__':unittest.main()
