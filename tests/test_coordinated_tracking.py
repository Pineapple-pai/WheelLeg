"""v10 soft pose coupling, reversal coverage, and warm-start compatibility."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.spec import STAGE_BY_NAME, env_params_for_stage, precision_rewards_for_stage


class CoordinatedTrackingTest(unittest.TestCase):
    def setUp(self):
        p=env_params_for_stage('joint_tracking_v10')
        p.domain_randomization.enabled=False
        p.robust.pushes_enabled=False
        self.env=UZ05Env('joint_tracking_v10',params=p)
        self.env.reset(seed=5)

    def tearDown(self):
        self.env.close()

    def test_quality_is_soft_symmetric_and_retains_recovery_signal(self):
        e=self.env
        self.assertEqual(e._tracking_pose_quality(0,0),1)
        self.assertLess(e._tracking_pose_quality(.017,0),.8)
        for h,pitch in ((.01,.05),(.04,.3),(.1,1.)):
            q=e._tracking_pose_quality(h,pitch)
            self.assertGreaterEqual(q,.3)
            self.assertLess(q,1)
            self.assertEqual(q,e._tracking_pose_quality(-h,-pitch))

    def test_wrong_height_has_pose_cost_without_weakening_speed_reward(self):
        e=self.env
        e.command[:]=[.5,0,0,.27,0]
        e.command_target[:]=e.command
        e.command_steps_left=1000
        action=np.array([.1,-.2,.9,-.8,.7,-.6])
        e._obs()
        _,_,_,_,info=e.step(action)
        height=e.sim.base_pos[2]
        state={'joint_pos':e.sim.joint_positions(),'joint_vel':e.sim.joint_velocities()}
        e.command[3]=height
        _,good=e._reward(e._previous_control,state,action,False)
        e.command[3]=height+.02
        _,bad=e._reward(e._previous_control,state,action,False)
        self.assertEqual(good['track_vx']+good['track_vx_coarse'],bad['track_vx']+bad['track_vx_coarse'])
        self.assertGreater(good['tracking_pose_cost'],bad['tracking_pose_cost'])
        self.assertEqual(good['height_error_dense'],0)
        self.assertLess(bad['height_error_dense'],0)
        np.testing.assert_allclose(e._previous_control['wheel_target'],[4.8,-9.6])
        np.testing.assert_allclose(e._previous_control['joint_target'],action[2:]*np.pi)
        self.assertTrue(.3<=info['tracking_pose_quality']<=1)

    def test_fixed_direct_reversals_preserve_height_and_have_short_holds(self):
        e=self.env;e.steps=100
        reversals=0
        for _ in range(2000):
            e.command_target[:]=[.9,0,0,.27,0]
            e.command[:]=e.command_target
            e._sample_command()
            if e.command[0]==-.9:
                reversals+=1
                np.testing.assert_array_equal(e.command,[-.9,0,0,.27,0])
                self.assertLess(e.command_steps_left,500)
        self.assertTrue(.20<reversals/2000<.30)

    def test_v9_warm_start_is_compatible_and_old_horizon_is_preserved(self):
        from train_uz05 import write_contract,require_compatible_contract
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'policy'
            write_contract(p,'joint_tracking_v9',1)
            require_compatible_contract(p,'joint_tracking_v10')
        old=STAGE_BY_NAME['joint_tracking_v9']
        new=self.env.stage
        self.assertEqual(old.ppo_gamma,.99)
        self.assertEqual(old.ppo_gae_lambda,.95)
        self.assertGreater(new.ppo_gamma**500,10*old.ppo_gamma**500)
        self.assertEqual(new.wheel_velocity_scale_rad_s,old.wheel_velocity_scale_rad_s)
        self.assertEqual(precision_rewards_for_stage(old,env_params_for_stage(old)).vx_error_dense,.8)


if __name__=='__main__':
    unittest.main()
