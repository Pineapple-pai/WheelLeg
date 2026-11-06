"""v9 heading observability, tracking incentives, and fixed command coverage."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage, precision_rewards_for_stage


class StraightTrackingTest(unittest.TestCase):
    def setUp(self):
        params = env_params_for_stage('joint_tracking_v9')
        params.domain_randomization.enabled = False
        params.robust.pushes_enabled = False
        self.env = UZ05Env('joint_tracking_v9', params=params)
        self.env.reset(seed=5)

    def tearDown(self):
        self.env.close()

    def yaw(self, angle):
        quat = Rotation.from_euler('z', angle).as_quat()
        self.env.sim.data.qpos[3:7] = quat[[3, 0, 1, 2]]
        self.env.sim.forward()

    def command(self, vx, yaw=0, height=.27):
        self.env.command[:] = [vx, 0, yaw, height, 0]
        self.env.command_target[:] = self.env.command
        self.env.command_steps_left = 1000
        self.env._obs()

    def test_straight_drift_is_observable_and_not_recaptured_every_frame(self):
        self.command(0)
        self.yaw(.1)
        self.command(.7)
        self.yaw(.3)
        for _ in range(5):
            self.env._obs()
        self.assertAlmostEqual(self.env._heading_error(), .2)
        np.testing.assert_allclose(self.env._raw_cache['heading_error_sincos'], [np.sin(.2), np.cos(.2)])
        self.command(-.7, height=.32)
        self.assertAlmostEqual(self.env._heading_error(), .2)
        terms = self.env._straight_terms(0, 0, 0, 0)
        self.assertLess(terms['straight_heading'], 0)

    def test_turn_and_stop_use_their_own_heading_reference(self):
        self.command(0)
        self.yaw(3.1)
        self.command(.5)
        self.yaw(-3.1)
        self.env._obs()
        self.assertAlmostEqual(self.env._heading_error(), 2*np.pi-6.2)
        self.command(.5, yaw=1)
        self.assertEqual(self.env._heading_error(), 0)
        self.assertEqual(self.env._straight_terms(0, 0, 0, 0)['straight_heading'], 0)
        self.yaw(.8)
        self.command(-.5)
        self.assertAlmostEqual(self.env._heading_error(), 0)
        self.command(0)
        self.yaw(.9)
        self.env._obs()
        self.assertAlmostEqual(self.env._heading_error(), .1)

    def test_large_velocity_error_still_rewards_progress_in_both_directions(self):
        e = self.env
        p = precision_rewards_for_stage(e.stage, e.params)
        def score(error):
            return (e.stage.tracking_vx_weight*np.exp(-(error/e.stage.tracking_vx_sigma)**2)
                    + e._straight_terms(error,0,0,0)['track_vx_coarse']
                    + e._transition_terms(0,error,0)['vx_error_dense'] - .5*error**2)
        for sign in (-1, 1):
            self.assertGreater(score(sign*.6)-score(sign*.7), .15)
            self.assertGreater(score(sign*1.9)-score(sign*2), .15)
        self.assertGreater(score(0), score(.15))
        self.assertEqual(score(.4), score(-.4))
        self.assertGreater(p.pitch_error,0)

    def test_motion_bounce_cost_is_soft_and_does_not_prescribe_joint_angles(self):
        self.command(.5)
        zero = self.env._straight_terms(0,0,0,0)
        positive = self.env._straight_terms(0,0,.15,.2)
        negative = self.env._straight_terms(0,0,-.15,-.2)
        self.assertEqual(zero['motion_vertical_velocity'], 0)
        self.assertEqual(positive, negative)
        self.assertLess(positive['leg_common_velocity'], 0)
        action = np.array([.1,-.2,.9,-.8,.7,-.6])
        _,_,_,_,info = self.env.step(action)
        np.testing.assert_allclose(self.env._previous_control['wheel_target'], [4.8,-9.6])
        np.testing.assert_allclose(self.env._previous_control['joint_target'], action[2:]*np.pi)
        self.assertIn('straight_heading',info['reward_terms'])

    def test_fixed_distribution_covers_mid_speed_high_speed_and_both_signs(self):
        commands=[]
        self.env.steps=0
        for _ in range(2000):
            self.env._sample_command()
            commands.append(self.env.command.copy())
        commands=np.asarray(commands)
        straight=(commands[:,0]!=0)&(commands[:,2]==0)
        self.assertTrue(.38 < straight.mean() < .52)
        vx=commands[commands[:,0]!=0,0]
        self.assertTrue(.4 < np.mean(vx>0) < .6)
        self.assertGreater(np.mean(np.abs(vx)>1),.25)
        self.assertGreater(np.mean((np.abs(vx)>=.4)&(np.abs(vx)<=1)),.30)
        self.assertLess(commands[:,3].min(),.225)
        self.assertGreater(commands[:,3].max(),.315)

    def test_coarse_tracking_cannot_bypass_zero_command_position_quality(self):
        e=self.env
        self.command(0)
        _,_,_,_,info=e.step(np.zeros(6))
        before=info['reward_terms']['track_vx_coarse']
        e.sim.data.qpos[0]+=.3
        e.sim.forward()
        e._obs()
        _,terms=e._reward(e._previous_control,
                         {'joint_pos':e.sim.joint_positions(), 'joint_vel':e.sim.joint_velocities()},
                         np.zeros(6),False)
        self.assertGreater(before,0)
        self.assertLess(terms['track_vx_coarse'],before*.1)

    def test_v8_contract_cannot_silently_use_new_heading_observation(self):
        from train_uz05 import write_contract,require_compatible_contract
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy'
            write_contract(path,'joint_tracking_v8',1)
            with self.assertRaises(SystemExit):
                require_compatible_contract(path,'joint_tracking_v9')


if __name__ == '__main__':
    unittest.main()
