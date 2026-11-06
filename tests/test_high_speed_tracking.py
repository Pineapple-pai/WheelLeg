"""Physical action capacity, fixed full-range commands and dense transition costs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage,action_spec_for_stage,STAGE_BY_NAME


class HighSpeedTrackingTests(unittest.TestCase):
    def setUp(self):
        p=env_params_for_stage('joint_tracking_v8')
        p.domain_randomization.enabled=False;p.robust.pushes_enabled=False
        self.env=UZ05Env('joint_tracking_v8',params=p);self.env.reset(seed=5)

    def tearDown(self):
        self.env.close()

    def test_wheel_targets_cover_two_m_s_with_turning_margin(self):
        e=self.env
        e.step(np.array([1,-1,0,0,0,0]))
        np.testing.assert_array_equal(e._previous_control['wheel_target'],[48,-48])
        required=(2+.2103*1.5)/e.params.robot.wheel_radius
        self.assertLess(required,48)
        self.assertLess(48,e.params.wheel.velocity_limit)
        self.assertEqual(e.params.wheel.current_limit_a,10)
        self.assertEqual(e.params.joint.torque_limit,20)
        self.assertEqual(action_spec_for_stage('joint_tracking_v7')[0][2],16)

    def test_all_speed_bands_and_height_endpoints_are_present_from_start(self):
        e=self.env;vx=[];height=[];holds=[]
        for _ in range(1500):
            e._sample_command();vx.append(e.command[0]);height.append(e.command[3]);holds.append(e.command_steps_left)
        vx=np.asarray(vx)
        for lo,hi in ((.01,.5),(.5,1.25),(1.25,2.0)):
            for sign in (-1,1):self.assertGreater(np.sum((vx*sign>lo)&(vx*sign<=hi)),40)
        self.assertLess(min(height),.225);self.assertGreater(max(height),.315)
        self.assertTrue(any(h<500 for h in holds) and any(h>=1500 for h in holds))

    def test_motion_only_and_height_only_changes_both_exist(self):
        e=self.env;e.steps=100;motion_only=height_only=0
        for _ in range(500):
            prev=e.command_target.copy();e._sample_command()
            same_motion=np.array_equal(prev[:3],e.command_target[:3])
            same_height=prev[3]==e.command_target[3]
            motion_only+=int(same_height and not same_motion)
            height_only+=int(same_motion and not same_height)
        self.assertGreater(motion_only,50);self.assertGreater(height_only,30)

    def test_transition_costs_are_symmetric_dense_and_have_no_sag_cost_above_target(self):
        e=self.env
        z=e._transition_terms(0,0,0)
        self.assertTrue(all(v==0 for v in z.values()))
        self.assertEqual(e._transition_terms(.05,0,0)['height_shortfall'],0)
        self.assertLess(e._transition_terms(-.05,0,0)['height_shortfall'],-1)
        self.assertLess(e._transition_terms(0,2,0)['vx_error_dense'],e._transition_terms(0,1,0)['vx_error_dense'])
        self.assertEqual(e._transition_terms(0,2,3),e._transition_terms(0,-2,-3))
        _,_,_,_,info=e.step(np.zeros(6))
        for key in z:self.assertIn(key,info['reward_terms'])

    def test_scale_change_blocks_silent_old_checkpoint_loading(self):
        from train_uz05 import write_contract,require_compatible_contract
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'policy'
            write_contract(p,'joint_tracking_v7',1)
            with self.assertRaises(SystemExit):require_compatible_contract(p,'joint_tracking_v8')
            c=write_contract(p,'joint_tracking_v8',1)
            d=json.loads(c.read_text())
            self.assertEqual(d['action_blocks'][0]['scale'],48)
            self.assertEqual(d['actuator_config']['wheel_velocity_scale_rad_s'],48)
            require_compatible_contract(p,'joint_tracking_v8')

    def test_initial_physical_wheel_exploration_matches_previous_stage(self):
        s=STAGE_BY_NAME['joint_tracking_v8']
        self.assertAlmostEqual(s.wheel_action_std_init*s.wheel_velocity_scale_rad_s,.06*16)

    def test_response_metric_requires_sustained_tracking(self):
        from eval_joint_tracking_transitions import settling_time
        self.assertIsNone(settling_time([0,1,0,1],1,0,.1,.05))
        self.assertAlmostEqual(settling_time([0,.91,.95,1],1,0,.1,.05),.2)
        self.assertAlmostEqual(settling_time([.30,.316,.318,.32],.32,.27,.1,.005),.2)


if __name__=='__main__':unittest.main()
