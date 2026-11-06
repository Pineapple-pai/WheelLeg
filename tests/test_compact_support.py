"""Physical rod reach and soft v11 posture objectives without action overrides."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage


class CompactSupportTest(unittest.TestCase):
    def setUp(self):
        p=env_params_for_stage('joint_tracking_v11')
        p.domain_randomization.enabled=False;p.robust.pushes_enabled=False
        self.env=UZ05Env('joint_tracking_v11',params=p)
        self.env.reset(seed=5)

    def tearDown(self):
        self.env.close()

    def test_physical_rod_reach_detects_protrusion_with_unchanged_wheel_centers(self):
        sim=self.env.sim
        before=sim.leg_envelope();wheel_before=sim.wheel_stance()
        idx=sim.leg_capsules[0][0]
        sim.data.geom_xpos[idx]+=sim.rotation_matrix()[:,0]*.4
        after=sim.leg_envelope()
        self.assertGreater(after['leg_forward_reach_left_m'],before['leg_forward_reach_left_m']+.1)
        self.assertEqual(sim.wheel_stance(),wheel_before)

    def test_reach_is_invariant_to_world_pose_and_full_joint_turns(self):
        sim=self.env.sim
        before=sim.leg_envelope()
        sim.data.qpos[:3]+=[2,-1,.3]
        quat=Rotation.from_euler('xyz',[.2,-.3,1.2]).as_quat()
        sim.data.qpos[3:7]=quat[[3,0,1,2]]
        sim.data.qpos[sim.hip_qpos_adr]+=2*np.pi
        sim.forward()
        np.testing.assert_allclose(list(sim.leg_envelope().values()),list(before.values()),atol=1e-12)

    def test_reach_cost_is_soft_finite_and_independent_for_both_legs(self):
        e=self.env
        normal={f'leg_{direction}_reach_{side}_m':limit for direction,limit in (('forward',.1),('backward',.255)) for side in ('left','right')}
        self.assertTrue(all(value==0 for value in e._leg_reach_terms(normal).values()))
        normal['leg_forward_reach_left_m']=.24
        cost=e._leg_reach_terms(normal)
        self.assertLess(cost['leg_forward_reach'],0)
        self.assertEqual(cost['leg_backward_reach'],0)
        normal['leg_forward_reach_left_m']=1.
        self.assertTrue(np.isfinite(e._leg_reach_terms(normal)['leg_forward_reach']))
        self.assertLess(e._leg_reach_terms(normal)['leg_forward_reach'],cost['leg_forward_reach'])

    def test_pitch_and_actual_length_soft_costs_are_symmetric_and_stronger(self):
        e=self.env
        forward=e._pitch_terms(.01,np.radians(4.25))
        reverse=e._pitch_terms(-.01,-np.radians(4.25))
        self.assertEqual(forward,reverse)
        self.assertLess(forward['pitch_error'],-.4)
        a=e._symmetry_terms([.22,.211],.02,0,.015)
        b=e._symmetry_terms([.211,.22],0,.02,-.015)
        self.assertEqual(a,b)
        self.assertLess(a['leg_length_difference'],-.4)
        old=env_params_for_stage('joint_tracking_v10').precision_coordinated
        self.assertEqual(old.pitch_deadband_rad,np.radians(3))
        self.assertEqual(old.leg_length_difference,.35)

    def test_reward_profile_changes_do_not_change_observations_or_independent_actions(self):
        oldp=env_params_for_stage('joint_tracking_v10')
        oldp.domain_randomization.enabled=False;oldp.robust.pushes_enabled=False
        old=UZ05Env('joint_tracking_v10',params=oldp)
        try:
            oldobs,_=old.reset(seed=12)
            newobs,_=self.env.reset(seed=12)
            np.testing.assert_array_equal(oldobs,newobs)
            action=np.array([.1,-.2,.9,-.8,.7,-.6])
            for _ in range(10):
                oldobs,_,_,_,_=old.step(action)
                newobs,_,_,_,info=self.env.step(action)
                np.testing.assert_array_equal(oldobs,newobs)
                np.testing.assert_array_equal(old.sim.data.qpos,self.env.sim.data.qpos)
            np.testing.assert_allclose(self.env._previous_control['wheel_target'],[4.8,-9.6])
            self.assertIn('leg_forward_reach',info['reward_terms'])
        finally:old.close()

    def test_warm_start_and_high_speed_limits_are_preserved(self):
        from train_uz05 import write_contract,require_compatible_contract
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'policy';write_contract(p,'joint_tracking_v10',1)
            require_compatible_contract(p,'joint_tracking_v11')
        self.assertEqual(self.env.stage.vx_range,(-2,2))
        self.assertEqual(self.env.stage.wheel_velocity_scale_rad_s,48)
        self.assertEqual(self.env.params.joint.torque_limit,20)
        self.assertEqual(self.env.stage.ppo_gamma,.997)


if __name__=='__main__':unittest.main()
