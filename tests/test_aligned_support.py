"""Physical lean sign, true axle alignment, and circular joint target diagnostics."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.model import wheel_axis_metrics, support_posture_metrics
from uz05.env import UZ05Env
from uz05.replay_viewer import telemetry_lines


class AlignedSupportTest(unittest.TestCase):
    def env(self, stage='joint_tracking_v16'):
        e = UZ05Env(stage)
        e.params.domain_randomization.enabled = False
        e.params.robust.pushes_enabled = False
        e.reset(seed=8)
        self.addCleanup(e.close)
        return e

    def test_positive_pitch_is_forward_and_follows_total_com_lean(self):
        wheels = np.array([[0.,.2,.055],[0.,-.2,.055]])
        for angle in (-.1,.1):
            rotation = Rotation.from_euler('y',angle).as_matrix()
            com = np.array([.4*np.tan(angle),0,.4])
            good = support_posture_metrics(com,wheels,rotation,.055)
            self.assertAlmostEqual(good['chassis_lean_rad'],angle)
            self.assertAlmostEqual(good['support_com_lean_rad'],angle)
            self.assertAlmostEqual(good['support_pitch_error_rad'],0)
            bad = support_posture_metrics(com,wheels,Rotation.from_euler('y',-angle).as_matrix(),.055)
            self.assertAlmostEqual(bad['support_pitch_error_rad'],-2*angle)
            yaw = Rotation.from_euler('z',1.2).as_matrix()
            shift = np.array([3.,-2.,.3])
            moved = support_posture_metrics(yaw@com+shift,wheels@yaw.T+shift,yaw@rotation,.055)
            np.testing.assert_allclose(list(moved.values()),list(good.values()),atol=1e-12)

    def test_parallel_axes_can_still_be_noncoaxial(self):
        centers = np.array([[0.,.2,.055],[.027,-.2,.055]])
        axes = np.array([[0.,1.,0.],[0.,-1.,0.]])
        r = wheel_axis_metrics(centers,axes)
        self.assertAlmostEqual(r['axle_line_offset_m'],.027)
        self.assertEqual(r['axle_parallel_error_deg'],0)
        rotation = Rotation.from_euler('xyz',[.2,.3,.5]).as_matrix()
        moved = wheel_axis_metrics(centers@rotation.T+[1.,2.,3.],axes@rotation.T)
        self.assertAlmostEqual(moved['axle_line_offset_m'],.027)
        centers[1,0] = 0
        self.assertAlmostEqual(wheel_axis_metrics(centers,axes)['axle_line_offset_m'],0)

    def test_paired_target_cost_is_circular_soft_and_turning_is_not_forced_equal(self):
        e = self.env()
        e.command[:] = [1.,0,0,.27,0]
        def terms(target):
            return e._aligned_support_terms({'axle_line_offset_m':.027}, {'joint_target':np.array(target)})
        zero = terms([0.,0.,2*np.pi,-2*np.pi])
        self.assertAlmostEqual(zero['paired_joint_target_alignment'],0)
        different = terms([.2,-.3,-.2,.3])
        mirrored = terms([-.2,.3,.2,-.3])
        self.assertEqual(different,mirrored)
        self.assertLess(different['paired_joint_target_alignment'],0)
        self.assertLess(different['wheel_axle_alignment'],0)
        e.command[2] = 1.
        turn = terms([.2,-.3,-.2,.3])
        self.assertEqual(turn['paired_joint_target_alignment'],0)
        self.assertEqual(turn['wheel_axle_alignment'],different['wheel_axle_alignment'])

    def test_reward_uses_physical_support_error_instead_of_zero_pitch(self):
        e = self.env()
        e.command[:] = [1,0,0,.27,0]
        e.command_target[:] = e.command
        e.command_steps_left = 100
        _,_,_,_,info = e.step(np.array([.1,-.2,.9,-.8,.7,-.6]))
        expected = e._pitch_terms(info['roll'],info['support_pitch_error_rad'])
        for key,value in expected.items():
            self.assertAlmostEqual(info['reward_terms'][key],value)
        self.assertAlmostEqual(e._tracking_quality,
                               e._tracking_pose_quality(info['tracking_height_error'],info['support_pitch_error_rad']))
        self.assertIn('wheel_axle_alignment',info['reward_terms'])

    def test_v15_physics_interface_and_independent_actions_are_preserved(self):
        from train_uz05 import write_contract, require_compatible_contract
        e, old = self.env(),self.env('joint_tracking_v15')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'policy'
            write_contract(path,'joint_tracking_v15',1)
            require_compatible_contract(path,'joint_tracking_v16')
        action = np.array([.1,-.2,.9,-.8,.7,-.6])
        for _ in range(10):
            a,b = e.step(action),old.step(action)
            np.testing.assert_array_equal(a[0],b[0])
            np.testing.assert_array_equal(e.sim.data.qpos,old.sim.data.qpos)
            self.assertEqual(a[2:4],b[2:4])
        np.testing.assert_allclose(e._previous_control['joint_target'],action[2:]*np.pi)
        self.assertNotIn('wheel_axle_alignment',b[4]['reward_terms'])
        self.assertEqual(e.stage.tilt_limit,old.stage.tilt_limit)
        self.assertEqual(e.stage.wheel_offset_limit_m,old.stage.wheel_offset_limit_m)
        self.assertEqual(e.params.precision_aligned.vx_error_dense,old.params.precision_centered.vx_error_dense)

    def test_replay_distinguishes_pitch_lean_axle_offset_and_targets_without_stepping(self):
        e = self.env()
        e.step(np.zeros(6))
        qpos,qvel,t = e.sim.data.qpos.copy(),e.sim.data.qvel.copy(),e.sim.data.time
        lines = telemetry_lines(e,'stand')
        for label in ('Pitch','COM lean','Axle offset','Joint target deg'):
            self.assertIn(label,'\n'.join(lines))
        np.testing.assert_array_equal(e.sim.data.qpos,qpos)
        np.testing.assert_array_equal(e.sim.data.qvel,qvel)
        self.assertEqual(e.sim.data.time,t)


if __name__ == '__main__':
    unittest.main()
