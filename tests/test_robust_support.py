"""v5 observations, real ground support, disturbances and randomization checks."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.model import UZ05Model
from uz05.spec import env_params_for_stage


class RobustSupportTests(unittest.TestCase):
    def setUp(self):
        p = env_params_for_stage('joint_tracking_v5')
        p.domain_randomization.enabled = False
        p.robust.pushes_enabled = False
        p.noise.enabled = False
        self.env = UZ05Env('joint_tracking_v5', params=p)
        self.env.reset(seed=5)

    def tearDown(self):
        self.env.close()

    def test_moving_task_does_not_observe_accumulated_position_or_heading(self):
        e = self.env
        e.command[:] = [.3, 0, 1, .27, 0]
        e._obs()
        e.sim.data.qpos[:2] += [20, -10]
        q = Rotation.from_euler('z', 1.4).as_quat()
        e.sim.data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        e.sim.forward()
        e._obs()
        np.testing.assert_allclose(e._raw_cache['base_pos_rel'], 0)
        np.testing.assert_allclose(e._raw_cache['heading_error_sincos'], [0, 1])
        e.command[:3] = 0
        e._obs()
        reference = e.hold_xy.copy()
        e.sim.data.qpos[0] += .03
        e.sim.forward()
        e._obs()
        np.testing.assert_allclose(e._raw_cache['base_pos_rel'], [.03, 0])
        np.testing.assert_array_equal(e.hold_xy, reference)

    def test_command_sampler_contains_long_and_short_motion_from_start(self):
        short = long = zeros = 0
        values = []
        for _ in range(200):
            self.env._sample_command()
            values.append(self.env.command.copy())
            if not self.env.command_active:
                zeros += 1
                self.assertGreaterEqual(self.env.command_steps_left, 1500)
            elif self.env.command_steps_left >= 1500:
                long += 1
            else:
                short += 1
        self.assertTrue(short and long and zeros)
        values = np.array(values)
        for axis in (0, 2):
            self.assertLess(values[:, axis].min(), -.1)
            self.assertGreater(values[:, axis].max(), .1)
        self.assertGreater(np.ptp(values[:, 3]), .04)

    def test_randomized_gains_do_not_compound_across_resets(self):
        e = self.env
        e.params.domain_randomization.enabled = True
        for seed in range(30):
            e.reset(seed=seed)
            self.assertEqual(e.params.joint.kp, 100)
            self.assertAlmostEqual(e.actuators.joint.params.kp, 100*e.domain['joint_kp_scale'][0])
            self.assertAlmostEqual(e.actuators.wheel.params.torque_per_amp_joint,
                                   .246*e.domain['wheel_torque_scale'][0])
        e.params.domain_randomization.enabled = False
        e.reset(seed=5)
        self.assertEqual(e.actuators.joint.params.kp, 100)
        self.assertEqual(e.actuators.wheel.params.torque_per_amp_joint, .246)

    def test_reset_wheels_do_not_start_deep_inside_ground(self):
        for seed in range(30):
            e = self.env
            e.reset(seed=seed)
            for gid in e.sim.wheel_geoms:
                axis = e.sim.data.geom_xmat[gid].reshape(3, 3)[:, 2]
                radius, half = e.sim.model.geom_size[gid, :2]
                extent = radius*np.sqrt(max(0, 1-axis[2]**2)) + half*abs(axis[2])
                self.assertGreaterEqual(e.sim.data.geom_xpos[gid, 2] - extent, -1e-5)

    def test_external_push_is_applied_temporarily_during_real_physics(self):
        e = self.env
        e.params.robust.pushes_enabled = True
        e._next_push_step = 0
        e.sim.data.xfrc_applied[e.sim.chassis_body, :3] = [1, 2, 0]
        original = e.sim.step
        applied = []
        def step():
            applied.append(e.sim.data.xfrc_applied[e.sim.chassis_body, :3].copy())
            original()
        with patch.object(e.sim, 'step', side_effect=step):
            e.step(np.zeros(6))
        self.assertGreaterEqual(np.linalg.norm(e._push_force), 5)
        self.assertLessEqual(np.linalg.norm(e._push_force), 15)
        self.assertEqual(e._push_force[2], 0)
        self.assertEqual(len(applied), e.params.robot.control_substeps)
        for force in applied:
            np.testing.assert_allclose(force, np.array([1, 2, 0]) + e._push_force)
        np.testing.assert_allclose(e.sim.data.xfrc_applied[e.sim.chassis_body, :3], [1, 2, 0])

    def test_motor_feedback_updates_each_physics_substep(self):
        e = self.env
        action = np.array([.1, -.1, .02, -.02, .02, -.02])
        with patch.object(e.actuators, 'compute', wraps=e.actuators.compute) as compute:
            _, _, _, _, info = e.step(action)
        self.assertEqual(compute.call_count, 4)
        for call in compute.call_args_list:
            self.assertAlmostEqual(call.kwargs['dt'], .002)
            np.testing.assert_array_equal(call.args[0], action)
        samples = e._previous_control['motor_leg_torque_samples']
        self.assertEqual(samples.shape, (4, 4))
        np.testing.assert_allclose(info['joint_saturation_per_joint'], np.mean(np.abs(samples) >= 19.8, axis=0))

    def test_motor_cost_does_not_hide_substep_saturation_in_zero_mean(self):
        e = self.env
        state = {'nominal_joint_pos': np.zeros(4), 'joint_pos': e.sim.joint_positions(),
                 'joint_vel': e.sim.joint_velocities(), 'wheel_vel': e.sim.wheel_velocities()}
        control = e.actuators.compute(np.zeros(6), state=state, dt=.002)
        control['leg_torque'][:] = 0
        control['motor_leg_torque_samples'] = np.array([[20]*4, [-20]*4, [0]*4, [0]*4])
        control['motor_leg_demand_samples'] = control['motor_leg_torque_samples'].copy()
        control['motor_wheel_current_samples'] = np.zeros((4, 2))
        self.assertEqual(e._output_metrics(control)['joint_saturation_fraction'], .5)
        self.assertLess(e._precision_terms(control)['effort'], 0)
        self.assertLess(e._precision_terms(control)['saturation'], 0)

    def test_v5_reference_and_motor_contract_rejects_v4_checkpoint(self):
        import tempfile
        import json
        from train_uz05 import write_contract, require_compatible_contract
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy'
            write_contract(path, 'joint_tracking_v4', 1)
            with self.assertRaises(SystemExit):
                require_compatible_contract(path, 'joint_tracking_v5')
            contract = write_contract(path, 'joint_tracking_v5', 1)
            data = json.loads(contract.read_text())
            self.assertEqual(data['actuator_config']['actuator_dt_s'], .002)
            self.assertEqual(data['observation_reference'], 'stop_anchor_else_zero')
            self.assertEqual(data['environment_config']['domain_randomization']['actuator_delay_steps'], [0, 0])
            require_compatible_contract(path, 'joint_tracking_v5')

    def test_ground_contact_filter_excludes_wheel_link_collision(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <geom name="floor" type="plane" size="2 2 .1" pos="0 0 -10"/>
        <body><freejoint/><geom name="wheel" type="sphere" size=".1" mass="1"/></body>
        <body pos=".15 0 0"><geom name="leg" type="sphere" size=".1"/></body>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        sim = SimpleNamespace(model=model, data=data, wheel_geoms=[model.geom('wheel').id],
                              body_geoms=[], _contact_force=np.zeros(6))
        self.assertEqual(UZ05Model.contact_state(sim)[1][0], 1)
        self.assertEqual(UZ05Model.contact_state(sim, terrain_only=True)[1][0], 0)

    def test_support_objectives_penalize_actual_differential_motion(self):
        e = self.env
        e.command[:] = [0, 0, 0, .27, 0]
        e._obs()
        state = {'nominal_joint_pos': np.zeros(4), 'joint_pos': e.sim.joint_positions(),
                 'joint_vel': e.sim.joint_velocities(), 'wheel_vel': e.sim.wheel_velocities()}
        control = e.actuators.compute(np.zeros(6), state=state, dt=.008)
        e._support_diag['leg_drop_difference_m'] = .03
        e._support_diag['leg_differential_velocity_m_s'] = .3
        with patch.object(e.sim, 'contact_state', return_value=(np.array([100, 0]), np.array([1, 0]), 0., 0.)):
            _, terms = e._reward(control, state, np.zeros(6), False)
        for key in ('leg_drop_difference', 'leg_differential_velocity', 'wheel_support'):
            self.assertLess(terms[key], 0)


if __name__ == '__main__':
    unittest.main()
