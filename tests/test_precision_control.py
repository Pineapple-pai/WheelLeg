"""Checks for observable stop references and physical output costs in v3."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.spec import EnvParams


class PrecisionControlTests(unittest.TestCase):
    def setUp(self):
        p = EnvParams()
        p.noise.enabled = False
        p.domain_randomization.enabled = False
        self.env = UZ05Env('joint_tracking_v3', params=p)
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def control(self, action=None):
        sim = self.env.sim
        state = {'nominal_joint_pos': np.zeros(4), 'joint_pos': sim.joint_positions(),
                 'joint_vel': sim.joint_velocities(), 'wheel_vel': sim.wheel_velocities()}
        return self.env.actuators.compute(np.zeros(6) if action is None else action, state=state, dt=.008)

    def test_stop_reference_is_captured_once_after_motion_and_observed(self):
        e = self.env
        e.command[:] = [.3, 0, 0, .27, 0]
        e._obs()
        e.sim.data.qpos[:2] += [1.0, .2]
        e.sim.forward()
        e.command[:2] = 0
        e._obs()
        reference = e.hold_xy.copy()
        np.testing.assert_allclose(reference, e.sim.base_pos[:2])
        e.sim.data.qpos[:2] += [.04, -.02]
        e.sim.forward()
        e.command[3] = .295
        e._obs()
        np.testing.assert_array_equal(e.hold_xy, reference)
        np.testing.assert_allclose(e._raw_cache['base_pos_rel'], [.04, -.02])
        self.assertLess(e._precision_terms(self.control())['hold_position'], 0)

    def test_moving_commands_do_not_receive_station_penalty(self):
        e = self.env
        e.command[:] = [.3, 0, 1., .27, 0]
        e.sim.data.qpos[0] += 5
        e.sim.forward()
        e._obs()
        terms = e._precision_terms(self.control())
        self.assertEqual(terms['hold_position'], 0)
        self.assertEqual(terms['hold_heading'], 0)

    def test_periodic_targets_do_not_create_false_output_change(self):
        e = self.env
        old = self.control()
        e._previous_control = {k: np.array(v).copy() for k, v in old.items()}
        current = {k: np.array(v).copy() for k, v in old.items()}
        current['joint_target'] += 2*np.pi
        self.assertAlmostEqual(e._precision_terms(current)['joint_target_change'], 0, places=12)
        current['wheel_target'] += 8
        self.assertLess(e._precision_terms(current)['wheel_target_change'], 0)

    def test_preclamp_demand_provides_cost_above_torque_saturation(self):
        e = self.env
        control = self.control()
        control['leg_torque_demand'][:] = 20
        control['leg_torque'][:] = 20
        cost_at_limit = e._precision_terms(control)['saturation']
        control['leg_torque_demand'][:] = 60
        self.assertLess(e._precision_terms(control)['saturation'], cost_at_limit)
        # Both requests would deliver the same clipped effort, yet receive different costs.
        np.testing.assert_array_equal(control['leg_torque'], [20]*4)

    def test_first_control_has_no_artificial_change_from_zero(self):
        terms = self.env._precision_terms(self.control())
        for key in ('wheel_target_change', 'joint_target_change', 'effort_change'):
            self.assertEqual(terms[key], 0)

    def test_v4_reduces_control_cost_without_disabling_any_objective(self):
        from uz05.spec import STAGE_BY_NAME, precision_rewards_for_stage
        e = self.env
        e.command[:] = [.3, 0, 0, .27, 0]
        e._obs()
        control = self.control()
        e._previous_control = {k: np.zeros_like(v) for k, v in control.items()}
        control['joint_target'][:] = .2
        control['wheel_target'][:] = 8
        control['leg_torque'][:] = 20
        control['leg_torque_demand'][:] = 20
        control['wheel_current'][:] = 10
        control['wheel_torque'][:] = 1.722
        v3 = e._precision_terms(control)
        e.stage = STAGE_BY_NAME['joint_tracking_v4']
        v4 = e._precision_terms(control)
        for key in ('wheel_target_change', 'joint_target_change', 'effort', 'effort_change', 'saturation'):
            self.assertLess(v4[key], 0)
            self.assertGreater(v4[key], v3[key])
        p = precision_rewards_for_stage(e.stage, e.params)
        for key in ('hold_position', 'hold_heading', 'wheel_target_change', 'joint_target_change', 'effort', 'effort_change', 'saturation'):
            self.assertGreater(getattr(p, key), 0)
        self.assertTrue(e.stage.precision_control and e.stage.stance_geometry)
        self.assertGreater(e.stage.tracking_upright_weight, STAGE_BY_NAME['joint_tracking_v3'].tracking_upright_weight)

    def test_v4_contract_records_actual_profile(self):
        import json
        from train_uz05 import write_contract
        with tempfile.TemporaryDirectory() as directory:
            contract = write_contract(Path(directory) / 'policy', 'joint_tracking_v4', 1)
            data = json.loads(contract.read_text())
            self.assertEqual(data['precision_objective']['saturation'], .03)
            self.assertEqual(data['precision_objective']['effort'], .05)
            self.assertEqual(data['stage_config']['tracking_upright_weight'], 3.)

    def test_checkpoint_reference_contract_preserves_v2_and_blocks_cross_loading(self):
        from train_uz05 import write_contract, require_compatible_contract
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy'
            write_contract(path, 'joint_tracking_v2', 1)
            require_compatible_contract(path, 'joint_tracking_v2')
            with self.assertRaises(SystemExit):
                require_compatible_contract(path, 'joint_tracking_v3')
            write_contract(path, 'joint_tracking_v3', 1)
            require_compatible_contract(path, 'joint_tracking_v3')
            with self.assertRaises(SystemExit):
                require_compatible_contract(path, 'joint_tracking_v2')


if __name__ == '__main__':
    unittest.main()
