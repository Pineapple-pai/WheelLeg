"""Closed-chain resets must preserve sampled active states and assembly branch."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import itertools

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.env import UZ05Env
from uz05.spec import env_params_for_stage
from uz05.reset_projection import ProjectionError
from uz05 import model as model_module


class ResetProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = env_params_for_stage('joint_tracking_v16')
        cls.fixed = UZ05Env('joint_tracking_v16', params=p)
        p = env_params_for_stage('joint_tracking_v16')
        p.robot.closed_chain_reset = False
        cls.legacy = UZ05Env('joint_tracking_v16', params=p)

    @classmethod
    def tearDownClass(cls):
        cls.fixed.close()
        cls.legacy.close()

    def test_100_paired_resets_preserve_all_nonpassive_coordinates_and_randomization(self):
        f, old = self.fixed, self.legacy
        qp, qv = f.sim.reset_projector.qpos, f.sim.reset_projector.dofs
        for seed in range(100):
            f.reset(seed=seed); old.reset(seed=seed)
            if f.reset_projection_info['resamples'] == 0:
                np.testing.assert_array_equal(np.delete(f.sim.data.qpos, qp), np.delete(old.sim.data.qpos, qp))
                np.testing.assert_array_equal(np.delete(f.sim.data.qvel, qv), np.delete(old.sim.data.qvel, qv))
                np.testing.assert_array_equal(f.command, old.command)
            else:
                self.assertIn(seed, (28, 60, 72, 82))
                self.assertTrue(all('penetration' in r for r in f.reset_projection_info['rejections']))
            self.assertEqual(f.domain, old.domain)
            self.assertLess(f.sim.reset_projector.errors(f.sim.data).max(), 1e-8)
            v = f.sim.reset_projector.jacobian(f.sim.data) @ f.sim.data.qvel
            self.assertLess(np.max(np.abs(v)), 1e-8)
            self.assertTrue(np.all(np.sign(f.sim.reset_projector._branch_determinants(
                f.sim.data)) == f.sim.reset_projector.reference_orientation))
            self.assertTrue(all(c.dist >= -1e-5 for c in f.sim.data.contact))

    def test_projection_itself_never_changes_prescribed_coordinates_even_on_rejected_draws(self):
        s = self.legacy.sim
        for seed in range(100):
            self.legacy.reset(seed=seed)
            q, v = s.data.qpos.copy(), s.data.qvel.copy()
            s.reset_projector.project(s.data)
            np.testing.assert_array_equal(np.delete(q, s.reset_projector.qpos),
                                          np.delete(s.data.qpos, s.reset_projector.qpos))
            np.testing.assert_array_equal(np.delete(v, s.reset_projector.dofs),
                                          np.delete(s.data.qvel, s.reset_projector.dofs))

    def test_legacy_switch_reproduces_measured_error(self):
        errors = []
        for seed in range(100):
            self.legacy.reset(seed=seed)
            errors.append(self.legacy.sim.reset_projector.errors(self.legacy.sim.data).max()*1000)
        np.testing.assert_allclose(np.percentile(errors, [50, 95, 100]),
                                   [4.23192523, 7.82848151, 9.16169689], atol=1e-7)

    def test_zero_reference_is_not_displaced(self):
        s = self.fixed.sim
        s.reset(.315)
        q, v = s.data.qpos.copy(), s.data.qvel.copy()
        result = s.reset_projector.project(s.data)
        np.testing.assert_array_equal(s.data.qpos, q)
        np.testing.assert_array_equal(s.data.qvel, v)
        self.assertEqual(result['iterations'], 0)

    def test_all_active_sampling_box_corners_close_on_reference_branch(self):
        s = self.fixed.sim
        for angles in itertools.product((-.05, .05), repeat=4):
            s.reset(.315)
            s.data.qpos[s.hip_qpos_adr] = angles
            result = s.reset_projector.project(s.data)
            self.assertLess(result['max_error_m'], 1e-8)
            np.testing.assert_array_equal(s.data.qpos[s.hip_qpos_adr], angles)

    def test_nonfinite_and_wrong_branch_are_rejected(self):
        s = self.fixed.sim
        s.reset(.315)
        s.data.qvel[s.hip_dof_adr[0]] = np.nan
        with self.assertRaisesRegex(ProjectionError, 'nonfinite'):
            s.reset_projector.project(s.data)
        s.reset(.315)
        with patch.object(s.reset_projector, '_branch_determinants',
                          return_value=-s.reset_projector.reference_orientation):
            with self.assertRaisesRegex(ProjectionError, 'branch orientation'):
                s.reset_projector.project(s.data)

    def test_site_jacobian_uses_dof_indices_and_matches_finite_difference(self):
        s = self.fixed.sim
        self.fixed.reset(seed=12)
        q = s.data.qpos.copy()
        analytic = s.reset_projector.jacobian(s.data)[:, s.reset_projector.dofs]
        columns = []
        for adr in s.reset_projector.qpos:
            s.data.qpos[:] = q
            s.data.qpos[adr] += 1e-7
            s.forward()
            plus = s.reset_projector.residual(s.data).ravel()
            s.data.qpos[adr] -= 2e-7
            s.forward()
            minus = s.reset_projector.residual(s.data).ravel()
            columns.append((plus-minus)/2e-7)
        s.data.qpos[:] = q; s.forward()
        np.testing.assert_allclose(analytic, np.array(columns).T, atol=1e-8)

    def test_solver_failure_is_bounded_and_explicit(self):
        with patch.object(self.fixed.sim.reset_projector, 'project', side_effect=ProjectionError('forced')) as solve:
            with self.assertRaisesRegex(RuntimeError, 'failed after 8 attempts.*forced'):
                self.fixed.reset(seed=7)
            self.assertEqual(solve.call_count, 8)

    def test_retry_is_counted(self):
        solve = self.fixed.sim.reset_projector.project
        calls = 0
        def once(data):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ProjectionError('forced retry')
            return solve(data)
        with patch.object(self.fixed.sim.reset_projector, 'project', side_effect=once):
            self.fixed.reset(seed=7)
            self.assertEqual(self.fixed.reset_projection_info['resamples'], 1)

    def test_actuator_xml_declaration_order_does_not_change_physics(self):
        build = model_module._build_xml
        def reordered(*args, **kwargs):
            path = build(*args, **kwargs)
            tree = ET.parse(path)
            block = tree.getroot().find('actuator')
            block[:] = list(reversed(list(block)))
            tree.write(path)
            return path
        with patch.object(model_module, '_build_xml', side_effect=reordered):
            other = UZ05Env('joint_tracking_v16')
        try:
            self.fixed.reset(seed=10); other.reset(seed=10)
            action = np.array([.1, -.1, .01, -.02, .03, -.01])
            self.fixed.step(action); other.step(action)
            np.testing.assert_allclose(self.fixed.sim.data.qpos, other.sim.data.qpos, atol=1e-12)
            np.testing.assert_allclose(self.fixed.sim.data.qvel, other.sim.data.qvel, atol=1e-12)
        finally:
            other.close()


if __name__ == '__main__':
    unittest.main()
