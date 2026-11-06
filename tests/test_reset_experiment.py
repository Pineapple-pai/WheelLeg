"""Experiment logging must leave the actual environment trajectory unchanged."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.env import UZ05Env
from uz05.experiment_audit import AuditedResetEnv, fingerprint
from uz05.spec import env_params_for_stage


class ExperimentAuditTests(unittest.TestCase):
    def test_logging_does_not_change_rng_observations_or_dynamics(self):
        with tempfile.TemporaryDirectory() as directory:
            for projected in (False, True):
                p = env_params_for_stage('joint_tracking_v16')
                p.robot.closed_chain_reset = projected
                raw = UZ05Env('joint_tracking_v16', params=p)
                p = env_params_for_stage('joint_tracking_v16')
                p.robot.closed_chain_reset = projected
                audit = AuditedResetEnv('joint_tracking_v16', params=p,
                                        audit_path=Path(directory)/f'{projected}.csv')
                try:
                    for seed in (1, 28, 60):
                        a, _ = raw.reset(seed=seed); b, _ = audit.reset(seed=seed)
                        np.testing.assert_array_equal(a, b)
                        for _ in range(4):
                            action = np.array([.1, -.1, .01, -.02, .03, -.04])
                            a = raw.step(action); b = audit.step(action)
                            np.testing.assert_array_equal(a[0], b[0])
                            self.assertEqual(a[1:4], b[1:4])
                            np.testing.assert_array_equal(raw.sim.data.qpos, audit.sim.data.qpos)
                finally:
                    raw.close(); audit.close()
                self.assertEqual(len((Path(directory)/f'{projected}.csv').read_text().splitlines()), 4)

    def test_fingerprint_detects_optimizer_moment_change(self):
        a = {'state': {0: {'step': 12., 'exp_avg': np.array([.1, .2])}}}
        b = {'state': {0: {'step': 12., 'exp_avg': np.array([.1, .21])}}}
        self.assertNotEqual(fingerprint(a), fingerprint(b))


if __name__ == '__main__': unittest.main()
