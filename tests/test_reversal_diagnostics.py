import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.reversal_diagnostics import CircularTargetLimiter, command_at, replace_observed_command, wrap
from uz05.spec import ACTOR_OBS_BLOCKS, OBS_DIM, OBS_SCALE

class ReversalControlTests(unittest.TestCase):
    def test_wrap_short_arc_and_multiple_turns(self):
        initial = np.array([np.pi-.01, 7*np.pi-.01, -np.pi+.01, -7*np.pi+.01])
        limiter = CircularTargetLimiter(initial, 1., .008)
        a = np.r_[.3, -.4, [-1+.01/np.pi]*2, [1-.01/np.pi]*2]
        out, active = limiter.apply(a)
        np.testing.assert_allclose(wrap(out[2:]*np.pi-initial), [.008,.008,-.008,-.008], atol=1e-12)
        np.testing.assert_array_equal(out[:2], a[:2]); self.assertTrue(active.all())

    def test_disabled_exact_and_zero_delta(self):
        a = np.array([.123, -.7, 1, -1, .2, -.2])
        out, active = CircularTargetLimiter(np.zeros(4), None, .008).apply(a)
        np.testing.assert_array_equal(out, a); self.assertFalse(active.any())
        out, active = CircularTargetLimiter(a[2:]*np.pi, 10., .008).apply(a)
        self.assertFalse(active.any()); np.testing.assert_allclose(wrap(out[2:]*np.pi-a[2:]*np.pi),0,atol=1e-12)

    def test_command_patch_preserves_sensor_values(self):
        obs = np.arange(OBS_DIM, dtype=np.float32)/100
        cmd = np.array([2,0,0,.27,0])
        result = replace_observed_command(obs, cmd, 100.)
        index = next(i for i,(n,w) in enumerate(ACTOR_OBS_BLOCKS) if n == 'command')
        offset = sum(w for n,w in ACTOR_OBS_BLOCKS[:index])
        np.testing.assert_allclose(result[offset:offset+5], cmd*OBS_SCALE['command'])
        np.testing.assert_array_equal(result[:offset],obs[:offset]); np.testing.assert_array_equal(result[offset+5:],obs[offset+5:])

    def test_ramp_duration_and_focal_event(self):
        schedule = [(0,0,0,.27),(3,1,0,.27),(7,2,0,.27),(11,-2,0,.27),(16,0,1,.27)]
        self.assertEqual(command_at(schedule,7,.5,'motion')[1][0],2)
        self.assertEqual(command_at(schedule,11,.5,'motion')[1][0],2)
        self.assertAlmostEqual(command_at(schedule,11.25,.5,'motion')[1][0],0)
        self.assertEqual(command_at(schedule,11.5,.5,'motion')[1][0],-2)
        self.assertEqual(command_at(schedule,11,0,'motion')[1][0],-2)

if __name__ == '__main__': unittest.main()
