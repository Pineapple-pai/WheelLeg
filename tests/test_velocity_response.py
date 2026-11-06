"""A reversal can satisfy T90 while still missing the requested speed."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from eval_joint_tracking_transitions import velocity_response, settling_time


class VelocityResponseTest(unittest.TestCase):
    def test_t90_does_not_imply_tight_tracking(self):
        vx = np.r_[np.linspace(1, -.85, 100), np.full(200, -.85)]
        result = velocity_response(vx, -1, 1, .008)
        self.assertIsNotNone(settling_time(vx, -1, 1, .008, .05))
        self.assertIsNone(result['vx_settling_005_m_s_s'])
        self.assertGreater(result['vx_zero_crossing_s'], 0)
        self.assertGreater(result['vx_reverse_acceleration_to_t90_s'], 0)
        mirrored = velocity_response(-vx, 1, -1, .008)
        for key in ('vx_zero_crossing_s', 'vx_reverse_acceleration_to_t90_s',
                    'vx_settling_005_m_s_s', 'vx_integral_absolute_error_m'):
            self.assertEqual(result[key], mirrored[key])

    def test_failure_before_crossing_has_no_response_times(self):
        result = velocity_response([.9, .8, .7], -1, 1, .008)
        self.assertIsNone(result['vx_zero_crossing_s'])
        self.assertIsNone(result['vx_reverse_acceleration_to_t90_s'])
        self.assertIsNone(result['vx_settling_005_m_s_s'])
        self.assertEqual(result['vx_at_elapsed_m_s'], {})


if __name__ == '__main__':
    unittest.main()
