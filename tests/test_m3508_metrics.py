import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.m3508_metrics import score_interval,longest_run
from run_m3508_validation import THRESHOLDS


def series(vx,end=8):
    t=np.arange(1,round(end/.008)+1)*.008;n=len(t)
    z=np.zeros(n);return {'time_s':t,'vx_m_s':np.full(n,vx),
        'command_vx_m_s':np.full(n,2.),'pitch_rad':z.copy(),'roll_rad':z.copy(),
        'height_error_m':z.copy(),'normal_force_n_0':np.full(n,80.),'normal_force_n_1':np.full(n,80.),
        'common_x_m':z.copy(),'differential_x_m':z.copy(),'pitch_rate_rad_s':z.copy()}

class MotorMetricTests(unittest.TestCase):
    def test_survival_cannot_replace_tracking_and_no_sign_cancellation(self):
        c=series(0.)
        r=score_interval(c,3,8,2,THRESHOLDS,'time_limit')
        self.assertEqual(r['posture_success'],1);self.assertEqual(r['tracking_success'],0)
        c=series(2.);c['vx_m_s'][::2]=1.8;c['vx_m_s'][1::2]=2.2
        r=score_interval(c,3,8,2,THRESHOLDS,'time_limit')
        self.assertAlmostEqual(r['steady_bias_m_s'],0)
        self.assertEqual(r['tracking_success'],0)

    def test_boundary_allowed_5percent_and_full_window(self):
        r=score_interval(series(1.9),3,8,2,THRESHOLDS,'time_limit')
        self.assertEqual(r['composite_success'],1);self.assertAlmostEqual(r['response_time_s'],1.)
        self.assertEqual(r['steady_1p90_to_1p95'],1)

    def test_brief_target_crossing_not_success(self):
        c=series(1.5);c['vx_m_s'][-10:]=2.
        self.assertEqual(score_interval(c,3,8,2,THRESHOLDS,'time_limit')['tracking_success'],0)

    def test_early_death_cannot_shrink_scored_window(self):
        r=score_interval(series(2.,4),3,8,2,THRESHOLDS,'tilt_limit')
        self.assertEqual(r['interval_complete'],0);self.assertEqual(r['posture_success'],0)
        self.assertEqual(r['tracking_success'],0)

    def test_zero_speed_absolute_band_and_posture_independent(self):
        c=series(.04);c['pitch_rad'][:]=np.radians(20)
        r=score_interval(c,3,8,0,THRESHOLDS,'time_limit')
        self.assertEqual(r['tracking_success'],1);self.assertEqual(r['posture_success'],0)
        self.assertEqual(r['composite_success'],0);self.assertEqual(r['speed_band_m_s'],.05)

    def test_longest_run_boundary(self):
        self.assertAlmostEqual(longest_run([True,True,False,True]),.016)
        self.assertEqual(longest_run([]),0)

if __name__=='__main__':unittest.main()
