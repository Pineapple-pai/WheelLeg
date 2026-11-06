import sys
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.actuators import WheelEscController
from uz05.spec import WheelEscParams
from uz05.m3508_diagnostics import DiagnosticWheelEsc,reference_at

class M3508DiagnosticTests(unittest.TestCase):
    def test_legacy_exact_including_integral_and_reset(self):
        old=WheelEscController();new=DiagnosticWheelEsc(WheelEscParams(),'A_legacy')
        rng=np.random.default_rng(13)
        for _ in range(30):
            goal=rng.uniform(-70,70,2);actual=rng.uniform(-70,70,2)
            a=old.current(goal,actual,.002);b=new.current(goal,actual,.002)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(old.torque_from_current(a),new.torque_from_current(b))
            np.testing.assert_array_equal(old.integral,new.integral)
        old.reset();new.reset();np.testing.assert_array_equal(old.integral,new.integral)

    def test_rated_peak_and_symmetry(self):
        c=DiagnosticWheelEsc(WheelEscParams(),'B_rated')
        np.testing.assert_allclose(c.torque_from_current(np.array([-10.,0.,10.])),[-2.46,0,2.46])
        self.assertEqual(c.params.efficiency,.7);self.assertEqual(c.params.current_limit_a,10)

    def test_burst_expiry_and_no_retrigger_for_reference_ticks(self):
        c=DiagnosticWheelEsc(WheelEscParams(),'C_burst')
        self.assertEqual(c.effective_cap_nm,2.46)
        c.begin_control(3.,-2.);self.assertEqual(c.effective_cap_nm,3.)
        c.begin_control(3.2,-2.);c.time_s=3.498;self.assertEqual(c.effective_cap_nm,3.)
        c.time_s=3.5;self.assertEqual(c.effective_cap_nm,2.46)
        c.begin_control(10.,0.);self.assertEqual(c.effective_cap_nm,3.)
        c.reset();self.assertEqual(c.effective_cap_nm,2.46)

    def test_gains_current_limits_and_xml_conversion_unchanged(self):
        base=WheelEscParams()
        for name,cap in [('A_legacy',1.722),('B_rated',2.46),('C_burst',3.),('D_stall_bound',3.69)]:
            c=DiagnosticWheelEsc(deepcopy(base),name);c.begin_control(0.,2.)
            current=c.current(np.array([100.,-100.]),np.zeros(2),.002)
            np.testing.assert_array_equal(current,[10.,-10.])
            np.testing.assert_allclose(c.torque_from_current(current),[cap,-cap])
            np.testing.assert_allclose(c.motor_command(np.array([cap,-cap]))*base.gear_ratio,[cap,-cap])
            self.assertEqual(c.params,base)

    def test_acceleration_reference_bound_and_signed_transition(self):
        schedule=[(0,2,0,.27),(11,-2,0,.27)]
        previous=2.
        for i in range(400):
            phase,cmd,goal=reference_at(schedule,11+i*.008,'accel_2.0',previous,.008,'motion')
            self.assertLessEqual(abs(cmd[0]-previous),.016+1e-12)
            self.assertLessEqual(cmd[0],previous+1e-12)
            previous=cmd[0]
        self.assertEqual(previous,-2.);self.assertEqual(goal,-2.)

    def test_ramp_focal_only_and_endpoint(self):
        schedule=[(0,0,0,.27),(3,1,0,.27),(7,2,0,.27),(11,-2,0,.27)]
        self.assertEqual(reference_at(schedule,7,'ramp_2.0',1,.008,'motion')[1][0],2.)
        self.assertEqual(reference_at(schedule,11,'ramp_2.0',2,.008,'motion')[1][0],2.)
        self.assertEqual(reference_at(schedule,12,'ramp_2.0',0,.008,'motion')[1][0],0.)
        self.assertEqual(reference_at(schedule,13,'ramp_2.0',-2,.008,'motion')[1][0],-2.)

if __name__=='__main__':unittest.main()
