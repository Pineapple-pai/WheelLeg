import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.wheel_speed_diagnostics import SpeedLoopProbe,contact_diagnostics
from uz05.m3508_diagnostics import DiagnosticWheelEsc
from uz05.spec import WheelEscParams,env_params_for_stage
from uz05.env import UZ05Env

class WheelSpeedDiagnosticTests(unittest.TestCase):
    def test_original_p_exact(self):
        for model in ['A_legacy','B_rated']:
            old=DiagnosticWheelEsc(WheelEscParams(),model);new=SpeedLoopProbe(WheelEscParams(),model=model)
            rng=np.random.default_rng(4)
            for _ in range(30):
                target=rng.uniform(-48,48,2);actual=rng.uniform(-55,55,2)
                a=old.current(target,actual,.002);b=new.current(target,actual,.002)
                np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(old.integral,new.integral)
                np.testing.assert_array_equal(old.torque_from_current(a),new.torque_from_current(b))

    def test_antiwindup_blocks_outward_saturation_and_resets(self):
        c=SpeedLoopProbe(WheelEscParams(),'PI')
        for _ in range(1000):
            current=c.current(np.array([48.,-48.]),np.zeros(2),.002)
            np.testing.assert_array_equal(current,[10.,-10.])
        np.testing.assert_array_equal(c.integral_current,[0.,0.]);self.assertTrue(c.antiwindup_blocked.all())
        for _ in range(1000):c.current(np.array([1.,-1.]),np.zeros(2),.002)
        self.assertGreater(c.integral_current[0],0);self.assertLess(c.integral_current[1],0)
        c.reset();np.testing.assert_array_equal(c.integral_current,[0.,0.])

    def test_integral_clamp_and_unwind(self):
        c=SpeedLoopProbe(WheelEscParams(),'PI')
        for _ in range(5000):c.current(np.array([2.,-2.]),np.zeros(2),.002)
        np.testing.assert_allclose(c.integral_current,[4.,-4.])
        old=c.integral_current.copy();c.current(np.array([-2.,2.]),np.zeros(2),.002)
        self.assertLess(c.integral_current[0],old[0]);self.assertGreater(c.integral_current[1],old[1])
        self.assertLessEqual(np.max(np.abs(c.raw_current)),10.)

    def test_ff_balance_and_shared_limits(self):
        c=SpeedLoopProbe(WheelEscParams(),'P_FF')
        current=c.current(np.array([36.36,-36.36]),np.array([36.36,-36.36]),.002)
        np.testing.assert_allclose(c.torque_from_current(current),.02*np.array([36.36,-36.36]))
        current=c.current(np.array([48.,-48.]),np.array([-48.,48.]),.002)
        np.testing.assert_array_equal(current,[10.,-10.])
        self.assertLessEqual(abs(c.torque_from_current(current)).max(),2.46)

    def test_contact_identity_and_passive_loss_actual_model(self):
        p=env_params_for_stage('joint_tracking_v16');p.domain_randomization.enabled=False;p.robust.pushes_enabled=False
        env=UZ05Env('joint_tracking_v16',params=p);env.reset(seed=51000);seen=False
        try:
            for _ in range(65):
                _,_,terminated,truncated,_=env.step(np.zeros(6))
                d=contact_diagnostics(env.sim)
                for side in range(2):
                    self.assertAlmostEqual(d[f'wheel_passive_nm_{side}'],d[f'wheel_viscous_predicted_nm_{side}'],places=9)
                    if np.isfinite(d[f'kinematic_identity_error_{side}']):
                        seen=True;self.assertLess(abs(d[f'kinematic_identity_error_{side}']),1e-9)
                if terminated or truncated:break
            self.assertTrue(seen)
        finally:env.close()

if __name__=='__main__':unittest.main()
