"""Single-factor wheel-map contract and randomized-reset regression tests."""
import json
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_uz05 import make_env,actuator_contract,write_contract,require_compatible_contract
from uz05.m3508_diagnostics import DiagnosticWheelEsc

class TorqueMapTests(unittest.TestCase):
    def test_contract_requires_explicit_adaptation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint'
            write_contract(p,'joint_tracking_v16',1,wheel_torque_model='B_rated')
            require_compatible_contract(p,'joint_tracking_v16','B_rated')
            with self.assertRaises(SystemExit):require_compatible_contract(p,'joint_tracking_v16')
            require_compatible_contract(p,'joint_tracking_v16','A_legacy',True)
            f=Path(str(p)+'_contract.json');c=json.loads(f.read_text());c['actuator_config']['joint']['kp']=101
            f.write_text(json.dumps(c))
            with self.assertRaises(SystemExit):require_compatible_contract(p,'joint_tracking_v16','A_legacy',True)

    def test_nominal_formal_and_diagnostic_B_trajectories_match(self):
        a=make_env('joint_tracking_v16',0,31,0,domain_randomization=False,wheel_torque_model='B_rated')()
        b=make_env('joint_tracking_v16',0,31,0,domain_randomization=False,wheel_torque_model='B_rated')()
        b.actuators.wheel=DiagnosticWheelEsc(b.actuators.wheel.params,'B_rated')
        try:
            np.testing.assert_array_equal(a.reset(seed=31)[0],b.reset(seed=31)[0])
            rng=np.random.default_rng(7)
            for _ in range(30):
                action=rng.uniform(-.01,.01,6)
                x=a.step(action);y=b.step(action)
                np.testing.assert_allclose(a.sim.data.qpos,b.sim.data.qpos,rtol=0,atol=1e-10)
                np.testing.assert_allclose(a.sim.data.qvel,b.sim.data.qvel,rtol=0,atol=1e-9)
                self.assertEqual(x[2:4],y[2:4])
                if x[2] or x[3]:break
        finally:a.close();b.close()

    def test_reset_mapping_and_randomization_only_difference(self):
        a=make_env('joint_tracking_v16',0,11,0)()
        b=make_env('joint_tracking_v16',0,11,0,wheel_torque_model='B_rated')()
        try:
            for seed in range(5):
                oa,_=a.reset(seed=seed);ob,_=b.reset(seed=seed)
                np.testing.assert_array_equal(oa,ob)
                np.testing.assert_array_equal(a.sim.model.dof_damping,b.sim.model.dof_damping)
                ca=vars(a.params.wheel).copy();cb=vars(b.params.wheel).copy()
                self.assertEqual(ca.pop('efficiency'),.7);self.assertEqual(cb.pop('efficiency'),1.)
                self.assertEqual(ca,cb)
                pa=vars(a.actuators.wheel.params).copy();pb=vars(b.actuators.wheel.params).copy()
                pa.pop('efficiency');pb.pop('efficiency');self.assertEqual(pa,pb)
                currents=np.linspace(-10,10,22).reshape(11,2)
                for current in currents:
                    diag=DiagnosticWheelEsc(b.actuators.wheel.params,'B_rated')
                    np.testing.assert_array_equal(b.actuators.wheel.torque_from_current(current),np.clip(current*b.actuators.wheel.params.torque_per_amp_joint,-2.46,2.46))
                    # The diagnostic B uses nominal .246; training retains the unchanged random torque scale.
                    if not b.params.domain_randomization.enabled:
                        np.testing.assert_allclose(b.actuators.wheel.torque_from_current(current),diag.torque_from_current(current),atol=1e-15)
            c=actuator_contract('joint_tracking_v16');d=actuator_contract('joint_tracking_v16','B_rated')
            d['wheel']['efficiency']=.7;self.assertEqual(c,d)
        finally:a.close();b.close()
if __name__=='__main__':unittest.main()
