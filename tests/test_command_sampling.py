"""Sampling arms must preserve plant/reward/controller and reset semantics."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_uz05 import make_env
from uz05.env import UZ05Env
from uz05.spec import precision_rewards_for_stage

class CommandSamplingTests(unittest.TestCase):
    def test_original_factory_preserves_trajectory(self):
        a=make_env('joint_tracking_v16',0,37,0)()
        b=UZ05Env('joint_tracking_v16',seed=37,stand_level=0)
        try:
            np.testing.assert_array_equal(a.reset(seed=37)[0],b.reset(seed=37)[0])
            rng=np.random.default_rng(8)
            for _ in range(40):
                action=rng.uniform(-.01,.01,6)
                x=a.step(action);y=b.step(action)
                np.testing.assert_array_equal(x[0],y[0])
                self.assertEqual(x[1:4],y[1:4])
                if x[2] or x[3]:break
        finally:a.close();b.close()

    def test_low_band_only_changes_nonzero_translation_on_reset(self):
        a=make_env('joint_tracking_v16',0,0,0)()
        b=make_env('joint_tracking_v16',0,0,0,command_sampling='low_band')()
        try:
            for seed in range(30):
                a.reset(seed=seed);b.reset(seed=seed)
                np.testing.assert_array_equal(a.sim.data.qpos,b.sim.data.qpos)
                np.testing.assert_array_equal(a.sim.data.qvel,b.sim.data.qvel)
                self.assertEqual(a.domain,b.domain)
                np.testing.assert_array_equal(a.command[1:],b.command[1:])
                if abs(a.command[0])>1e-12:
                    self.assertAlmostEqual(b.command[0],np.sign(a.command[0])*(.25+.25*np.clip(abs(a.command[0])/2,0,1)))
                else:self.assertEqual(b.command[0],0)
        finally:a.close();b.close()

    def test_launch_schedule_and_unchanged_parameters(self):
        arms=[make_env('joint_tracking_v16',0,31,0,command_sampling=k)()
              for k in ('original','low_band','launch_focus')]
        try:
            for e in arms:
                self.assertEqual(asdict(e.params),asdict(arms[0].params))
                self.assertEqual(asdict(precision_rewards_for_stage(e.stage,e.params)),
                                 asdict(precision_rewards_for_stage(arms[0].stage,arms[0].params)))
                # Reset sampling consumes different RNG draws in launch_focus;
                # compare nominal physics, not independently drawn DR samples.
                np.testing.assert_array_equal(e._nominal_model['damping'],arms[0]._nominal_model['damping'])
                np.testing.assert_array_equal(e.sim.model.dof_armature,arms[0].sim.model.dof_armature)
            e=arms[-1]
            self.assertEqual(e.stage.moving_start_probability,0)
            for seed in range(10):
                e.reset(seed=seed);self.assertEqual(e.command[0],0)
                e.steps=375;e._sample_command()
                self.assertIn(e.command[0],[-.5,-.25,.25,.5])
                np.testing.assert_array_equal(e.command[1:],[0,0,.27,0])
        finally:
            for e in arms:e.close()

if __name__=='__main__':unittest.main()
