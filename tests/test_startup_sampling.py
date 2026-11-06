"""Foundation sampling isolates tasks without constraining PPO actions."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_uz05 import make_env, environment_contract
from uz05.command_sampling import sampling_contract
from uz05.spec import STAGE_BY_NAME, env_params_for_stage, precision_rewards_for_stage

class StartupSamplingTests(unittest.TestCase):
    def test_stratified_workers_cover_each_speed_without_command_remapping(self):
        from collections import Counter
        from uz05.command_sampling import STRATIFIED_GOALS
        envs=[make_env('joint_tracking_startup_precision_fine',rank,17,0,
                      command_sampling='startup_stratified')() for rank in range(8)]
        try:
            for episode in range(8):
                for env in envs:
                    env.reset(seed=100+episode)
                    self.assertEqual(env.command[0],0.)
                    env.steps=375;env._sample_command()
                    self.assertEqual(env.command[0],env._launch_goal)
                    self.assertEqual(env.command_target[0],env._launch_goal)
                self.assertEqual(Counter(env.command[0] for env in envs),Counter(STRATIFIED_GOALS))
        finally:
            for env in envs:env.close()

    def test_mixed_ramp_schedule_and_same_prelaunch_state(self):
        a=make_env('joint_tracking_startup_precision',0,0,0,command_sampling='startup_balanced')()
        b=make_env('joint_tracking_startup_precision',0,0,0,command_sampling='startup_mixed_ramp')()
        try:
            for seed in range(8):
                np.testing.assert_array_equal(a.reset(seed=seed)[0],b.reset(seed=seed)[0])
                self.assertEqual(a._launch_goal,b._launch_goal)
            for goal in (-.5,-.25,0.,.25,.5):
                for ramp in (False,True):
                    b.reset(seed=4);b._launch_goal=goal;b._launch_ramp=ramp
                    b.command_steps_left=0;previous=0.
                    for step in range(375,651):
                        b.steps=step;b._advance_command()
                        expected=np.sign(goal)*min(abs(goal),.25*(step-375)*.008) if ramp else goal
                        self.assertAlmostEqual(b.command[0],expected)
                        self.assertEqual(b.command_target[0],goal)
                        if ramp:self.assertLessEqual(abs(b.command[0]-previous),.002+1e-12)
                        previous=b.command[0]
                    self.assertAlmostEqual(b.command[0],goal)
            self.assertIn('50% step',sampling_contract('startup_mixed_ramp')['reference'])
        finally:a.close();b.close()

    def test_schedule_and_no_hidden_disturbances(self):
        e=make_env('joint_tracking_startup',0,0,0,command_sampling='startup_balanced')()
        try:
            seen=set()
            for seed in range(40):
                e.reset(seed=seed)
                self.assertEqual(e.stage.moving_start_probability,0.)
                self.assertEqual(e._initial_rolling_vx,0.)
                self.assertFalse(e.params.robust.pushes_enabled)
                np.testing.assert_array_equal(e.command,[0,0,0,.27,0])
                e.steps=374;e._sample_command();self.assertEqual(e.command[0],0)
                e.steps=375;e._sample_command();goal=e.command[0];seen.add(goal)
                self.assertIn(goal,[0.,-.1,.1,-.25,.25,-.5,.5])
                np.testing.assert_array_equal(e.command[1:],[0,0,.27,0])
                e.steps=2000;e._sample_command();self.assertEqual(e.command[0],goal)
            self.assertIn(0.,seen)
            self.assertTrue(any(v>0 for v in seen));self.assertTrue(any(v<0 for v in seen))
        finally:e.close()

    def test_physics_rewards_and_contract(self):
        a=env_params_for_stage('joint_tracking_startup');b=env_params_for_stage('joint_tracking_v16')
        for name in ['robot','joint','wheel','domain_randomization','noise']:
            self.assertEqual(asdict(getattr(a,name)),asdict(getattr(b,name)))
        self.assertEqual(asdict(precision_rewards_for_stage(STAGE_BY_NAME['joint_tracking_startup'],a)),
                         asdict(precision_rewards_for_stage(STAGE_BY_NAME['joint_tracking_v16'],b)))
        self.assertFalse(environment_contract('joint_tracking_startup',True)['disturbance']['pushes_enabled'])
        self.assertAlmostEqual(sum(sampling_contract('startup_balanced')['goal_probabilities']),1.)
        with self.assertRaises(ValueError):make_env('joint_tracking_v16',0,0,0,command_sampling='startup_balanced')()

if __name__=='__main__':unittest.main()
