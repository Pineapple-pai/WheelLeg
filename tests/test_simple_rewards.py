"""Guard the minimal reward boundary and the one-factor-at-a-time comparison."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.env import UZ05Env
from uz05.spec import (STAGE_BY_NAME, SimpleTrackingRewards, TightTrackingRewards,
                       TightSurvivalRewards, ResponsiveSimpleRewards, ResponsiveSurvivalRewards)
from uz05.simple_rewards import tracking_terms
from compare_simple_rewards import ARMS


class SimpleRewardTest(unittest.TestCase):
    def env(self,stage='joint_tracking_simple'):
        e = UZ05Env(stage)
        e.params.domain_randomization.enabled=False
        e.params.robust.pushes_enabled=False
        e.reset(seed=8)
        self.addCleanup(e.close)
        return e

    def terms(self, **kwargs):
        values = dict(vx_error=0.,yaw_rate_error=0.,height_error=0.,roll=0.,pitch_error=0.,
                      heading_error=0.,stop_distance=0.,action_delta=np.zeros(6),terminated=False,
                      axle_offset=0.,axle_enabled=False,dt=.008)
        values.update(kwargs)
        return tracking_terms(SimpleTrackingRewards(),**values)

    def test_only_four_groups_and_optional_axle_cost(self):
        base=self.terms()
        self.assertEqual(set(base),{'task_tracking','body_attitude','action_smoothness','failure'})
        extra=self.terms(axle_offset=.03,axle_enabled=True)
        self.assertEqual(set(extra)-set(base),{'axle_alignment'})
        for key in base:self.assertEqual(base[key],extra[key])
        self.assertLess(extra['axle_alignment'],0)
        self.assertEqual(self.terms(axle_offset=.3),base)

    def test_time_integral_and_terminal_impulse_use_distinct_units(self):
        a=self.terms(vx_error=.6,terminated=True)
        b=self.terms(vx_error=.6,terminated=True,dt=.016)
        for key in a:
            self.assertAlmostEqual(b[key],a[key] if key=='failure' else 2*a[key])
        self.assertEqual(a['failure'],-5)
        self.assertEqual(self.terms()['failure'],0)

    def test_large_error_has_progress_signal_and_all_errors_are_symmetric(self):
        for error in (.05,.5,2.,4.):
            a=self.terms(vx_error=error)
            self.assertEqual(a,self.terms(vx_error=-error))
            self.assertGreater(self.terms(vx_error=error-.01)['task_tracking'],a['task_tracking'])
        self.assertEqual(self.terms(roll=.2,pitch_error=.3,heading_error=.1),
                         self.terms(roll=-.2,pitch_error=-.3,heading_error=-.1))

    def test_legacy_reward_helpers_are_never_called(self):
        e=self.env()
        with patch.object(e,'_precision_terms',side_effect=AssertionError('old reward entered')):
            _,reward,_,_,info=e.step(np.zeros(6))
        self.assertEqual(set(info['reward_terms']),set(self.terms()))
        self.assertAlmostEqual(reward,sum(info['reward_terms'].values()))

    def test_shared_sampling_observations_physics_and_termination_match_v15(self):
        environments=[self.env(stage) for stage in ARMS.values()]
        action=np.array([.1,-.2,.9,-.8,.7,-.6])
        for _ in range(10):
            results=[e.step(action) for e in environments]
            for e,result in zip(environments[1:],results[1:]):
                np.testing.assert_array_equal(result[0],results[0][0])
                np.testing.assert_array_equal(e.sim.data.qpos,environments[0].sim.data.qpos)
                self.assertEqual(result[2:4],results[0][2:4])
        for _ in range(50):
            for e in environments:e._sample_command()
            for e in environments[1:]:
                np.testing.assert_array_equal(e.command,environments[0].command)
                self.assertEqual(e.command_steps_left,environments[0].command_steps_left)

    def test_two_by_two_factors_and_checkpoint_interface(self):
        from train_uz05 import write_contract,require_compatible_contract
        base=asdict(STAGE_BY_NAME['joint_tracking_v15'])
        for name in list(ARMS.values())[1:]:
            config=asdict(STAGE_BY_NAME[name])
            for key in ('name','simple_reward','simple_pitch_reference','simple_axle_cost'):
                config.pop(key);base.pop(key,None)
            self.assertEqual(config,base)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy';write_contract(path,'joint_tracking_v15',1)
            for stage in list(ARMS.values())[1:]:require_compatible_contract(path,stage)

    def test_stop_anchor_and_turn_heading_are_explicit_task_targets(self):
        e=self.env();action=np.zeros(6)
        e.command[:]=[0,0,0,.27,0];e.command_target[:]=e.command;e.command_steps_left=100
        e._obs();e.hold_xy=e.sim.base_pos[:2].copy()
        a=e._simple_reward({},action,False)
        e.hold_xy-=np.array([.2,0])
        b=e._simple_reward({},action,False)
        self.assertLess(b['task_tracking'],a['task_tracking'])
        self.assertEqual(b['body_attitude'],a['body_attitude'])
        e.command[2]=1.
        with patch.object(e,'_heading_error',side_effect=AssertionError('turn heading locked')):
            e._simple_reward({},action,False)

    def test_periodic_action_change_does_not_penalize_full_joint_turn(self):
        e=self.env()
        e.previous_action=np.array([0,0,.99,0,0,0])
        near=np.array([0,0,-.99,0,0,0])
        far=np.array([0,0,0,0,0,0])
        self.assertGreater(e._simple_reward({},near,False)['action_smoothness'],
                           e._simple_reward({},far,False)['action_smoothness'])

    def test_plan_can_be_reviewed_repeatedly_without_starting_training(self):
        from compare_simple_rewards import ROOT, main
        with tempfile.TemporaryDirectory(prefix='simple_plan_test_',dir=ROOT/'logs') as folder:
            argv=['compare_simple_rewards.py','--from-scratch','--arms','minimal',
                  '--seeds','0','--name',Path(folder).name]
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
                main()
                before=(Path(folder)/'manifest.json').read_text()
                main()
                self.assertEqual((Path(folder)/'manifest.json').read_text(),before)

    def test_tight_velocity_keeps_far_error_slope_and_isolates_failure_cost(self):
        def score(p,error,failed=False):
            return tracking_terms(p,vx_error=error,yaw_rate_error=0.,height_error=0.,roll=0.,
                pitch_error=0.,heading_error=0.,stop_distance=0.,action_delta=np.zeros(6),
                terminated=failed,axle_offset=0.,axle_enabled=True,dt=.008)
        base,tight,survival=SimpleTrackingRewards(),TightTrackingRewards(),TightSurvivalRewards()
        for sign in (-1,1):
            self.assertAlmostEqual(score(base,sign*2)['task_tracking']-score(base,sign*2.1)['task_tracking'],
                                   score(tight,sign*2)['task_tracking']-score(tight,sign*2.1)['task_tracking'])
        a=score(base,0)['task_tracking']-score(base,.01)['task_tracking']
        b=score(tight,0)['task_tracking']-score(tight,.01)['task_tracking']
        self.assertAlmostEqual(b,6*a)
        self.assertEqual(score(tight,.1),score(survival,.1))
        self.assertEqual(score(tight,.1,True)['failure'],-5.)
        self.assertEqual(score(survival,.1,True)['failure'],-20.)

    def test_velocity_priority_changes_only_tracking_cost_and_preserves_five_groups(self):
        a,b=asdict(TightSurvivalRewards()),asdict(ResponsiveSimpleRewards())
        self.assertEqual(b.pop('vx_weight'),2*a.pop('vx_weight'))
        a.pop('schema');b.pop('schema')
        self.assertEqual(a,b)
        e=self.env('joint_tracking_simple_responsive')
        _,_,_,_,info=e.step(np.zeros(6))
        self.assertEqual(set(info['reward_terms']),
                         {'task_tracking','body_attitude','action_smoothness','failure','axle_alignment'})

    def test_responsive_recovery_changes_only_failure_budget(self):
        a,b=asdict(ResponsiveSimpleRewards()),asdict(ResponsiveSurvivalRewards())
        self.assertEqual(a.pop('failure_cost'),20.)
        self.assertEqual(b.pop('failure_cost'),30.)
        a.pop('schema');b.pop('schema')
        self.assertEqual(a,b)
        e=self.env('joint_tracking_simple_responsive_survival')
        terms=e._simple_reward({},np.zeros(6),True)
        self.assertEqual(terms['failure'],-30.)
        self.assertEqual(len(terms),5)

    def test_recovery_sampling_targets_high_speed_full_reversals_and_high_stops(self):
        e=self.env('joint_tracking_simple_recovery');e.steps=100
        for height,expected_stop,expected_reverse in ((.31,.40,.35),(.27,.20,.50)):
            stops=reversals=0
            for _ in range(2000):
                e.command[:]=[1.9,0,0,height,0];e.command_target[:]=e.command
                e._sample_command()
                if e.command[0]==0 and e.command[2]==0 and e.command[3]==height:stops+=1
                if e.command[0]==-1.9:
                    reversals+=1
                    self.assertEqual(e.command[3],height)
                    self.assertEqual(e.command[2],0)
            self.assertAlmostEqual(stops/2000,expected_stop,delta=.04)
            self.assertAlmostEqual(reversals/2000,expected_reverse,delta=.04)
        e.steps=0;commands=[]
        for _ in range(2000):e._sample_command();commands.append(e.command.copy())
        a=np.asarray(commands)
        self.assertTrue(.46<np.mean(a[:,3]>=.30)<.58)
        moving=np.abs(a[a[:,0]!=0,0])
        self.assertAlmostEqual(float(np.mean(moving>=1.5)),.45,delta=.05)

    def test_recovery_sampling_adds_no_force_or_extra_reward(self):
        a,b=self.env('joint_tracking_simple_recovery'),self.env('joint_tracking_simple_responsive')
        action=np.array([.1,-.2,.9,-.8,.7,-.6])
        for _ in range(10):
            aa,bb=a.step(action),b.step(action)
            np.testing.assert_array_equal(a.sim.data.qpos,b.sim.data.qpos)
            self.assertEqual(set(aa[4]['reward_terms']),set(bb[4]['reward_terms']))
        self.assertEqual(len(aa[4]['reward_terms']),5)


if __name__=='__main__':unittest.main()
