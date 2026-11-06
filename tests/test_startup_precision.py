"""The precision candidate changes one velocity cost, including zero commands."""
from dataclasses import asdict
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from uz05.spec import STAGE_BY_NAME,env_params_for_stage,precision_rewards_for_stage
from train_uz05 import make_env,can_preserve_reward_statistics
from uz05.command_sampling import sampling_contract

class StartupPrecisionTests(unittest.TestCase):
 def test_low_sampler_stays_within_declared_goals_and_preserves_initial_state(self):
  a=make_env('joint_tracking_startup_precision',0,44,0,command_sampling='startup_balanced')()
  b=make_env('joint_tracking_startup_precision',0,44,0,command_sampling='startup_low')()
  try:
   for seed in range(20):
    np.testing.assert_array_equal(a.reset(seed=seed)[0],b.reset(seed=seed)[0])
    for env in (a,b):env.steps=375;env._sample_command()
    self.assertIn(b.command[0],[0.,-.1,.1,-.25,.25])
   contract=sampling_contract('startup_low')
   self.assertEqual(contract['goal_probabilities'],[.25,.1875,.1875,.1875,.1875])
   self.assertAlmostEqual(sum(contract['goal_probabilities']),1.)
  finally:a.close();b.close()
 def test_only_velocity_error_scale_changes(self):
  a=STAGE_BY_NAME['joint_tracking_startup'];b=STAGE_BY_NAME['joint_tracking_startup_precision']
  x=asdict(a);y=asdict(b);x.pop('name');y.pop('name');self.assertEqual(x,y)
  pa=env_params_for_stage(a);pb=env_params_for_stage(b)
  for key in ('robot','joint','wheel','domain_randomization','noise','robust'):
   self.assertEqual(asdict(getattr(pa,key)),asdict(getattr(pb,key)))
  ra=asdict(precision_rewards_for_stage(a,pa));rb=asdict(precision_rewards_for_stage(b,pb))
  self.assertEqual([k for k in ra if ra[k]!=rb[k]],['vx_error_scale_m_s'])
  self.assertEqual(ra['vx_error_scale_m_s'],.15);self.assertEqual(rb['vx_error_scale_m_s'],.05)
  self.assertFalse(can_preserve_reward_statistics(dict(stage=a.name,precision_objective=ra),b.name))
 def test_identical_plant_sampling_and_stronger_error_cost(self):
  a=make_env('joint_tracking_startup',0,44,0,command_sampling='startup_balanced')()
  b=make_env('joint_tracking_startup_precision',0,44,0,command_sampling='startup_balanced')()
  try:
   for seed in range(8):
    oa=a.reset(seed=seed)[0];ob=b.reset(seed=seed)[0]
    np.testing.assert_array_equal(oa,ob)
    np.testing.assert_array_equal(a.sim.data.qpos,b.sim.data.qpos)
    for env in (a,b):env.steps=375;env._sample_command()
    np.testing.assert_array_equal(a.command,b.command)
   for error in (0.,.01,-.01,.1,-.1):
    ca=a._transition_terms(0.,error,0.)['vx_error_dense']
    cb=b._transition_terms(0.,error,0.)['vx_error_dense']
    if error==0:self.assertEqual(ca,cb)
    else:self.assertLess(cb,ca)
   a.command[0]=0.;b.command[0]=0.
   self.assertLess(b._transition_terms(0.,.01,0.)['vx_error_dense'],a._transition_terms(0.,.01,0.)['vx_error_dense'])
  finally:a.close();b.close()
if __name__=='__main__':unittest.main()
