"""A completed run must not be mistaken for a retained-skill improvement."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_refinement import assess

def fixture():
 config=dict(tasks=['stand','forward','reverse'],seeds=[1],thresholds={'relative_speed_band':.05},model_xml_sha256='same',schedules={},domain_randomization=False,observation_noise=True,moving_starts=False,pushes=False,duration_s=24)
 return dict(config=config,task_metrics=[dict(task=t,seed=1,posture_success=int(t!='reverse'),composite_success=int(t!='reverse')) for t in config['tasks']])
class RefinementGateTests(unittest.TestCase):
 def test_direction_tradeoff_is_rejected(self):
  a=fixture();b=copy.deepcopy(a);b['task_metrics'][1]['composite_success']=0;b['task_metrics'][2].update(posture_success=1,composite_success=1)
  r=assess(a,b);self.assertFalse(r['candidate_passes_gate']);self.assertEqual(len(r['regressions']),1)
 def test_unchanged_is_not_an_upgrade_and_retained_gain_passes(self):
  a=fixture();self.assertFalse(assess(a,a)['candidate_passes_gate']);b=copy.deepcopy(a);b['task_metrics'][2].update(posture_success=1,composite_success=1);self.assertTrue(assess(a,b)['candidate_passes_gate'])
 def test_unmatched_or_incomplete_data_rejected(self):
  a=fixture();b=copy.deepcopy(a);b['config']['seeds']=[2]
  with self.assertRaises(ValueError):assess(a,b)
  b=copy.deepcopy(a);b['task_metrics'].pop()
  with self.assertRaises(ValueError):assess(a,b)
if __name__=='__main__':unittest.main()
