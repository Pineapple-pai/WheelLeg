"""A smaller mean error cannot hide lost balance or a direction tradeoff."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_transfer_progress import evaluate_progress

def data():
 tasks=['stand','p','m'];return dict(config=dict(tasks=tasks,seeds=[1],thresholds={'relative_speed_band':.05},model_xml_sha256='same',schedules={'stand':[[0,0]],'p':[[3,.25]],'m':[[3,-.25]]},domain_randomization=False,observation_noise=True,moving_starts=False,pushes=False,duration_s=24),task_metrics=[dict(task=t,seed=1,posture_success=1,composite_success=int(t=='stand'),steady_complete=1,steady_rmse_m_s=.05 if t!='stand' else .005) for t in tasks])
class TransferProgressTests(unittest.TestCase):
 def test_balanced_progress_is_not_full_acceptance(self):
  a=data();b=copy.deepcopy(a)
  for r in b['task_metrics'][1:]:r['steady_rmse_m_s']=.025
  q=evaluate_progress(a,b);self.assertTrue(q['accepted_for_further_training']);self.assertFalse(q['fully_passed_all_tasks'])
 def test_regression_and_direction_tradeoff_rejected(self):
  a=data();b=copy.deepcopy(a);b['task_metrics'][1]['steady_rmse_m_s']=.001;b['task_metrics'][2]['steady_rmse_m_s']=.06
  self.assertFalse(evaluate_progress(a,b)['accepted_for_further_training'])
  b=copy.deepcopy(a)
  for r in b['task_metrics'][1:]:r['steady_rmse_m_s']=.025
  b['task_metrics'][1]['posture_success']=0;self.assertFalse(evaluate_progress(a,b)['accepted_for_further_training'])
 def test_missing_tail_rejected(self):
  a=data();b=copy.deepcopy(a);b['task_metrics'][1].update(steady_complete=0,steady_rmse_m_s=None)
  self.assertFalse(evaluate_progress(a,b)['accepted_for_further_training'])
if __name__=='__main__':unittest.main()
