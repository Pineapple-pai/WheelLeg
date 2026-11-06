"""Accept a continued PPO candidate only against both source and initialization."""
from pathlib import Path
import json
from check_transfer_progress import evaluate_progress
from review_completed_transfer import read_verified
from initialize_model_transfer import digest
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/pure_ppo_precision_20261106'

def main():
 init=read_verified(OUT/'source_evaluation/results.json')
 source=read_verified(OUT/'seed_711/reward_fine/evaluation/results.json')
 candidate=read_verified(OUT/'continuation_s711/evaluation/results.json')
 dr_init=read_verified(OUT/'holdout_dr/source/results.json')
 dr_source=read_verified(OUT/'holdout_dr/reward_fine_s711/results.json')
 dr_candidate=read_verified(OUT/'holdout_dr/reward_cont_s711/results.json')
 evidence={
  'continuation_vs_200_nominal':evaluate_progress(source,candidate),
  'continuation_vs_initial_nominal':evaluate_progress(init,candidate),
  'continuation_vs_200_dr':evaluate_progress(dr_source,dr_candidate),
  'continuation_vs_initial_dr':evaluate_progress(dr_init,dr_candidate)}
 for seed in (711,713,714):
  d=read_verified(OUT/f'seed_{seed}/reward_fine/evaluation/results.json')
  assert 'diagnostic_intervention' not in d['config']
  evidence[f'replication_s{seed}']=evaluate_progress(init,d)
 assert 'diagnostic_intervention' not in candidate['config'] and 'diagnostic_intervention' not in dr_candidate['config']
 ready=all(v['accepted_for_further_training'] or (v['fully_passed_all_tasks'] and not v['regressions']) for v in evidence.values())
 cp=ROOT/'checkpoints/pure_ppo_reward_cont_s711_20261106/checkpoint'
 result=dict(recipe='reward_fine_continued',ready_for_bounded_long_training=ready,
  fully_passed_all_tasks=all(v['composite_success'] for d in (candidate,dr_candidate) for v in d['task_metrics']),
  approved_sources={str(cp):{s:digest(str(cp)+s) for s in ('.zip','_contract.json','_vecnormalize.pkl')}} if ready else {},
  evidence=evidence,training_seeds=[711,713,714],continuation_seed=1711,
  limitation='Training readiness only; retain original task thresholds. All inference is ordinary PPO, no calibration or auxiliary controller.')
 (OUT/'readiness_reward_fine_continued.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:{x:v[x] for x in ('accepted_for_further_training','relative_improvement','regressions','per_goal_regression_within_half_band')} for k,v in evidence.items()},indent=2))
 print('Ready:',ready)

if __name__=='__main__':main()
