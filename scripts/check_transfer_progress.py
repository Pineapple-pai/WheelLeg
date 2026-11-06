"""Distinguish retained-skill adaptation progress from full task acceptance."""
import argparse,json,math
from pathlib import Path
import numpy as np
from check_refinement import assess

def evaluate_progress(source,candidate):
 strict=assess(source,candidate)  # Protocol/coverage validation and per-seed regression list.
 a={(r['task'],r['seed']):r for r in source['task_metrics']};b={(r['task'],r['seed']):r for r in candidate['task_metrics']}
 scores=[]
 for key,r in a.items():
  goal=source['config']['schedules'][key[0]][-1][1]
  if goal==0:continue
  q=b[key]
  x=r.get('steady_rmse_m_s');y=q.get('steady_rmse_m_s')
  if not r.get('steady_complete') or not q.get('steady_complete') or x is None or y is None or not math.isfinite(x) or not math.isfinite(y):
   return dict(accepted_for_further_training=False,fully_passed_all_tasks=False,reason='Incomplete nonzero steady interval',regressions=strict['regressions'])
  band=source['config']['thresholds']['relative_speed_band']*abs(goal)
  scores.append(dict(task=key[0],seed=key[1],goal=goal,source=x/band,candidate=y/band))
 if not scores:raise ValueError('Nonzero task coverage required')
 mean=lambda records,key:float(np.mean([r[key] for r in records]))
 before=mean(scores,'source');after=mean(scores,'candidate');ratio=(before-after)/max(before,1e-12)
 per_goal={}
 for task in dict.fromkeys(r['task'] for r in scores):
  x=[r for r in scores if r['task']==task];per_goal[task]=dict(source=mean(x,'source'),candidate=mean(x,'candidate'))
 task_retained=all(r['candidate']<=r['source']+.5 for r in per_goal.values())
 direction_improved=all(mean([r for r in scores if sign*r['goal']>0],'candidate')<mean([r for r in scores if sign*r['goal']>0],'source') for sign in (1,-1))
 gained=bool(strict['new_nonzero_successes'])
 accepted=not strict['regressions'] and task_retained and ((ratio>=.10 and direction_improved) or gained)
 full=all(r['composite_success'] for r in b.values())
 return dict(accepted_for_further_training=bool(accepted),fully_passed_all_tasks=bool(full),regressions=strict['regressions'],
  normalized_steady_rmse_source=before,normalized_steady_rmse_candidate=after,relative_improvement=ratio,
  normalized_rmse_by_task=per_goal,both_directions_improved=bool(direction_improved),per_goal_regression_within_half_band=bool(task_retained),
  new_nonzero_successes=strict['new_nonzero_successes'],
  criterion='Keep each source posture/composite success; no goal loses >0.5 speed band meanRMSE; require new nonzero success OR >=10% relative-band-normalized RMSE gain with both directions improved.',
  interpretation='Progress gate for bounded adaptation only. Frozen5% tracking/posture task acceptance is unchanged; not deployment approval.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=evaluate_progress(json.loads(a.source.read_text()),json.loads(a.candidate.read_text()));a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if r['accepted_for_further_training'] else 3)
if __name__=='__main__':main()
