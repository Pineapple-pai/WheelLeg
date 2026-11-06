"""Reject unproven/regressing candidates under matched fixed evaluation protocols."""
import argparse,json
from pathlib import Path

def assess(source,candidate):
 keys=('tasks','seeds','thresholds','model_xml_sha256','schedules','domain_randomization',
       'observation_noise','moving_starts','pushes','duration_s')
 for key in keys:
  if source['config'][key]!=candidate['config'][key]:raise ValueError(f'Unmatched evaluation protocol: {key}')
 def index(data):
  result={(r['task'],r['seed']):r for r in data['task_metrics']}
  expected={(t,s) for t in data['config']['tasks'] for s in data['config']['seeds']}
  if set(result)!=expected or len(result)!=len(data['task_metrics']):raise ValueError('Incomplete or duplicate task metrics')
  return result
 a=index(source);b=index(candidate);regressions=[];improvements=[]
 for key,r in a.items():
  for metric in ('posture_success','composite_success'):
   if r[metric] and not b[key][metric]:regressions.append(dict(task=key[0],seed=key[1],metric=metric))
  if key[0]!='stand' and not r['composite_success'] and b[key]['composite_success']:
   improvements.append(dict(task=key[0],seed=key[1]))
 passed=not regressions and bool(improvements)
 return dict(candidate_passes_gate=passed,regressions=regressions,new_nonzero_successes=improvements,
  action='Candidate for independent validation; not automatically promoted.' if passed else 'Keep source checkpoint; no demonstrated retained-skill upgrade.',
  limitation='Small nominal launch test only; passing does not establish DR, stopping, reversal or deployment readiness.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=assess(json.loads(a.source.read_text()),json.loads(a.candidate.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 raise SystemExit(0 if result['candidate_passes_gate'] else 3)
if __name__=='__main__':main()
