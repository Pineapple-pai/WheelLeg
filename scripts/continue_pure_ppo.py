"""Checkpointed pure-PPO continuation; reject regressions, never add inference helpers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime
from initialize_model_transfer import digest
from check_transfer_progress import evaluate_progress
from train_uz05 import require_compatible_contract
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,required=True)
 p.add_argument('--readiness',type=Path,required=True,help='Evidence from replicated unassisted PPO evaluations')
 p.add_argument('--version',default='pure_ppo_long_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
 p.add_argument('--segments',type=int,default=10);p.add_argument('--updates-per-segment',type=int,default=200)
 p.add_argument('--seed',type=int,default=1701);p.add_argument('--eval-seeds',nargs='+',type=int,default=[66800,66801,66802])
 a=p.parse_args()
 if Path(a.version).name!=a.version or a.segments<1 or a.updates_per_segment<1:p.error('Invalid version/budget')
 evidence=json.loads(a.readiness.read_text());source=a.source.resolve()
 if not evidence.get('ready_for_bounded_long_training'):raise SystemExit('Readiness not established')
 approved=evidence['approved_sources'].get(str(source))
 if not approved:raise SystemExit('Source is absent from readiness evidence')
 for suffix,h in approved.items():
  if digest(str(source)+suffix)!=h:raise SystemExit('Source changed: '+suffix)
 contract=json.loads(Path(str(source)+'_contract.json').read_text());stage=contract['stage']
 sampler=contract.get('command_sampling',{}).get('name','startup_balanced')
 if sampler not in ('startup_balanced','startup_stratified'):raise SystemExit('Unsupported sampler')
 if stage not in ('joint_tracking_startup_precision','joint_tracking_startup_precision_fine'):raise SystemExit('Unsupported stage')
 require_compatible_contract(source,stage)
 out=ROOT/'diagnostics'/a.version
 if out.exists():raise FileExistsError(out)
 out.mkdir()
 (out/'plan.json').write_text(json.dumps(dict(arguments=vars(a)|{'source':str(source),'readiness':str(a.readiness)},
   stage=stage,sampler=sampler,readiness_sha256=digest(a.readiness),inference='Original PPO only',
   objective='Same as accepted source; noise/DR ON; no teacher, correction, residual or filter'),indent=2)+'\n')
 def run(command,log):
  with log.open('w') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 def evaluate(cp,folder):
  run([sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),str(cp),'--output',str(folder),
       '--launch-speeds','.1','.25','.5','--seeds',*map(str,a.eval_seeds)],folder.parent/(folder.name+'.log'))
  d=json.loads((folder/'results.json').read_text())
  assert 'diagnostic_intervention' not in d['config']
  return d
 baseline=evaluate(source,out/'source_evaluation')
 if not all(r['posture_success'] for r in baseline['task_metrics']):raise SystemExit('Source lost posture on new evaluation seeds')
 for i in range(1,a.segments+1):
  version=f'{a.version}_segment_{i:02d}';cp=ROOT/'checkpoints'/version/'checkpoint';folder=out/f'segment_{i:02d}'
  if cp.parent.exists():raise FileExistsError(cp.parent)
  folder.mkdir()
  command=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),'--stage',stage,'--command-sampling',sampler,
    '--checkpoint',str(source),'--preserve-action-std','--vecnormalize-path',str(source)+'_vecnormalize.pkl',
    '--version',version,'--updates',str(a.updates_per_segment),'--rollout-steps','512','--batch-size','512',
    '--num-envs','8','--vec-env','subproc','--device','cpu','--learning-rate','.00001','--disable-adaptive-lr',
    '--entropy-coef','0','--seed',str(a.seed+i),'--save-interval','50','--log-interval','25',
    '--initial-state-audit',str(folder/'initial_state.json'),'--tensorboard-log',str(folder/'tensorboard')]
  (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
  print(f'Training segment {i}/{a.segments}: {version}',flush=True);run(command,folder/'train.log')
  candidate=evaluate(cp,folder/'evaluation');gate=evaluate_progress(baseline,candidate)
  (folder/'progress_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
  if not (gate['accepted_for_further_training'] or (gate['fully_passed_all_tasks'] and not gate['regressions'])):
   (out/'last_accepted_source.txt').write_text(str(source)+'\n')
   print('Regression/insufficient progress: stopping; candidate retained, source unchanged.',flush=True)
   raise SystemExit(3)
  source=cp;baseline=candidate;(out/'last_accepted_source.txt').write_text(str(source)+'\n')
  print('Accepted bounded segment; full nominal pass:',gate['fully_passed_all_tasks'],flush=True)
  if gate['fully_passed_all_tasks']:
   print('Nominal target reached; stop for independent randomized validation.',flush=True);break

if __name__=='__main__':main()
