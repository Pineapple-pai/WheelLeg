"""Frozen v17 single-factor wheel torque adaptation; separate from diagnostics."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','TORCH_NUM_THREADS'):os.environ[k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_m3508_validation import CHECKPOINT,PROFILES,TASKS,THRESHOLDS,digest
OUT=ROOT/'diagnostics/ppo_v17_torque_ab'
OLD_SEEDS=list(range(51000,51005));NEW_SEEDS=list(range(61000,61003))
MILESTONES=[50,100,150,200]


def prepare():
    from train_uz05 import actuator_contract,environment_contract
    from uz05.spec import STAGE_BY_NAME
    path=OUT/'experiment_config.json'
    if path.exists():return json.loads(path.read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in [1,2,3]:
      for model in (['A_legacy','B_rated'] if seed%2 else ['B_rated','A_legacy']):
        version=f'ppo_v17_torque_ab_s{seed}_{model}'
        checkpoint=ROOT/'checkpoints'/version/'checkpoint';directory=OUT/f'seed_{seed}'/model
        if checkpoint.parent.exists():raise RuntimeError(f'Refuse existing checkpoint directory: {checkpoint.parent}')
        directory.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),
          '--stage','joint_tracking_v16','--checkpoint',str(CHECKPOINT),'--version',version,
          '--updates','200','--rollout-steps','512','--batch-size','512','--num-envs','8','--vec-env','subproc',
          '--device','cpu','--learning-rate','0.00003','--disable-adaptive-lr','--entropy-coef','0.001',
          '--preserve-action-std','--seed',str(seed),'--vecnormalize-path',str(CHECKPOINT)+'_vecnormalize.pkl',
          '--save-interval','50','--stand-level','0','--tensorboard-log',str(directory/'tensorboard'),
          '--initial-state-audit',str(directory/'initial_state.json'),
          '--wheel-torque-model',model]
        if model=='B_rated':command+=['--adapt-wheel-torque-model']
        runs.append(dict(seed=seed,mode=model,directory=str(directory),checkpoint=str(checkpoint),command=command))
    protected=['uz05_修正后(3).xml','scripts/uz05/actuators.py','scripts/uz05/spec.py','scripts/uz05/env.py',
      'scripts/uz05/model.py','scripts/uz05/reset_projection.py','scripts/uz05/simple_rewards.py',
      str(CHECKPOINT.relative_to(ROOT))+'.zip',str(CHECKPOINT.relative_to(ROOT))+'_vecnormalize.pkl',
      str(CHECKPOINT.relative_to(ROOT))+'_contract.json']
    sources=['scripts/train_uz05.py','scripts/run_wheel_speed_steady_state.py','scripts/run_ppo_v17_torque_ab.py',
             'scripts/uz05/experiment_audit.py','scripts/uz05/m3508_metrics.py','scripts/uz05/wheel_speed_diagnostics.py']
    cfg={'frozen_utc':datetime.now(timezone.utc).isoformat(),'runs':runs,'training_seeds':[1,2,3],
      'steps_per_update':4096,'updates_per_run':200,'steps_per_run':819200,'total_training_steps':4915200,
      'source_checkpoint':str(CHECKPOINT),'source_training_config':json.loads((CHECKPOINT.parent/'training_config.json').read_text()),
      'actuator_models':{x:actuator_contract('joint_tracking_v16',x) for x in ['A_legacy','B_rated']},
      'training_environment':environment_contract('joint_tracking_v16',True,True),
      'training_stage':asdict(STAGE_BY_NAME['joint_tracking_v16']),
      'domain_randomization_semantics':'Same draws/ranges and relative torque coefficient multiplier .95..1.05. Only efficiency multiplier .70->1.00; shared output cap2.46 retained. Thus B high gain samples may reach cap, not raise cap. Wheel damping nominal.02 with unchanged .9..1.1 randomization.',
      'evaluation':{'historical_seeds':OLD_SEEDS,'independent_seeds':NEW_SEEDS,
        'milestones':[0]+MILESTONES,'intermediate_dynamics':'B_rated','intermediate_seeds':OLD_SEEDS,
        'baseline_and_final_dynamics':['A_legacy','B_rated'],'baseline_and_final_seeds':OLD_SEEDS+NEW_SEEDS,
        'profiles':PROFILES,'tasks':TASKS,'thresholds':THRESHOLDS,
        'environment':'Projected reset; nominal model; original observation noise; no randomization/pushes/moving starts; original step reference; deterministic fixed weights;24s.'},
      'resume_limitations':['No environment/RNG/rollout/discounted-return state restored',
        'Local learn timestep/progress restarts; source constant LR/clip unchanged; cumulative PPO epoch/Adam steps retained'],
      'decision_rule':{'A':'All3 training seeds improve mean forward/reverse steady RMSE >=.02m/s vs A-trained on same B dynamics; improve5% success; preserve basic task posture, stop tracking and high reversal posture vs v16 B. Confirm direction on independent seeds.',
        'C':'Any basic stand/forward/reverse/launch/stop posture or stop tracking rate drops >=.20 vs v16 B in >=2 training seeds, or high reversal posture drops >=.20 in >=2 seeds; preserve earliest measured regression interval.',
        'B':'Otherwise no reliable superiority; do not extend4000. C takes precedence over A. No posthoc checkpoint selection.'},
      'protected_hashes':{p:digest(ROOT/p) for p in protected},'source_hashes':{p:digest(ROOT/p) for p in sources}}
    path.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
    conda='/home/p/NavRL/isaac-training/yes/condabin/conda run --no-capture-output -n sim python'
    text='#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+conda+' scripts/run_ppo_v17_torque_ab.py --baseline\n'
    text+='\n'.join(shlex.join(r['command']) for r in runs)+'\n'
    text+=conda+' scripts/run_ppo_v17_torque_ab.py --evaluate\n'+conda+' scripts/analyze_ppo_v17_torque_ab.py\n'
    (OUT/'commands.sh').write_text(text)
    return cfg


def verify(cfg):
    for path,h in {**cfg['protected_hashes'],**cfg['source_hashes']}.items():
        assert digest(ROOT/path)==h,path


def eval_job(checkpoint,model,protocol,seeds,destination):
    from run_wheel_speed_steady_state import evaluate
    result=evaluate('base',protocol,seeds,checkpoint_override=checkpoint,destination=destination,torque_model_override=model)
    metadata={'checkpoint':checkpoint,'checkpoint_sha256':digest(checkpoint+'.zip'),
      'normalizer_sha256':digest(checkpoint+'_vecnormalize.pkl'),'eval_torque_model':model,
      'nominal_wheel_damping':.02,'current_limit_a':10,'wheel_output_limit_nm':1.722 if model=='A_legacy' else 2.46,
      'eval_seeds':seeds,'reference':'step','domain_randomization':False,'closed_chain_reset':True}
    Path(destination,'evaluation_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return len(result)


def evaluate_jobs(jobs,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(eval_job,*job):job[1:3] for job in jobs}
        for f in as_completed(fs):print('EVAL',fs[f],f.result(),flush=True)


def baseline(cfg,workers):
    # Reuse exact historical nominal A/B traces only, source hashes checked first.
    verify(cfg);jobs=[]
    previous=ROOT/'diagnostics/wheel_speed_steady_state'
    frozen=json.loads((previous/'experiment_config.json').read_text())
    assert frozen['checkpoint_sha256']==digest(str(CHECKPOINT)+'.zip')
    for model,old in [('A_legacy','legacy_A'),('B_rated','base')]:
      for protocol in PROFILES:
        dest=OUT/'evaluation'/'baseline'/model/protocol;dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'episodes.json').exists():
            for seed in OLD_SEEDS:shutil.copy2(previous/old/protocol/f'{seed}.npz',dest/f'{seed}.npz')
            shutil.copy2(previous/old/protocol/'episodes.json',dest/'episodes.json')
        jobs.append((str(CHECKPOINT),model,protocol,OLD_SEEDS+NEW_SEEDS,str(dest)))
    evaluate_jobs(jobs,workers);verify(cfg)
    (OUT/'baseline_complete.json').write_text(json.dumps({'complete':True,'episodes':96,'historical_reused':60,'new_evaluations':36},indent=2)+'\n')


def train(cfg):
    assert (OUT/'baseline_complete.json').exists(),'Run Update0 first'
    verify(cfg)
    for run in cfg['runs']:
        dest=Path(run['directory']);done=dest/'completed.json'
        if done.exists():continue
        if Path(run['checkpoint']).parent.exists():raise RuntimeError('Partial run requires manual inspection: '+run['checkpoint'])
        (OUT/'status.json').write_text(json.dumps({'active_training':run},indent=2)+'\n')
        print('TRAIN',run['seed'],run['mode'],flush=True);start=time.time()
        with (dest/'train.log').open('w') as f:
            result=subprocess.run(run['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Training failed: '+str(dest/'train.log'))
        done.write_text(json.dumps({'seconds':time.time()-start,'checkpoint_sha256':digest(run['checkpoint']+'.zip')},indent=2)+'\n')
        audits=[json.loads((Path(r['directory'])/'initial_state.json').read_text()) for r in cfg['runs'] if (Path(r['directory'])/'initial_state.json').exists()]
        assert all(x==audits[0] for x in audits),'Pre-learn states differ'
        (OUT/'resume_state_verification.json').write_text(json.dumps({'verified_initial_runs':len(audits),'all_available_states_identical':True,'initial_state':audits[0]},indent=2)+'\n')
        print('TRAIN DONE',run['seed'],run['mode'],time.time()-start,flush=True)
    verify(cfg);(OUT/'training_complete.json').write_text('{"complete": true}\n')


def final_evaluation(cfg,workers):
    assert (OUT/'training_complete.json').exists();verify(cfg);jobs=[]
    for run in cfg['runs']:
      for update in MILESTONES:
        checkpoint=run['checkpoint']+(f'_iter_{update}' if update<200 else '')
        for model in (['A_legacy','B_rated'] if update==200 else ['B_rated']):
          for protocol in PROFILES:
            dest=OUT/'evaluation'/f'seed_{run["seed"]}'/run['mode']/f'update_{update}'/model/protocol
            seeds=OLD_SEEDS+NEW_SEEDS if update==200 else OLD_SEEDS
            jobs.append((checkpoint,model,protocol,seeds,str(dest)))
    evaluate_jobs(jobs,workers);verify(cfg)
    (OUT/'evaluation_complete.json').write_text(json.dumps({'complete':True,'evaluation_jobs':len(jobs)},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();cfg=prepare()
    if args.baseline or args.execute:baseline(cfg,args.workers)
    if args.train or args.execute:train(cfg)
    if args.evaluate or args.execute:final_evaluation(cfg,args.workers)
if __name__=='__main__':main()
