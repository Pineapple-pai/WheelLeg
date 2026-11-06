"""Frozen three-seed, bounded startup adaptation with milestone evaluations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'checkpoints/joint_tracking_pos_v16_model20261104/checkpoint'
OUT=ROOT/'diagnostics/startup_warmstart'
SEEDS=[81,82,83]
EVAL_SEEDS=['65100','65101','65102']

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def eval_command(cp,folder,dr=False):
    return [sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),str(cp),
        '--output',str(folder),'--launch-speeds','.1','.25','.5','--seeds',*EVAL_SEEDS]+(
        ['--domain-randomization'] if dr else [])

def prepare():
    from uz05.command_sampling import sampling_contract
    from uz05.spec import MODEL_XML
    protected=[Path(str(SOURCE)+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')]
    protected += [MODEL_XML,*[ROOT/f'scripts/uz05/{name}.py' for name in ('env','spec','actuators','command_sampling')],ROOT/'scripts/train_uz05.py']
    runs=[]
    for seed in SEEDS:
        folder=OUT/f'seed_{seed}';version=f'startup_warm150_s{seed}_20261105';cp=ROOT/'checkpoints'/version/'checkpoint'
        command=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),
            '--stage','joint_tracking_startup','--command-sampling','startup_balanced',
            '--checkpoint',str(SOURCE),'--vecnormalize-path',str(SOURCE)+'_vecnormalize.pkl',
            '--preserve-action-std','--version',version,'--updates','150',
            '--rollout-steps','256','--batch-size','256','--num-envs','4','--vec-env','subproc',
            '--device','cpu','--learning-rate','.0001','--disable-adaptive-lr','--entropy-coef','.001',
            '--seed',str(seed),'--save-interval','50','--log-interval','10',
            '--initial-state-audit',str(folder/'initial_state.json'),
            '--tensorboard-log',str(folder/'tensorboard')]
        evaluations=[eval_command(str(cp)+f'_iter_{step}',folder/f'evaluation_{step}') for step in (50,150)]
        runs.append(dict(seed=seed,directory=str(folder),checkpoint=str(cp),command=command,evaluations=evaluations))
    cfg=dict(source=str(SOURCE),seeds=SEEDS,evaluation_seeds=EVAL_SEEDS,
        protected={str(p):digest(p) for p in protected},sampling=sampling_contract('startup_balanced'),runs=runs,
        baseline=eval_command(SOURCE,OUT/'source_evaluation'),
        source_dr=eval_command(SOURCE,OUT/'source_dr',True),
        transitions=3*150*4*256,objective='Retain standing; improve low-speed launch; do not auto-promote lucky seeds.',
        gate='Stand3/3 in all training seeds and all6nonzero tasks3/3 in >=2training seeds; otherwise no speed expansion.',
        comparison='Compare fixed source on paired evaluation seeds to50/150milestones. No equal-budget scratch arm; no claim of initialization-only causality.')
    OUT.mkdir(exist_ok=True);path=OUT/'experiment_config.json'
    if path.exists():assert json.loads(path.read_text())==cfg,'Frozen plan differs'
    else:path.write_text(json.dumps(cfg,indent=2)+'\n')
    commands=[cfg['baseline'],cfg['source_dr']]
    for r in runs:commands += [r['command'],*r['evaluations'],eval_command(r['checkpoint'],Path(r['directory'])/'evaluation_dr',True)]
    (OUT/'commands.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+'\n'.join(map(shlex.join,commands))+'\n')
    return cfg

def evaluate(command,log):
    folder=Path(command[command.index('--output')+1]);result=folder/'results.json'
    if result.exists():
        cfg=json.loads(result.read_text())['config'];cp=str(command[2])
        assert cfg['checkpoint']==cp
        assert cfg['seeds']==list(map(int,EVAL_SEEDS))
        for suffix,h in cfg['protected'].items():assert digest(cp+suffix)==h
        return
    with log.open('w') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def execute(run):
    folder=Path(run['directory']);folder.mkdir(exist_ok=True);marker=folder/'training_completed.json'
    cp=run['checkpoint']
    if marker.exists():
        for suffix,h in json.loads(marker.read_text()).items():assert digest(cp+suffix)==h
    else:
        assert not Path(cp+'.zip').exists(),'Refuse overwrite'
        with (folder/'train.log').open('w') as stream:subprocess.run(run['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        marker.write_text(json.dumps({s:digest(cp+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')})+'\n')
    for step,command in zip((50,150),run['evaluations']):evaluate(command,folder/f'evaluation_{step}.log')
    evaluate(eval_command(cp,folder/'evaluation_dr',True),folder/'evaluation_dr.log')
    print('Completed seed',run['seed'],flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');args=p.parse_args()
    cfg=prepare()
    if not args.execute:print('Frozen plan:',OUT/'experiment_config.json');return
    evaluate(cfg['baseline'],OUT/'source_evaluation.log')
    evaluate(cfg['source_dr'],OUT/'source_dr.log')
    print('Source baseline complete',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(execute,r) for r in cfg['runs']]):future.result()
    assert all(digest(p)==h for p,h in cfg['protected'].items())
    (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True))+'\n')

if __name__=='__main__':main()
