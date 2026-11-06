"""Paired continuation from the new-model PPO; only sampling differs by arm."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/ppo_launch_sampling'
SOURCE=ROOT/'checkpoints/joint_tracking_pos_v16_model20261104/checkpoint'
SEEDS=[71,72,73]
ARMS=['original','low_band','launch_focus']
PYTHON='/home/p/NavRL/isaac-training/yes/envs/wheelleg-train/bin/python'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    sys.path.insert(0,str(ROOT/'scripts'))
    from uz05.command_sampling import sampling_contract
    from uz05.spec import MODEL_XML
    protected={str(SOURCE)+s:digest(str(SOURCE)+s) for s in ['.zip','_vecnormalize.pkl','_contract.json']}
    protected[str(MODEL_XML)]=digest(MODEL_XML)
    for f in ['scripts/uz05/env.py','scripts/uz05/actuators.py','scripts/uz05/spec.py',
              'scripts/uz05/model.py','scripts/uz05/command_sampling.py','scripts/train_uz05.py']:
        protected[f]=digest(ROOT/f)
    runs=[]
    for seed in SEEDS:
        for arm in ARMS:
            folder=OUT/f'seed_{seed}'/arm
            version=f'ppo_launch_sampling_s{seed}_{arm}'
            checkpoint=ROOT/'checkpoints'/version/'checkpoint'
            cmd=[PYTHON,'-u',str(ROOT/'scripts/train_uz05.py'),'--stage','joint_tracking_v16',
                 '--checkpoint',str(SOURCE),'--version',version,'--updates','100',
                 '--rollout-steps','256','--batch-size','256','--num-envs','4','--vec-env','subproc',
                 '--device','cpu','--learning-rate','0.0001','--disable-adaptive-lr',
                 '--entropy-coef','0.001','--preserve-action-std','--seed',str(seed),
                 '--vecnormalize-path',str(SOURCE)+'_vecnormalize.pkl','--save-interval','50',
                 '--stand-level','0','--command-sampling',arm,'--log-interval','10',
                 '--initial-state-audit',str(folder/'initial_state.json'),
                 '--tensorboard-log',str(folder/'tensorboard')]
            ev=[PYTHON,str(ROOT/'scripts/eval_new_model_training.py'),str(checkpoint),
                '--output',str(folder/'evaluation'),'--launch-speeds','.25','.5',
                '--seeds','64000','64001','64002']
            runs.append(dict(seed=seed,arm=arm,directory=str(folder),checkpoint=str(checkpoint),command=cmd,evaluate=ev))
    cfg=dict(source=str(SOURCE),protected=protected,seeds=SEEDS,arms={a:sampling_contract(a) for a in ARMS},runs=runs,
        control='same source Actor/Critic/Adam/normalization/std; same optimizer and DR/push distribution across arms',
        common_short_run='100 updates x4env x256=102400 transitions/arm/seed; batch256; lr1e-4,5epochs',
        evaluation='24s held launch,0.27m,no moving start/push,nominal; observation noise retained; original strict5% thresholds',
        expansion_gate='Only extend training speed coverage after all4low-speed posture/tracking tasks pass3/3eval seeds in at least2/3training seeds, with stand3/3 in every training seed. Then evaluate +/-1 before +/-2, keeping original thresholds.',
        adoption='Do not overwrite source or auto-select a lucky training seed. Report all arms; compare paired seed outcomes and stand regression. No automatic long training.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'experiment_config.json'
    if path.exists():
        old=json.loads(path.read_text());old.pop('frozen_utc',None)
        if old!=cfg:raise RuntimeError('Frozen configuration differs')
    else:
        cfg['frozen_utc']=datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
    lines=['#!/usr/bin/env bash','set -euo pipefail','cd '+shlex.quote(str(ROOT)),
           '# Frozen commands; no source checkpoint is overwritten.']
    for run in runs:lines.extend([shlex.join(run['command']),shlex.join(run['evaluate'])])
    (OUT/'commands.sh').write_text('\n'.join(lines)+'\n')
    return cfg

def execute(run):
    folder=Path(run['directory']);folder.mkdir(parents=True,exist_ok=True)
    # Existing successful outputs are checked, not trained over again.
    if not (folder/'training_completed.json').exists():
        if Path(run['checkpoint']+'.zip').exists():raise RuntimeError('Checkpoint exists without run completion marker')
        with (folder/'train.log').open('w') as log:
            subprocess.run(run['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        (folder/'training_completed.json').write_text(json.dumps({'checkpoint_sha256':digest(run['checkpoint']+'.zip')})+'\n')
    else:
        expected=json.loads((folder/'training_completed.json').read_text())['checkpoint_sha256']
        assert digest(run['checkpoint']+'.zip')==expected
    if not (folder/'evaluation/results.json').exists():
        with (folder/'evaluation.log').open('w') as log:
            subprocess.run(run['evaluate'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return dict(seed=run['seed'],arm=run['arm'],completed=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    cfg=prepare()
    if not args.execute:print('Frozen plan:',OUT/'experiment_config.json');return
    if not 1<=args.workers<=2:p.error('workers must be1or2 to bound CPU load')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(execute,r) for r in cfg['runs']]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    for p,h in cfg['protected'].items():assert digest(Path(p) if Path(p).is_absolute() else ROOT/p)==h
    (OUT/'execution_integrity.json').write_text(json.dumps({'protected_files_unchanged':True,'runs':len(cfg['runs'])})+'\n')

if __name__=='__main__':main()
