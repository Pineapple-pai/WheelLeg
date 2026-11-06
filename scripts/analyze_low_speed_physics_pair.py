"""Audit matched frozen-policy low-speed scans and separate plant/PPO effects."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import torch
from train_uz05 import TrainingPPO
from uz05.experiment_audit import fingerprint
from review_completed_transfer import read_verified

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/low_speed_physics_pair_20261106'
VERSION = 'joint_transfer_armature001_20261106_095202'

def main():
    torch.set_num_threads(1)
    files = {
        'legacy': OUT/'legacy/results.json',
        'current': OUT/'current/results.json',
        'iter25': ROOT/f'diagnostics/{VERSION}/iter_25_evaluation/results.json',
        'final50': ROOT/f'diagnostics/{VERSION}/evaluation/results.json',
    }
    data = {name: read_verified(path) for name, path in files.items()}
    reference = data['current']['config']
    for d in data.values():
        for key in ('tasks', 'seeds', 'schedules', 'thresholds', 'domain_randomization',
                    'no_learning', 'observation_noise', 'moving_starts', 'pushes', 'duration_s'):
            assert d['config'][key] == reference[key], key
    policies = {}
    for name in ('legacy', 'current'):
        cp = ROOT/data[name]['config']['checkpoint']
        policies[name] = fingerprint(TrainingPPO.load(str(cp), device='cpu').policy.state_dict())
        with Path(str(cp)+'_vecnormalize.pkl').open('rb') as f:
            norm = pickle.load(f)
        assert not norm.norm_obs, 'Observation normalization must be identical or audited separately'
    assert policies['legacy'] == policies['current'], 'Policy parameters changed between plants'
    old = ROOT/'uz05_修正后(3).xml'
    new = ROOT/'models/uz05_20261105_armature001/uz05_rl.xml'
    for name, path in [('legacy', old), ('current', new), ('iter25', new), ('final50', new)]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == data[name]['config']['model_xml_sha256']
    # Fresh current scan must reproduce the historical source before attributing PPO deltas.
    previous = read_verified(ROOT/f'diagnostics/{VERSION}_source_evaluation/results.json')
    assert data['current']['segments'] == previous['segments']
    assert data['current']['task_metrics'] == previous['task_metrics']
    for trajectory in (OUT/'current').glob('*.npz'):
        historical = ROOT/f'diagnostics/{VERSION}_source_evaluation'/trajectory.name
        with np.load(trajectory) as a, np.load(historical) as b:
            assert set(a.files) == set(b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    records = []
    for name, d in data.items():
        for r in d['segments']:
            if r['end_s'] != 24:
                continue
            score = next(s for s in d['task_metrics'] if s['task']==r['task'] and s['seed']==r['seed'])
            records.append(dict(snapshot=name, **r, posture_success=score['posture_success'],
                                composite_success=score['composite_success']))
    df = pd.DataFrame(records)
    df.to_csv(OUT/'per_episode.csv', index=False)
    columns = ['steady_vx', 'target_equivalent_vx', 'wheel_equivalent_vx',
               'steady_speed_rmse', 'steady_wheel_sat', 'steady_leg_sat',
               'posture_success', 'composite_success']
    agg = df.groupby(['snapshot','vx_goal'])[columns].mean()
    agg.to_csv(OUT/'per_speed.csv')
    fits = {}
    for name in data:
        z = df[(df.snapshot==name)&(df.vx_goal!=0)]
        assert len(z)==18 and z.complete.all() and z[columns].notna().all().all()
        fits[name] = {}
        for key in ('target_equivalent_vx', 'wheel_equivalent_vx', 'steady_vx'):
            x=z.vx_goal.to_numpy(); y=z[key].to_numpy(); gain,bias=np.polyfit(x,y,1)
            fits[name][key] = dict(gain=float(gain), bias=float(bias),
                r_squared=float(1-np.sum((y-gain*x-bias)**2)/np.sum((y-y.mean())**2)))
    pairs=[]
    for a,b,label in [('legacy','current','plant'),('current','iter25','ppo25'),('current','final50','ppo50')]:
        aa=df[df.snapshot==a].set_index(['task','seed'])
        bb=df[df.snapshot==b].set_index(['task','seed'])
        for key in aa.index:
            r,s=aa.loc[key],bb.loc[key]
            pairs.append(dict(effect=label, task=key[0], seed=int(key[1]), goal=r.vx_goal,
                body_delta=s.steady_vx-r.steady_vx,
                target_delta=s.target_equivalent_vx-r.target_equivalent_vx,
                wheel_delta=s.wheel_equivalent_vx-r.wheel_equivalent_vx,
                rmse_delta=s.steady_speed_rmse-r.steady_speed_rmse))
    pd.DataFrame(pairs).to_csv(OUT/'paired_deltas.csv',index=False)
    verification=dict(policy_parameters_identical=True, policy_fingerprint=policies['legacy'],
        observation_normalization_disabled=True, checkpoint_hashes_verified=True,
        protocol_matched=True, fresh_current_exactly_reproduces_historical_source=True,
        new_evaluation_episodes=42, reused_candidate_episodes=42, no_training=True,
        inputs={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in files.items()})
    (OUT/'analysis.json').write_text(json.dumps(dict(verification=verification,fits=fits),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name in data:
        z=agg.loc[name]; z=z[z.index!=0]
        for ax,key in zip(axes,('target_equivalent_vx','steady_vx','steady_speed_rmse')):
            ax.plot(z.index,z[key],'-o',label=name)
    for ax in axes[:2]:
        ax.plot([-.5,.5],[-.5,.5],'k--',lw=1)
    axes[2].plot([-.5,-.25,-.1,.1,.25,.5],[.025,.0125,.005,.005,.0125,.025],'k--',label='5% steady bound')
    for ax,title in zip(axes,('Policy wheel target','Body velocity','Steady velocity RMSE')):
        ax.set(title=title,xlabel='Command (m/s)',ylabel='m/s');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    print(agg.to_string())
    print(json.dumps(fits,indent=2))

if __name__=='__main__':
    main()
