"""Summarize the frozen low-band LR experiment and its paired controls."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
from stable_baselines3.common.save_util import load_from_zip_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uz05.experiment_audit import fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/ppo_low_band_refinement'
OLD = ROOT / 'diagnostics/ppo_launch_sampling'
TASKS = ['stand', 'launch_p0.25', 'launch_m0.25', 'launch_p0.5', 'launch_m0.5']

def main():
    cfg = json.loads((OUT / 'experiment_config.json').read_text())
    rows = []
    initial = []
    for run in cfg['runs']:
        seed = run['seed']; folder = Path(run['folder'])
        initial.append(json.loads((folder / 'initial_state.json').read_text()))
        sources = [('lr1e-4_100', OLD / f'seed_{seed}/low_band/evaluation/results.json')]
        sources += [(f'lr3e-5_{step}', folder / f'evaluation_{step}/results.json') for step in (100, 300)]
        for arm, path in sources:
            data = json.loads(path.read_text())
            for task in data['task_metrics']:
                ep = next(e for e in data['episodes'] if e['task'] == task['task'] and e['seed'] == task['seed'])
                rows.append(dict(arm=arm, train_seed=seed, **task,
                    survived=bool(ep['success']), survival_s=ep['survival_s']))
    assert all(a == initial[0] for a in initial)
    old = json.loads((OLD / 'seed_71/low_band/initial_state.json').read_text())
    for key in ['actor_sha256', 'critic_sha256', 'reward_normalization_sha256']:
        assert initial[0][key] == old[key], key
    # The optimizer fingerprint includes param-group LR, the intended intervention.
    _, params, _ = load_from_zip_file(cfg['source'] + '.zip', device='cpu')
    optimizer = params['policy.optimizer']
    assert fingerprint(optimizer) == old['optimizer_sha256']
    for group in optimizer['param_groups']:
        group['lr'] = 0.00003
    assert fingerprint(optimizer) == initial[0]['optimizer_sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in cfg['protected'].items())
    df = pd.DataFrame(rows)
    assert len(df) == 135
    df.to_csv(OUT / 'task_metrics.csv', index=False)
    agg = df.groupby(['arm','task'], sort=False).agg(n=('seed','size'),
        survived=('survived','sum'), posture=('posture_success','sum'),
        tracking=('tracking_success','sum'), combined=('composite_success','sum'),
        mean_survival_s=('survival_s','mean'), steady_vx=('steady_mean_vx_m_s','mean'),
        steady_rmse=('steady_rmse_m_s','mean'))
    agg.to_csv(OUT / 'summary.csv')
    per = df.groupby(['arm','train_seed','task'], sort=False).agg(n=('seed','size'),
        survived=('survived','sum'), combined=('composite_success','sum'),
        mean_survival_s=('survival_s','mean'), steady_rmse=('steady_rmse_m_s','mean'))
    per.to_csv(OUT / 'per_training_seed.csv')
    gates = {}
    for arm in df.arm.unique():
        passing = []; stand_ok = []
        for seed in (71,72,73):
            a = df[(df.arm == arm)&(df.train_seed == seed)]
            stand_ok.append(int(a[a.task == 'stand'].composite_success.sum()) == 3)
            if int(a[a.task != 'stand'].composite_success.sum()) == 12: passing.append(seed)
        gates[arm] = dict(low_all_pass_training_seeds=passing, stand_all_training_seeds=all(stand_ok),
            expand_to_1m_s=len(passing)>=2 and all(stand_ok))
    (OUT / 'expansion_gate.json').write_text(json.dumps(gates,indent=2)+'\n')
    (OUT / 'verification.json').write_text(json.dumps(dict(
        protected_unchanged=True, paired_source_states_equal=True, evaluation_records=len(df),
        new_evaluation_episodes=90, new_training_transitions=921600),indent=2)+'\n')
    print(agg.to_string()); print(json.dumps(gates,indent=2))

if __name__ == '__main__': main()
