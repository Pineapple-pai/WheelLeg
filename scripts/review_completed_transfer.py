"""Review a completed bounded transfer without changing acceptance thresholds."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from analyze_precision_completed import rows, summary
from check_transfer_progress import evaluate_progress

ROOT = Path(__file__).resolve().parents[1]

def read_verified(path):
    data = json.loads(path.read_text())
    cp = Path(data['config']['checkpoint'])
    if not cp.is_absolute():
        cp = ROOT / cp
    for suffix, expected in data['config']['protected'].items():
        assert hashlib.sha256(Path(str(cp)+suffix).read_bytes()).hexdigest() == expected
    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version')
    args = parser.parse_args()
    if Path(args.version).name != args.version:
        parser.error('version must be a single directory name')
    out = ROOT / 'diagnostics' / args.version
    files = {'source': ROOT/'diagnostics'/(args.version+'_source_evaluation')/'results.json'}
    early = out/'iter_25_evaluation/results.json'
    if early.exists():
        files['iter25'] = early
    files['final50'] = out/'evaluation/results.json'
    datasets = {name: read_verified(path) for name, path in files.items()}
    source = datasets['source']
    gates = {name: evaluate_progress(source, data) for name, data in datasets.items() if name != 'source'}
    for name, gate in gates.items():
        (out/f'{name}_review_gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    original_gate = json.loads((out/'progress_gate.json').read_text())
    assert original_gate == gates['final50'], 'Recomputed gate differs from recorded launcher result'
    records = [r for name, path in files.items() for r in rows(path, snapshot=name)]
    df = pd.DataFrame(records)
    df.to_csv(out/'review_task_metrics.csv', index=False)
    agg = summary(df, ['snapshot', 'task'])
    agg.to_csv(out/'review_summary.csv')
    fits = {}
    for name, data in datasets.items():
        seg = [r for r in data['segments'] if r['complete'] and r['end_s']==24 and r['vx_goal'] != 0]
        assert len(seg) == 18
        fits[name] = {}
        x = np.array([r['vx_goal'] for r in seg])
        for key in ('steady_vx', 'target_equivalent_vx', 'wheel_equivalent_vx'):
            y = np.array([r[key] for r in seg])
            gain, bias = np.polyfit(x, y, 1)
            fits[name][key] = dict(gain=float(gain), bias_m_s=float(bias),
                r_squared=float(1-((y-gain*x-bias)**2).sum()/((y-y.mean())**2).sum()))
    motion = df[df.task != 'stand']
    ordinary = motion.groupby('snapshot',sort=False).steady_rmse_m_s.mean().to_dict()
    result = dict(version=args.version, input_checkpoint_hashes_verified=True, gates=gates,
                  mean_motion_rmse_m_s=ordinary, steady_affine_fits=fits,
                  interpretation='Post-hoc snapshot diagnosis; unchanged5% criterion. No automatic checkpoint selection, no controller compensation.')
    (out/'completed_review.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    goals = np.array([.1,-.1,.25,-.25,.5,-.5]); positions=np.arange(len(goals))
    for i, name in enumerate(files):
        z = motion[motion.snapshot==name].groupby('task').mean(numeric_only=True)
        tasks = [f'launch_{"p" if g>0 else "m"}{abs(g):g}' for g in goals]
        axes[0].plot(goals,z.loc[tasks,'steady_mean_vx_m_s'],'o',label=name)
        axes[1].bar(positions+(i-(len(files)-1)/2)*.24,z.loc[tasks,'steady_rmse_m_s'].to_numpy()/abs(goals)*100,width=.24,label=name)
    axes[0].plot([-.5,.5],[-.5,.5],'k--',lw=1);axes[0].set(xlabel='Command (m/s)',ylabel='Steady body vx (m/s)')
    axes[1].axhline(5,color='red',ls='--',label='5% steady RMSE bound')
    axes[1].set_xticks(positions,[f'{g:+g}' for g in goals]);axes[1].set(xlabel='Command (m/s)',ylabel='Steady RMSE / |command| (%)')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Completed transfer: retaining balance does not establish speed acceptance')
    fig.tight_layout();fig.savefig(out/'completed_review.png',dpi=150);plt.close(fig)
    print(agg.to_string())
    print(json.dumps(dict(mean_motion_rmse=ordinary,gates={k:dict(accepted=v['accepted_for_further_training'],improvement=v['relative_improvement'],per_goal_retained=v['per_goal_regression_within_half_band']) for k,v in gates.items()}),indent=2))

if __name__ == '__main__':
    main()
