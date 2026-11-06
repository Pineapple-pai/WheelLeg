"""Analyze paired model-switch trials without equating survival with tracking."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/model_switch_causes'
def main():
 cfg=json.loads((OUT/'experiment_config.json').read_text());rows=[]
 for variant,entry in cfg['variants'].items():
  d=json.loads((OUT/variant/'results.json').read_text())
  assert len(d['episodes'])==15 and d['no_training']
  assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
  rows.extend(dict(variant=variant,**r,survival_s=r['steps']*.008) for r in d['episodes'])
 df=pd.DataFrame(rows);assert len(df)==75
 df.to_csv(OUT/'episodes.csv',index=False)
 agg=df.groupby(['variant','task'],sort=False).agg(n=('seed','size'),survived=('survived_24s','sum'),mean_survival_s=('survival_s','mean'),steady_vx_survivors=('final2_vx_m_s','mean'),pitch_rms_deg=('pitch_rms_deg','mean'))
 agg.to_csv(OUT/'summary.csv');print(agg.to_string())
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in cfg['protected'].items())
 for name in ['new','legacy']:
  for path in (OUT/name).glob('*.npz'):
   np.testing.assert_array_equal(np.load(path)['samples'],np.load(ROOT/'diagnostics/new_model_validation'/name/path.name)['samples'])
 (OUT/'verification.json').write_text(json.dumps(dict(no_training=True,protected_unchanged=True,baseline_historical_exact=True,episodes=75),indent=2)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
 for j,task in enumerate(['forward2','reverse2']):
  for variant,label in [('legacy','Legacy'),('new','New'),('new_old_armature','New + legacy armature'),('new_old_gimbal','New + legacy gimbal')]:
   a=np.load(OUT/variant/f'{task}_51000.npz')['samples']
   axes[0,j].plot(a[:,0],a[:,3],label=label,lw=1.2);axes[1,j].plot(a[:,0],np.degrees(a[:,1]),lw=1.2)
  for ax in axes[:,j]:ax.set_xlim(2.8,5.5);ax.axvline(3,color='gray',ls=':');ax.grid(alpha=.25)
  axes[0,j].axhline(2 if j==0 else -2,color='k',ls='--',lw=.8)
  axes[0,j].set_title('0 to +2 m/s' if j==0 else '0 to -2 m/s');axes[1,j].set_xlabel('Time (s)')
 axes[0,0].set_ylabel('Body vx (m/s)');axes[1,0].set_ylabel('Pitch (deg)');axes[0,1].legend(fontsize=8)
 fig.suptitle('Same frozen legacy policy, seed 51000; curves stop at failure');fig.tight_layout();fig.savefig(OUT/'model_switch_transients.png',dpi=150)
if __name__=='__main__':main()
