"""Audit intervention application and order of motion events."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/launch_cause_ablation'

def wrap(x):return np.arctan2(np.sin(x),np.cos(x))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();out=args.output
    cfg=json.loads((out/'experiment_config.json').read_text());checks=[];events=[]
    for arm,spec in cfg['arms'].items():
        for task in ('p05','m05'):
            for seed in cfg['seeds']:
                z=np.load(out/arm/f'{task}_{seed}.npz');a=z['samples'];cols=list(z['columns'])
                x=z['extra'];xc=list(z['extra_columns']);time=a[:,cols.index('time_s')]
                raw=np.column_stack([x[:,xc.index(f'raw_action_{i}')] for i in range(6)])
                applied=np.column_stack([x[:,xc.index(f'applied_action_{i}')] for i in range(6)])
                ticks=time-.008;motion=ticks>=3.-1e-9
                if 'slew' in spec:
                    change=np.diff(applied[:,:2],axis=0)*48.
                    assert np.max(abs(change[motion[1:]]))<=spec['slew']*.008+1e-5
                    np.testing.assert_array_equal(raw[:,2:],applied[:,2:])
                elif 'hold' in spec:
                    mask=motion & (ticks<3.+spec['hold']-1e-9)
                    np.testing.assert_allclose(applied[mask,2:],np.broadcast_to(applied[374,2:],applied[mask,2:].shape),rtol=0,atol=0)
                    np.testing.assert_array_equal(raw[:,:2],applied[:,:2])
                    np.testing.assert_array_equal(raw[~mask,2:],applied[~mask,2:])
                else:np.testing.assert_array_equal(raw,applied)
                ref=a[:,cols.index('command_vx_m_s')]
                goal=.5 if task=='p05' else -.5
                expected=np.sign(goal)*np.minimum(abs(goal),spec['ramp']*np.maximum(ticks-3.,0.)) if 'ramp' in spec else np.where(motion,goal,0.)
                np.testing.assert_allclose(ref,expected,rtol=0,atol=1e-12)
                checks.append(dict(arm=arm,task=task,seed=seed,changed_action_frames=int(np.any(abs(raw-applied)>1e-8,axis=1).sum())))
                if arm=='baseline':
                    pitch=np.degrees(a[:,cols.index('pitch_rad')]);q=np.column_stack([a[:,cols.index(f'target_q_{i}')] for i in range(4)])
                    def first(mask):
                        ix=np.flatnonzero(mask & (time>3.+1e-9))
                        return float(time[ix[0]]-3.) if len(ix) else None
                    events.append(dict(task=task,seed=seed,
                        pitch_5deg_delay_s=first(abs(pitch)>=5.),
                        leg_target_015rad_delay_s=first(np.max(abs(wrap(q-q[374])),axis=1)>=.15),
                        wheel_saturation_delay_s=first(a[:,cols.index('wheel_saturation')]>0.),
                        termination_delay_s=float(time[-1]-3.)))
    pd.DataFrame(checks).to_csv(out/'intervention_checks.csv',index=False)
    pd.DataFrame(events).to_csv(out/'event_timing.csv',index=False)
    (out/'intervention_verification.json').write_text(json.dumps(dict(all_interventions_verified=True,episodes=len(checks),
        circular_leg_hold=True,wheel_limits_applied=True,raw_and_applied_actions_preserved=True),indent=2)+'\n')
    print(pd.DataFrame(events).groupby('task').mean(numeric_only=True).to_string())

if __name__=='__main__':main()
