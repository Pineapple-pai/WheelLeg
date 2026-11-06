"""Explicit evaluation-only 2x2 command-input calibration; never a PPO checkpoint."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/speed_calibration_diagnostic_20261106'
FIT=ROOT/'diagnostics/low_speed_physics_pair_20261106/analysis.json'
SOURCE=ROOT/'checkpoints/armature001_transfer_legacy_20261106/checkpoint'
ARMS=('control','bias','gain','both')

def coefficients():
    f=json.loads(FIT.read_text())['fits']['current']['steady_vx']
    return 1/f['gain'], -f['bias']/f['gain']

def transform(obs,gain,bias):
    from uz05.spec import ACTOR_OBS_BLOCKS, OBS_SCALE
    offset=0
    for name,width in ACTOR_OBS_BLOCKS:
        if name=='command':break
        offset+=width
    import numpy as np
    scale=float(np.broadcast_to(np.asarray(OBS_SCALE['command']), (5,))[0])
    value=obs[offset]/scale
    # Zero command invokes a separate standing policy regime; never offset it.
    if abs(value)>1e-8:
        obs[offset]=(gain*value+bias)*scale
    return obs

def worker(arm,dr):
    from eval_new_model_training import main
    g,b=coefficients();gain=g if arm in ('gain','both') else 1.;bias=b if arm in ('bias','both') else 0.
    folder=OUT/('holdout_dr' if dr else 'nominal')/arm
    seeds=['66500','66501','66502'] if dr else ['66400','66401','66402']
    sys.argv=['eval_new_model_training.py',str(SOURCE),'--output',str(folder),
              '--launch-speeds','.1','.25','.5','--seeds',*seeds]+(['--domain-randomization'] if dr else [])
    metadata=dict(arm=arm,gain=gain,bias_m_s=bias,fit_source=str(FIT),
        fit_sha256=hashlib.sha256(FIT.read_bytes()).hexdigest(),zero_command='unchanged',
        operation='Policy observation vx only; physical command, scoring and all actuator settings unchanged',
        evaluation_only=True,not_a_trainable_candidate=True)
    main(diagnostic_transform=lambda obs:transform(obs,gain,bias),diagnostic_metadata=metadata)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=ARMS)
    p.add_argument('--dr',action='store_true');a=p.parse_args()
    if a.worker:
        worker(a.worker,a.dr);return
    OUT.mkdir(exist_ok=True)
    plan=dict(coefficients=coefficients(),fit_source_sha256=hashlib.sha256(FIT.read_bytes()).hexdigest(),
        source=str(SOURCE),arms=ARMS,nominal_seeds=[66400,66401,66402],dr_seeds=[66500,66501,66502],
        gate='All 21 composite successes, original 5% band, 95% occupancy and response deadline. No mean-only acceptance.',
        next='Only if nominal both arm fully passes, run independent DR; no long-training claim from diagnostic wrapper.')
    frozen=OUT/'plan.json'
    if frozen.exists():assert json.loads(frozen.read_text())==json.loads(json.dumps(plan))
    else:frozen.write_text(json.dumps(plan,indent=2)+'\n')
    arms=('control','both') if a.dr else ARMS
    def run(arm):
        mode='holdout_dr' if a.dr else 'nominal';folder=OUT/mode/arm
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/'results.json').exists():raise FileExistsError(folder)
        with (folder/'evaluation.log').open('w') as stream:
            subprocess.run([sys.executable,__file__,'--worker',arm]+(['--dr'] if a.dr else []),
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('Completed',mode,arm,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run,arms))

if __name__=='__main__':main()
