"""Single-factor leg exploration test; the stage keeps wheel std at0.02."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
from initialize_model_transfer import digest
from run_startup_precision_ab import execute

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/armature001_fine_exploration'

def main():
    OUT.mkdir(exist_ok=True)
    old = json.loads((ROOT / 'diagnostics/armature001_transfer_precision_ab/experiment_config.json').read_text())
    runs = []
    for r in old['runs']:
        if r['arm'] != 'precision_reset':
            continue
        seed = r['seed']
        folder = OUT / f'seed_{seed}'
        cp = ROOT / f'checkpoints/armature001_fine_std_s{seed}_20261106/checkpoint'
        cmd = r['command'].copy()
        for key, value in {'--version': cp.parent.name, '--action-std': '.002',
                           '--initial-state-audit': str(folder/'initial_state.json'),
                           '--tensorboard-log': str(folder/'tensorboard')}.items():
            cmd[cmd.index(key)+1] = value
        ev = r['evaluate'].copy()
        ev[2] = str(cp)
        ev[ev.index('--output')+1] = str(folder/'evaluation')
        runs.append(dict(seed=seed, arm='fine_std', directory=str(folder),
                         checkpoint=str(cp), command=cmd, evaluate=ev))
    cfg = dict(runs=runs, protected=old['protected'],
               intervention='Only initial four leg exploration std0.02->0.002; stage wheel std remains0.02. Same mean Actor/Critic/Adam/reset rewardRMS, seeds, precision objective and8x256 samples.',
               budget='3x50x8x256=307200transitions;63matched evaluation episodes',
               gate='Retain standing/bidirectional posture; require benefit across training seeds, report all. Not a runtime action filter.')
    p = OUT/'experiment_config.json'
    if p.exists():
        previous=json.loads(p.read_text())
        if previous.get('intervention','').startswith('Only initial six-dimensional'):
            (OUT/'plan_wording_correction.json').write_text(json.dumps(dict(
                original_intervention=previous['intervention'], actual_intervention=cfg['intervention'],
                reason='Existing stage wheel std override is0.02; no command/code/data changed. Initial-state audits record actual scales.'),indent=2)+'\n')
            previous['intervention']=cfg['intervention']
            assert previous == cfg
            p.write_text(json.dumps(cfg,indent=2)+'\n')
        assert json.loads(p.read_text()) == cfg
    else:
        p.write_text(json.dumps(cfg, indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(execute, r) for r in runs]):
            f.result()
    assert all(digest(p) == h for p,h in cfg['protected'].items())
    (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True, new_training_transitions=307200))+'\n')

if __name__ == '__main__':
    main()
