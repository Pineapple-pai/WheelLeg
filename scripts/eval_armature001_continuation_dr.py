"""Evaluate all predeclared continuation lineages under the same randomized plant seeds."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, sys, time
from run_startup_precision_ab import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/armature001_precision_continuation'

def run(seed):
    folder = OUT / f'seed_{seed}'
    for _ in range(120):
        if (folder / 'training_completed.json').exists():
            break
        time.sleep(5)
    else:
        raise TimeoutError(f'Continuation {seed} did not finish')
    checkpoint = ROOT / f'checkpoints/armature001_precision_cont_s{seed}_20261106/checkpoint'
    command = [sys.executable, str(ROOT / 'scripts/eval_new_model_training.py'),
               str(checkpoint), '--output', str(folder / 'evaluation_dr'),
               '--launch-speeds', '.1', '.25', '.5', '--seeds', '66200', '66201', '66202',
               '--domain-randomization']
    evaluate(command, folder / 'evaluation_dr.log')
    print('Completed randomized holdout', seed, flush=True)

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run, seed) for seed in (201, 202, 203)]):
            future.result()
    (OUT / 'holdout_completed.json').write_text(json.dumps(dict(
        lineages=[201, 202, 203], seeds=[66200, 66201, 66202], episodes=63)) + '\n')
