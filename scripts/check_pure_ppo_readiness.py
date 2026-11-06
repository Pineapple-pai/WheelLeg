"""Require replicated unassisted PPO progress before bounded long continuation."""
from pathlib import Path
import json,argparse
from initialize_model_transfer import digest
from review_completed_transfer import read_verified
from check_transfer_progress import evaluate_progress
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/pure_ppo_precision_20261106'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipe',default='reward_fine',choices=['wheel_fine','reward_fine','both_fine'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[711,713,714])
    args=parser.parse_args()
    baseline=read_verified(OUT/'source_evaluation/results.json')
    evidence={}
    for seed in args.seeds:
        path=OUT/f'seed_{seed}/{args.recipe}/evaluation/results.json'
        d=read_verified(path)
        assert 'diagnostic_intervention' not in d['config']
        evidence[f'nominal_s{seed}']=evaluate_progress(baseline,d)
    dr_source=read_verified(OUT/'holdout_dr/source/results.json')
    dr_candidate=read_verified(OUT/f'holdout_dr/{args.recipe}_s711/results.json')
    assert 'diagnostic_intervention' not in dr_candidate['config']
    evidence['independent_dr_s711']=evaluate_progress(dr_source,dr_candidate)
    ready=all(r['accepted_for_further_training'] for r in evidence.values())
    cp=ROOT/f'checkpoints/pure_ppo_{args.recipe}_s711_20261106/checkpoint'
    result=dict(ready_for_bounded_long_training=ready,fully_passed_all_tasks=all(r['fully_passed_all_tasks'] for r in evidence.values()),
        evidence=evidence,approved_sources={str(cp):{s:digest(str(cp)+s) for s in ('.zip','_contract.json','_vecnormalize.pkl')}} if ready else {},
        training_seeds=args.seeds,nominal_evaluation_seeds=[66600,66601,66602],holdout_dr_seeds=[66700,66701,66702],
        recipe=args.recipe,
        inference='Unmodified PPO predict; existing actuator interface. No teacher, calibration, residual or action filter.',
        limitation='Readiness for monitored training, not convergence or hardware deployment. Small training-seed set provides limited replication.',
        continuation='continue_pure_ppo.py performs paired frozen evaluations every segment and stops on rejected progress.')
    (OUT/f'readiness_{args.recipe}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{x:v[x] for x in ('accepted_for_further_training','fully_passed_all_tasks','relative_improvement','per_goal_regression_within_half_band','regressions')} for k,v in evidence.items()},indent=2))
    print('Ready:',ready)

if __name__=='__main__':main()
