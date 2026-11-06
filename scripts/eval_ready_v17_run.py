"""Evaluate one completed frozen run while later independent runs train."""
import argparse
import json
from pathlib import Path
from run_ppo_v17_torque_ab import prepare,OUT,MILESTONES,OLD_SEEDS,NEW_SEEDS,PROFILES,evaluate_jobs,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-index',type=int,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--updates',type=int,nargs='+',choices=MILESTONES,default=MILESTONES);a=p.parse_args()
    cfg=prepare();run=cfg['runs'][a.run_index];assert (Path(run['directory'])/'completed.json').exists();verify(cfg);jobs=[]
    for update in a.updates:
      checkpoint=run['checkpoint']+(f'_iter_{update}' if update<200 else '')
      for model in (['A_legacy','B_rated'] if update==200 else ['B_rated']):
        for protocol in PROFILES:
          dest=OUT/'evaluation'/f'seed_{run["seed"]}'/run['mode']/f'update_{update}'/model/protocol
          seeds=OLD_SEEDS+NEW_SEEDS if update==200 else OLD_SEEDS
          jobs.append((checkpoint,model,protocol,seeds,str(dest)))
    evaluate_jobs(jobs,a.workers);verify(cfg)
    (Path(run['directory'])/('early_evaluation_complete.json' if a.updates==MILESTONES else 'early_evaluation_updates_'+'_'.join(map(str,a.updates))+'_complete.json')).write_text(json.dumps({'complete':True,'updates':a.updates,'jobs':len(jobs)},indent=2)+'\n')
if __name__=='__main__':main()
