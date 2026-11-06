"""Read completed eval jobs without influencing the frozen training plan."""
import json
from pathlib import Path
import pandas as pd
from analyze_ppo_v17_torque_ab import prepare,locations,PROFILES,TASKS,THRESHOLDS,load,score_interval,OUT

def main():
    cfg=prepare();path=OUT/'progress_task_metrics.csv'
    rows=pd.read_csv(path).to_dict('records') if path.exists() else []
    keys={(r['train_seed'],r['train_model'],r['update'],r['eval_model'],r['protocol'],r['eval_seed'],r['task']) for r in rows}
    for meta,root in locations(cfg):
      for protocol in PROFILES:
        file=root/protocol/'episodes.json'
        if not file.exists():continue
        try:records=json.loads(file.read_text())
        except json.JSONDecodeError:continue
        for r in records:
          tasks=[x for x in TASKS if x[1]==protocol and (*meta.values(),protocol,r['eval_seed'],x[0]) not in keys]
          if not tasks:continue
          c=load(root/protocol/f'{r["eval_seed"]}.npz')
          for task,source,start,end,goal in tasks:
            rows.append({**meta,'protocol':protocol,'task':task,'eval_seed':r['eval_seed'],
              'cohort':'historical' if r['eval_seed']<60000 else 'independent',
              **score_interval(c,start,end,goal,THRESHOLDS,r['termination_reason'])})
    df=pd.DataFrame(rows);df.to_csv(path,index=False)
    g=df.groupby(['train_seed','train_model','update','eval_model','cohort','task']).agg(n=('eval_seed','count'),posture=('posture_success','mean'),tracking=('tracking_success','mean'),vx=('steady_mean_vx_m_s','mean')).reset_index()
    g.to_csv(OUT/'progress_summary.csv',index=False)
    print(g[(g['update']>0)&(g.cohort=='historical')&g.task.isin(['forward2','reverse2','reverse_stop','high_reversal'])].to_string(index=False))
if __name__=='__main__':main()
