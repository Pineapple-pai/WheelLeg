"""One bounded continuation of the preselected pure-PPO fine-reward policy."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys
from pathlib import Path
from run_startup_precision_ab import execute
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/pure_ppo_precision_20261106/continuation_s711'
SOURCE=ROOT/'checkpoints/pure_ppo_reward_fine_s711_20261106/checkpoint'
CP=ROOT/'checkpoints/pure_ppo_reward_cont_s711_20261106/checkpoint'

def main():
 OUT.mkdir(exist_ok=True)
 if CP.parent.exists():raise FileExistsError(CP.parent)
 command=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),'--stage','joint_tracking_startup_precision_fine',
  '--command-sampling','startup_balanced','--checkpoint',str(SOURCE),'--preserve-action-std',
  '--vecnormalize-path',str(SOURCE)+'_vecnormalize.pkl','--version',CP.parent.name,'--updates','200',
  '--rollout-steps','512','--batch-size','512','--num-envs','8','--vec-env','subproc','--device','cpu',
  '--learning-rate','.00001','--disable-adaptive-lr','--entropy-coef','0','--seed','1711','--save-interval','50','--log-interval','25',
  '--initial-state-audit',str(OUT/'initial_state.json'),'--tensorboard-log',str(OUT/'tensorboard')]
 ev=[sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),str(CP),'--output',str(OUT/'evaluation'),
     '--launch-speeds','.1','.25','.5','--seeds','66600','66601','66602']
 run=dict(seed=1711,arm='continue_reward_fine',directory=str(OUT),checkpoint=str(CP),command=command,evaluate=ev)
 (OUT/'plan.json').write_text(json.dumps(dict(run=run,source=str(SOURCE),
    purpose='Test whether progress persists over200additional PPO updates before suggesting a longer command. All six learned stds,Adam,RMS retained; fresh environment seed.',
    acceptance='Compare to both the200-update policy and original initialization. Retain all source successes, no per-goal regression beyond half a5%band; no diagnostic wrappers.'),indent=2)+'\n')
 execute(run)

if __name__=='__main__':main()
