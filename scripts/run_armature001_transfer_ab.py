"""Paired post-transfer exploration trial on the verified0.001 plant."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,sys
from run_startup_precision_ab import execute
from initialize_model_transfer import digest
from uz05.spec import MODEL_XML
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/armature001_transfer_ab'
SOURCE=ROOT/'checkpoints/armature001_transfer_legacy_20261106/checkpoint'

def prepare(precision=False):
 out=OUT if not precision else ROOT/'diagnostics/armature001_transfer_precision_ab'
 out.mkdir(exist_ok=True);runs=[]
 arms=([('preserve','joint_tracking_startup',['--preserve-action-std']),('cap002','joint_tracking_startup',['--action-std','.02'])] if not precision else
       [('control_reset','joint_tracking_startup',['--action-std','.02','--reset-vecnormalize']),('precision_reset','joint_tracking_startup_precision',['--action-std','.02','--reset-vecnormalize'])])
 for seed in (201,202,203):
  for arm,stage,extra in arms:
   folder=out/f'seed_{seed}'/arm;cp=ROOT/'checkpoints'/f'armature001_{arm}_s{seed}_20261106'/'checkpoint'
   cmd=[sys.executable,'-u',str(ROOT/'scripts/train_uz05.py'),'--stage',stage,'--command-sampling','startup_balanced',
    '--checkpoint',str(SOURCE),'--version',cp.parent.name,'--updates','50','--rollout-steps','256','--batch-size','256',
    '--num-envs','8','--vec-env','subproc','--device','cpu','--learning-rate','.00003','--disable-adaptive-lr','--entropy-coef','.001',
    '--seed',str(seed),'--save-interval','25','--log-interval','10','--initial-state-audit',str(folder/'initial_state.json'),
    '--tensorboard-log',str(folder/'tensorboard'),*extra]
   evaluate=[sys.executable,str(ROOT/'scripts/eval_new_model_training.py'),str(cp),'--output',str(folder/'evaluation'),
    '--launch-speeds','.1','.25','.5','--seeds','66100','66101','66102']
   runs.append(dict(seed=seed,arm=arm,directory=str(folder),checkpoint=str(cp),command=cmd,evaluate=evaluate))
 protected=[Path(str(SOURCE)+s) for s in ('.zip','_vecnormalize.pkl','_contract.json','_initialization.json')]
 protected += [MODEL_XML,*[ROOT/f'scripts/uz05/{n}.py' for n in ('spec','env','actuators','command_sampling','experiment_audit')],ROOT/'scripts/train_uz05.py']
 cfg=dict(source=str(SOURCE),protected={str(p):digest(p) for p in protected},runs=runs,
  intervention=('Same transferred mean policy/Adam/rewardRMS and8x256 sampling; only inherited six stds vs explicit0.02std.' if not precision else
                'Both cap std0.02 and reset rewardRMS; only dense vx cost scale0.15->0.05 differs.'),
  budget='6x50x8x256=614400transitions; 126evaluationepisodes on frozen step goals.',
  gate='Report all3training seeds. Require retained stand and bidirectional posture across all7tasks; velocity benefit must replicate>=2seeds. No single-seed promotion or automatic expansion.')
 p=out/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==cfg
 else:p.write_text(json.dumps(cfg,indent=2)+'\n')
 return out,cfg

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--precision',action='store_true');a=p.parse_args();out,cfg=prepare(a.precision)
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(execute,r) for r in cfg['runs']]):f.result()
 assert all(digest(p)==h for p,h in cfg['protected'].items())
 (out/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True,new_training_transitions=614400))+'\n')
if __name__=='__main__':main()
