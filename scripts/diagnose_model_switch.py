"""Frozen-policy factorial model-switch diagnosis; XML variants are diagnostic only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import hashlib,json,subprocess,sys,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/model_switch_causes'
OLD=ROOT/'uz05_修正后(3).xml';NEW=ROOT/'models/uz05_20261104/uz05_rl.xml'
SOURCE=ROOT/'checkpoints/joint_tracking_pos_v16/checkpoint'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True);variants=OUT/'variants';variants.mkdir(exist_ok=True)
 old=ET.parse(OLD).getroot();new=ET.parse(NEW).getroot();body=old.find(".//body[@name='gimbal_fixed']/inertial")
 paths={'legacy':OLD,'new':NEW}
 for name,arm,gimbal in [('new_old_armature',True,False),('new_old_gimbal',False,True),('new_old_both',True,True)]:
  r=deepcopy(new)
  if arm:
   for side in ('left','right'):r.find(f".//joint[@name='{side}_J_link6_wheel']").set('armature','0.0001')
  if gimbal:
   b=r.find(".//body[@name='gimbal_fixed']");b.remove(b.find('inertial'));b.insert(0,deepcopy(body))
  data=ET.tostring(r,encoding='utf-8',xml_declaration=True);p=variants/f'{name}.xml'
  if p.exists():assert p.read_bytes()==data
  else:p.write_bytes(data)
  paths[name]=p
 protected=[OLD,NEW,*[Path(str(SOURCE)+s) for s in ('.zip','_vecnormalize.pkl','_contract.json')],ROOT/'scripts/validate_new_model.py',*[ROOT/f'scripts/uz05/{s}.py' for s in ('model','env','spec','actuators')]]
 cfg=dict(source=str(SOURCE),protected={str(p):digest(p) for p in protected},variants={k:dict(path=str(v),sha256=digest(v)) for k,v in paths.items()},
  protocol='Same frozen legacy policy; stand,+2,-2,reverse-stop,motion; seeds51000..51002; nominal24s with observation noise. No training. Survival is not tracking acceptance.',
  intervention='2x2 new/legacy wheel armature and gimbal inertial block, all other new settings retained. Gimbal block includes both COM and inertia; no unique intra-block attribution.',
  limits='Policy-specific diagnostic, not proof of physical correctness or retraining convergence. Legacy full model provides reference including solver/contact differences.')
 p=OUT/'experiment_config.json'
 if p.exists():assert json.loads(p.read_text())==cfg
 else:p.write_text(json.dumps(cfg,indent=2)+'\n')
 def run(item):
  name,path=item;folder=OUT/name;result=folder/'results.json'
  if not result.exists():
   env=os.environ.copy();env['UZ05_MODEL_XML']=str(path);env['UZ05_MESH_DIR']=str(ROOT/'models/uz05_20261104/meshes/stl')
   with (OUT/f'{name}.log').open('w') as stream:subprocess.run([sys.executable,str(ROOT/'scripts/validate_new_model.py'),'--output',str(folder),'--checkpoint',str(SOURCE)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
  d=json.loads(result.read_text());assert d['model_sha256']==digest(path);assert d['checkpoint_sha256']==digest(str(SOURCE)+'.zip');print('Completed',name,flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(run,pair) for pair in paths.items()]):f.result()
 assert all(digest(p)==h for p,h in cfg['protected'].items())
 (OUT/'integrity.json').write_text(json.dumps(dict(protected_unchanged=True,no_training=True,episodes=75))+'\n')
if __name__=='__main__':main()
