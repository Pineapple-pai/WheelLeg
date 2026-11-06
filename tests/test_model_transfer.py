"""Cross-model initialization must be explicit, provenance-bearing and non-destructive."""
import json,shutil,sys,unittest,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from initialize_model_transfer import initialize,validate_interface,digest
from train_uz05 import TrainingPPO,require_compatible_contract
from uz05.experiment_audit import fingerprint
from uz05.spec import REPO_ROOT,MODEL_XML
SOURCE=REPO_ROOT/'checkpoints/joint_tracking_pos_v16/checkpoint'
class ModelTransferTests(unittest.TestCase):
 def test_fresh_optimizer_keeps_policy_and_declares_reset(self):
  version='model_transfer_test_'+uuid.uuid4().hex
  try:
   target=initialize(SOURCE,version,True,True)
   m=TrainingPPO.load(str(target),device='cpu');old=TrainingPPO.load(str(SOURCE),device='cpu')
   self.assertEqual(fingerprint(m.policy.state_dict()),fingerprint(old.policy.state_dict()))
   self.assertFalse(m.policy.optimizer.state_dict()['state'])
   audit=json.loads(Path(str(target)+'_initialization.json').read_text())
   self.assertTrue(audit['reset_optimizer']);self.assertNotEqual(audit['source_optimizer_sha256'],audit['optimizer_sha256'])
  finally:
   folder=REPO_ROOT/'checkpoints'/version
   if folder.exists():shutil.rmtree(folder)
 def test_requires_explicit_operation_and_rejects_interface_changes(self):
  with self.assertRaisesRegex(ValueError,'Explicit'):initialize(SOURCE,'unused_transfer_test')
  source=json.loads(Path(str(SOURCE)+'_contract.json').read_text());bad=json.loads(json.dumps(source));bad['observation_scale']['base_lin_vel']*=2
  with self.assertRaisesRegex(ValueError,'observation_scale'):validate_interface(source,bad)
  bad=json.loads(json.dumps(source));bad['actuator_config']['wheel']['efficiency']=1.
  with self.assertRaisesRegex(ValueError,'actuator_config'):validate_interface(source,bad)
 def test_new_artifact_keeps_parameters_and_source_guard(self):
  version='model_transfer_test_'+uuid.uuid4().hex
  protected={s:digest(str(SOURCE)+s) for s in ('.zip','_contract.json','_vecnormalize.pkl')}
  try:
   target=initialize(SOURCE,version,True);a=json.loads(Path(str(target)+'_initialization.json').read_text());c=json.loads(Path(str(target)+'_contract.json').read_text())
   m=TrainingPPO.load(str(target),device='cpu');old=TrainingPPO.load(str(SOURCE),device='cpu')
   self.assertEqual(fingerprint(m.policy.state_dict()),fingerprint(old.policy.state_dict()))
   self.assertEqual(fingerprint(m.policy.optimizer.state_dict()),fingerprint(old.policy.optimizer.state_dict()))
   self.assertEqual(m.num_timesteps,0);self.assertEqual(c['total_updates'],0)
   self.assertEqual(c['model_xml_sha256'],digest(MODEL_XML));self.assertNotEqual(c['model_xml_sha256'],a['source_model_sha256'])
   require_compatible_contract(target,'joint_tracking_startup')
   with self.assertRaisesRegex(SystemExit,'model_xml_sha256'):require_compatible_contract(SOURCE,'joint_tracking_startup')
   self.assertEqual(protected,{s:digest(str(SOURCE)+s) for s in protected})
  finally:
   folder=REPO_ROOT/'checkpoints'/version
   if folder.exists():shutil.rmtree(folder)
if __name__=='__main__':unittest.main()
