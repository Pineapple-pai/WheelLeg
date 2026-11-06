"""Only the reward-identical v16/startup transition may inherit reward RMS."""
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_uz05 import can_preserve_reward_statistics, TrainingPPO
from uz05.spec import STAGE_BY_NAME, EnvParams, precision_rewards_for_stage
from uz05.experiment_audit import fingerprint

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'checkpoints/joint_tracking_pos_v16_model20261104/checkpoint'

class StartupWarmstartTests(unittest.TestCase):
    def test_guard_checks_reward_identity_and_keeps_other_transitions_blocked(self):
        contract=dict(stage='joint_tracking_v16',precision_objective=asdict(
            precision_rewards_for_stage(STAGE_BY_NAME['joint_tracking_v16'],EnvParams())))
        self.assertTrue(can_preserve_reward_statistics(contract,'joint_tracking_startup'))
        self.assertTrue(can_preserve_reward_statistics(json.loads(json.dumps(contract)),'joint_tracking_startup'))
        contract['precision_objective']['effort']+=1.
        self.assertFalse(can_preserve_reward_statistics(contract,'joint_tracking_startup'))
        contract['stage']='joint_tracking_v15'
        self.assertFalse(can_preserve_reward_statistics(contract,'joint_tracking_startup'))
        self.assertFalse(can_preserve_reward_statistics(contract,'joint_tracking_simple'))

    def test_real_cli_preserves_source_before_learning(self):
        with tempfile.TemporaryDirectory(prefix='startup_resume_test_',dir=ROOT/'checkpoints') as directory:
            folder=Path(directory)
            command=[sys.executable,str(ROOT/'scripts/train_uz05.py'),
                '--stage','joint_tracking_startup','--command-sampling','startup_balanced',
                '--checkpoint',str(SOURCE),'--preserve-action-std',
                '--out',str(folder/'result'),'--initial-state-audit',str(folder/'initial.json'),
                '--updates','1','--rollout-steps','8','--batch-size','8','--num-envs','1','--vec-env','dummy',
                '--learning-rate','.0001','--entropy-coef','.001','--tensorboard-log',str(folder/'tb')]
            # This fixture is a historical0.01 checkpoint, not the new default plant.
            process_env=os.environ.copy()
            process_env['UZ05_MODEL_XML']=str(ROOT/'models/uz05_20261104/uz05_rl.xml')
            result=subprocess.run(command,cwd=ROOT,env=process_env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            audit=json.loads((folder/'initial.json').read_text())
            source=TrainingPPO.load(str(SOURCE),device='cpu')
            self.assertEqual(fingerprint(source.policy.state_dict()),audit['policy_sha256'])
            self.assertEqual(fingerprint(source.policy.optimizer.state_dict()),audit['optimizer_sha256'])
            with Path(str(SOURCE)+'_vecnormalize.pkl').open('rb') as stream:vec=pickle.load(stream)
            self.assertEqual(fingerprint(vars(vec.ret_rms)),audit['reward_normalization_sha256'])
            self.assertFalse(audit['observation_normalization_enabled'])
            saved=json.loads((folder/'training_config.json').read_text())
            self.assertFalse(saved['arguments']['reset_vecnormalize'])
            self.assertTrue(saved['arguments']['preserve_action_std'])

if __name__=='__main__':unittest.main()
