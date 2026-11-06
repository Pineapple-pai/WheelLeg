"""Requested seed, rollout and LR must survive a real checkpoint-resume run."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stable_baselines3.common.vec_env import DummyVecEnv
from uz05.env import UZ05Env
from uz05.policy import AsymmetricActorCriticPolicy
from uz05.spec import ACTOR_OBS_DIM
from train_uz05 import TrainingPPO, write_contract


class ResumeOptionsTest(unittest.TestCase):
    def test_explicit_wheel_std_overrides_stage_default(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='wheel_std_test_',dir=root/'checkpoints') as directory:
            folder=Path(directory)
            result=subprocess.run([sys.executable,str(root/'scripts/train_uz05.py'),
                '--stage','joint_tracking_startup_precision_fine','--command-sampling','startup_balanced',
                '--action-std','.02','--wheel-action-std','.003','--updates','1',
                '--rollout-steps','8','--batch-size','8','--num-envs','1','--vec-env','dummy',
                '--initial-state-audit',str(folder/'initial.json'),
                '--out',str(folder/'result'),'--tensorboard-log',str(folder/'tb')],
                cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            audit=json.loads((folder/'initial.json').read_text())
            for value in audit['action_std'][:2]:self.assertAlmostEqual(value,.003,places=7)
            for value in audit['action_std'][2:]:self.assertAlmostEqual(value,.02,places=7)
            config=json.loads((folder/'training_config.json').read_text())
            self.assertEqual(config['precision_objective']['vx_error_scale_m_s'],.01)
            self.assertTrue(config['environment_config']['observation_noise']['enabled'])

    def test_actual_cli_resume_uses_requested_configuration(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='resume_options_test_',dir=root/'checkpoints') as directory:
            folder=Path(directory)
            env=DummyVecEnv([lambda:UZ05Env('joint_tracking_v15')])
            source=TrainingPPO(AsymmetricActorCriticPolicy,env,n_steps=8,batch_size=8,
                learning_rate=.0001,seed=7,n_epochs=5,
                policy_kwargs=dict(actor_dim=ACTOR_OBS_DIM,net_arch=dict(pi=[8],vf=[8])))
            source.save(folder/'source')
            write_contract(folder/'source','joint_tracking_v15',1)
            env.close()
            result=subprocess.run([sys.executable,str(root/'scripts/train_uz05.py'),
                '--stage','joint_tracking_simple','--checkpoint',str(folder/'source'),
                '--reset-vecnormalize','--preserve-action-std','--updates','2',
                '--rollout-steps','16','--batch-size','8','--num-envs','1','--vec-env','dummy',
                '--seed','23','--learning-rate','.00003','--disable-domain-randomization','--legacy-reset',
                '--out',str(folder/'result'),'--tensorboard-log',str(folder/'tb')],
                cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            restored=TrainingPPO.load(folder/'result')
            self.assertEqual(restored.seed,23)
            self.assertEqual(restored.n_steps,16)
            self.assertEqual(restored.rollout_buffer.buffer_size,16)
            self.assertEqual(restored.batch_size,8)
            self.assertEqual(restored.num_timesteps,32)
            self.assertAlmostEqual(restored.lr_schedule(1),.00003)
            saved=json.loads((folder/'training_config.json').read_text())
            self.assertFalse(saved['environment_config']['robot']['closed_chain_reset'])
            contract=json.loads((folder/'result_contract.json').read_text())
            self.assertFalse(contract['environment_config']['robot']['closed_chain_reset'])


if __name__=='__main__':unittest.main()
