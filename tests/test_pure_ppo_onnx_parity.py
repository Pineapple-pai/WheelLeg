from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

from uz05.policy import AsymmetricActorCriticPolicy
from uz05.pure_ppo_stand import ACTOR_OBS_DIM, ACTION_DIM, OBS_DIM


class _DeterministicActor(nn.Module):
    def __init__(self, policy: ActorCriticPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, actor_obs):
        latent = self.policy.mlp_extractor.forward_actor(actor_obs)
        mean = self.policy.action_net(latent)
        return torch.clamp(mean, -1.0, 1.0)


class PurePPOOnnxParityTests(unittest.TestCase):
    def test_random_policy_actor_matches_onnx_for_1000_observations(self):
        torch.manual_seed(42)
        policy = AsymmetricActorCriticPolicy(
            observation_space=spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32),
            action_space=spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32),
            lr_schedule=lambda _: 1e-4,
            actor_dim=ACTOR_OBS_DIM,
            net_arch={"pi": [32, 16], "vf": [32, 16]},
            activation_fn=nn.ELU,
            log_std_init=-1.5,
        )
        policy.set_training_mode(False)
        actor = _DeterministicActor(policy).eval()
        observations = np.random.default_rng(5).normal(size=(1000, OBS_DIM)).astype(np.float32)
        expected, _ = policy.predict(observations, deterministic=True)

        with tempfile.TemporaryDirectory(prefix="pure_ppo_onnx_test_") as tmp:
            path = Path(tmp) / "actor.onnx"
            torch.onnx.export(
                actor,
                torch.from_numpy(observations[:1, :ACTOR_OBS_DIM]),
                str(path),
                input_names=["actor_observation"],
                output_names=["action"],
                dynamic_axes={"actor_observation": {0: "batch"}, "action": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True,
            )
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            actual = session.run(
                ["action"], {"actor_observation": observations[:, :ACTOR_OBS_DIM]}
            )[0]
        abs_error = np.abs(expected - actual)
        self.assertLessEqual(float(abs_error.max()), 1e-4)
        self.assertLessEqual(float(abs_error.mean()), 1e-5)


if __name__ == "__main__":
    unittest.main()
