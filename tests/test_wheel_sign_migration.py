from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_uz05 import AsymmetricActorCriticPolicy
from uz05.policy_compat import ensure_forward_positive_wheel_actions
from uz05.spec import ACTOR_OBS_DIM, ACTION_DIM, OBS_DIM, OBS_SLICES, WHEEL_ACTION_FRAME


class WheelSignMigrationTest(unittest.TestCase):
    def test_legacy_policy_keeps_physical_actions_and_values(self):
        policy = AsymmetricActorCriticPolicy(
            spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32),
            spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32),
            lambda _: 1e-3,
            actor_dim=ACTOR_OBS_DIM,
        )
        model = SimpleNamespace(policy=policy)
        rng = np.random.default_rng(13)
        old_observations = torch.as_tensor(
            rng.normal(0.0, 0.2, (32, OBS_DIM)), dtype=torch.float32
        )
        new_observations = old_observations.clone()
        previous_wheels = OBS_SLICES["previous_action"].start
        new_observations[:, previous_wheels:previous_wheels + 2] *= -1

        with torch.no_grad():
            old_mean = policy.get_distribution(old_observations).distribution.mean.clone()
            old_value = policy.predict_values(old_observations).clone()

        self.assertTrue(ensure_forward_positive_wheel_actions(model))
        self.assertFalse(ensure_forward_positive_wheel_actions(model))
        self.assertEqual(model.wheel_action_frame, WHEEL_ACTION_FRAME)

        with torch.no_grad():
            new_mean = policy.get_distribution(new_observations).distribution.mean
            new_value = policy.predict_values(new_observations)
        torch.testing.assert_close(new_mean[:, :2], -old_mean[:, :2])
        torch.testing.assert_close(new_mean[:, 2:], old_mean[:, 2:])
        torch.testing.assert_close(new_value, old_value)


if __name__ == "__main__":
    unittest.main()
