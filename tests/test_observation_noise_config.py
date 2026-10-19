"""Regression tests for per-environment UZ-05 observation noise configuration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from uz05.env import UZ05Env
from uz05.spec import ACTOR_OBS_BLOCKS, ACTOR_OBS_DIM, EnvParams


class ObservationNoiseConfigTest(unittest.TestCase):
    @staticmethod
    def make_env(enabled: bool) -> UZ05Env:
        params = EnvParams()
        params.domain_randomization.enabled = False
        params.noise.enabled = enabled
        params.noise.level = 1.0
        return UZ05Env(stage="stand", params=params, seed=9,
                       stand_level=0, init_scale=0.0)

    def test_disabled_noise_preserves_scaled_actor_observation(self):
        env = self.make_env(enabled=False)
        try:
            obs, _ = env.reset(seed=9)
            expected = np.concatenate([
                env._obs_cache[name] for name, _ in ACTOR_OBS_BLOCKS
            ]).astype(np.float32)
            np.testing.assert_array_equal(obs[:ACTOR_OBS_DIM], expected)
        finally:
            env.close()

    def test_noise_enabled_per_environment_changes_actor_observation(self):
        env = self.make_env(enabled=True)
        try:
            obs, _ = env.reset(seed=9)
            clean = np.concatenate([
                env._obs_cache[name] for name, _ in ACTOR_OBS_BLOCKS
            ]).astype(np.float32)
            self.assertGreater(
                float(np.linalg.norm(obs[:ACTOR_OBS_DIM] - clean)), 1e-5
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
