from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_uz05 import TrainingCallback
from uz05.env import UZ05Env


class HorizonCurriculumTest(unittest.TestCase):
    def test_episode_horizon_can_change_without_changing_environment_contract(self):
        env = UZ05Env(
            stage="stand", seed=7, init_scale=0.0, stand_level=0,
            episode_steps_override=2,
        )
        try:
            obs, _ = env.reset(seed=7)
            self.assertEqual(obs.shape, env.observation_space.shape)
            _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            _, _, terminated, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            env.set_episode_steps(3)
            env.reset(seed=7)
            for _ in range(2):
                _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
                self.assertFalse(truncated)
            _, _, _, truncated, _ = env.step(np.zeros(6, dtype=np.float32))
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_forward_only_horizon_stage_does_not_require_reverse_episodes(self):
        args = SimpleNamespace(
            curriculum_window_episodes=10,
            curriculum_min_per_direction=4,
            curriculum_zero_prob=0.25,
            curriculum_min_survival=0.9,
            curriculum_max_vx_error=0.05,
            curriculum_min_direction_accuracy=0.7,
            curriculum_min_contact=0.95,
            curriculum_max_zero_drift_m=0.08,
            curriculum_stages=[0.08, 0.12, 0.30],
            horizon_curriculum=True,
            horizon_stages=[32, 64, 128, 256, 1000],
            horizon_max_pitch_deg=10.0,
            velocity_curriculum=False,
            vx_range=(0.08, 0.12),
        )
        callback = TrainingCallback(args, total_updates=1)
        callback.curriculum_episodes.extend([
            dict(sign=1, magnitude=0.1, survived=1.0, vx_mae=0.02,
                 direction_accuracy=0.8, contact_both=1.0,
                 motion_peak=0.02, station_peak=0.0,
                 pitch_p95_rad=np.deg2rad(5.0))
            for _ in range(7)
        ])
        callback.curriculum_episodes.extend([
            dict(sign=0, magnitude=0.0, survived=1.0, vx_mae=float("nan"),
                 direction_accuracy=float("nan"), contact_both=1.0,
                 motion_peak=0.0, station_peak=0.01,
                 pitch_p95_rad=float("nan"))
            for _ in range(3)
        ])
        summary = callback._curriculum_summary()
        self.assertTrue(summary["qualified"])
        self.assertEqual(summary["negative_near"], 0)
        calls = []
        fake_env = SimpleNamespace(env_method=lambda *values: calls.append(values))
        callback.model = SimpleNamespace(n_envs=2, get_env=lambda: fake_env)
        self.assertTrue(callback._advance_horizon_curriculum(summary))
        self.assertEqual(callback.curriculum_stage, 1)
        self.assertEqual(calls, [("set_episode_steps", 64)])
        self.assertEqual(len(callback.curriculum_episodes), 0)
        callback.curriculum_stage = 0
        callback.curriculum_episodes.extend([
            dict(sign=1, magnitude=0.1, survived=1.0, vx_mae=0.02,
                 direction_accuracy=0.8, contact_both=1.0,
                 motion_peak=0.02, station_peak=0.0,
                 pitch_p95_rad=np.deg2rad(5.0))
            for _ in range(7)
        ])
        callback.curriculum_episodes.extend([
            dict(sign=0, magnitude=0.0, survived=1.0, vx_mae=float("nan"),
                 direction_accuracy=float("nan"), contact_both=1.0,
                 motion_peak=0.0, station_peak=0.01,
                 pitch_p95_rad=float("nan"))
            for _ in range(3)
        ])
        callback.curriculum_episodes[0]["pitch_p95_rad"] = np.deg2rad(15.0)
        self.assertFalse(callback._curriculum_summary()["qualified"])


if __name__ == "__main__":
    unittest.main()
