from __future__ import annotations

import unittest
from collections import deque

import mujoco
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from uz05.observations import ObservationEncoder
from uz05.policy import AsymmetricMlpExtractor
from uz05.pure_ppo_stand import POLICY_DT_S, PurePPOStandEnv
from uz05.spec import (
    ACTOR_OBS_BLOCKS,
    ACTOR_OBS_DIM,
    OBS_BLOCKS,
    OBS_DIM,
    OBS_SCALE,
    PRIV_OBS_BLOCKS,
    PRIV_OBS_DIM,
    ObservationNoise,
)


class PurePPOObservationTests(unittest.TestCase):
    def test_declared_observation_dimensions_and_order(self):
        self.assertEqual(sum(width for _, width in ACTOR_OBS_BLOCKS), 38)
        self.assertEqual(sum(width for _, width in PRIV_OBS_BLOCKS), 56)
        self.assertEqual(ACTOR_OBS_DIM + PRIV_OBS_DIM, OBS_DIM)
        env = PurePPOStandEnv(seed=3)
        observation, _ = env.reset(seed=3)
        self.assertEqual(observation.shape, (OBS_DIM,))
        self.assertEqual(env.observation_space.shape, (OBS_DIM,))
        env.close()

    def test_noise_off_encoding_matches_scaled_blocks_exactly(self):
        encoder = ObservationEncoder(ObservationNoise(enabled=False), 100.0)
        blocks = {
            name: np.linspace(index + 1.0, index + 1.0 + width - 1, width)
            for index, (name, width) in enumerate(OBS_BLOCKS)
        }
        obs, raw, scaled = encoder.encode(blocks, np.random.default_rng(0))
        expected = np.concatenate([scaled[name] for name, _ in OBS_BLOCKS]).astype(np.float32)
        np.testing.assert_array_equal(obs, expected)
        for name, _ in OBS_BLOCKS:
            np.testing.assert_array_equal(raw[name], blocks[name])

    def test_noise_configuration_is_per_encoder_instance(self):
        noisy_config = ObservationNoise(enabled=True, level=1.0)
        clean_config = ObservationNoise(enabled=False)
        noisy = ObservationEncoder(noisy_config, 100.0)
        clean = ObservationEncoder(clean_config, 100.0)
        self.assertGreater(float(np.max(noisy.noise["base_ang_vel"])), 0.0)
        self.assertEqual(float(np.max(clean.noise["base_ang_vel"])), 0.0)
        self.assertEqual(float(np.max(noisy.noise["gravity"])), 0.05)
        self.assertEqual(float(np.max(clean.noise["gravity"])), 0.0)

    def test_actor_latent_ignores_privileged_tail(self):
        torch.manual_seed(4)
        extractor = AsymmetricMlpExtractor(
            ACTOR_OBS_DIM, OBS_DIM, {"pi": [32, 16], "vf": [32, 16]}, nn.ELU
        )
        obs = torch.randn(5, OBS_DIM)
        changed = obs.clone()
        changed[:, ACTOR_OBS_DIM:] += 1000.0
        actor_a = extractor.forward_actor(obs)
        actor_b = extractor.forward_actor(changed)
        critic_a = extractor.forward_critic(obs)
        critic_b = extractor.forward_critic(changed)
        torch.testing.assert_close(actor_a, actor_b)
        self.assertFalse(torch.allclose(critic_a, critic_b))

    def test_free_joint_angular_velocity_world_to_body_roundtrip_and_pitch_sign(self):
        env = PurePPOStandEnv(seed=9)
        cases = [
            (0.11, np.deg2rad(5.0), 0.73, 0.10),
            (-0.09, np.deg2rad(-5.0), -0.52, -0.10),
        ]
        for roll, pitch, yaw, pitch_rate in cases:
            with self.subTest(pitch=pitch, yaw=yaw):
                quat = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
                env.sim.data.qpos[3:7] = [quat[3], *quat[:3]]
                env.sim.data.qvel[3:6] = [0.03, pitch_rate, -0.07]
                mujoco.mj_forward(env.sim.model, env.sim.data)
                body_rate = env.sim.body_frame(env.sim.base_ang_vel_world)
                np.testing.assert_allclose(body_rate, env.sim.data.qvel[3:6], atol=1e-12)
                self.assertAlmostEqual(env._rpy(env.sim)[1], pitch, places=12)
                self.assertAlmostEqual(body_rate[1], pitch_rate, places=12)
                gravity_body = env._get_observation()[:3]
                self.assertEqual(np.sign(gravity_body[0]), np.sign(pitch))
        env.close()

    def test_dof_acceleration_is_new_minus_old(self):
        env = PurePPOStandEnv(seed=10)
        env._dof_vel_hist = deque(maxlen=9)
        env._dof_vel_hist.append(np.zeros(6))
        env._dof_vel_hist.append(np.ones(6))
        observation = env._get_observation()
        offset = ACTOR_OBS_DIM + sum(width for name, width in PRIV_OBS_BLOCKS[:6])
        # The first of eight privileged blocks after terrain_scan is dof_acc;
        # a one-second span is not used, so check its sign and exact 8 ms scale.
        # Current history length is two samples, hence one policy-step span.
        expected = np.full(6, (1.0 / POLICY_DT_S) * float(OBS_SCALE["dof_acc"]))
        np.testing.assert_allclose(observation[offset : offset + 6], expected, atol=1e-6)
        env.close()


if __name__ == "__main__":
    unittest.main()
