"""Regression checks for MuJoCo free-joint angular velocity conventions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from uz05.model import UZ05Model


class FreeJointAngularVelocityFrameTest(unittest.TestCase):
    def test_world_and_body_angular_velocity_frames(self):
        sim = UZ05Model()
        sim.reset(height=0.23825)
        euler = [0.17, -0.23, 0.09]
        quat = Rotation.from_euler("xyz", euler).as_quat()
        sim.data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        local_ang_vel = np.array([0.4, -0.7, 0.2], dtype=np.float64)
        sim.data.qvel[3:6] = local_ang_vel
        sim.forward()

        np.testing.assert_allclose(
            sim.base_ang_vel_world,
            sim.rotation_matrix() @ local_ang_vel,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            sim.body_frame(sim.base_ang_vel_world),
            local_ang_vel,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
