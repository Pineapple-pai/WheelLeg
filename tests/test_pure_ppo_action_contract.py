from __future__ import annotations

import unittest

import numpy as np

from uz05.pure_ppo_contract import contract_payload, require_compatible_contract
from uz05.pure_ppo_stand import (
    ACTION_DIM,
    ACTION_ORDER,
    LEG_POSITION_SCALE_RAD,
    Q_REF,
    WHEEL_SPEED_SCALE_RAD_S,
    PurePPOStandActuator,
    map_stand_offset_action,
)
from uz05.spec import EnvParams


class PurePPOActionContractTests(unittest.TestCase):
    def test_zero_action_is_zero_wheel_speed_and_static_qref(self):
        action, wheel, joint = map_stand_offset_action(np.zeros(ACTION_DIM))
        np.testing.assert_array_equal(action, np.zeros(ACTION_DIM))
        np.testing.assert_array_equal(wheel, np.zeros(2))
        np.testing.assert_array_equal(joint, Q_REF)

    def test_every_action_channel_changes_only_its_target_coordinate(self):
        for index, name in enumerate(ACTION_ORDER):
            with self.subTest(channel=name):
                action = np.zeros(ACTION_DIM)
                action[index] = 0.2
                _, wheel, joint = map_stand_offset_action(action)
                if index < 2:
                    expected = np.zeros(2)
                    expected[index] = WHEEL_SPEED_SCALE_RAD_S * 0.2
                    np.testing.assert_allclose(wheel, expected, atol=1e-12)
                    np.testing.assert_allclose(joint, Q_REF, atol=1e-12)
                else:
                    expected = Q_REF.copy()
                    expected[index - 2] += LEG_POSITION_SCALE_RAD * 0.2
                    np.testing.assert_allclose(wheel, np.zeros(2), atol=1e-12)
                    np.testing.assert_allclose(joint, expected, atol=1e-12)

    def test_action_clips_at_protocol_bounds_and_rejects_invalid_input(self):
        action = np.asarray([2.0, -2.0, 0, 0, 0, 0])
        clipped, wheel, _ = map_stand_offset_action(action)
        np.testing.assert_array_equal(clipped, [1, -1, 0, 0, 0, 0])
        np.testing.assert_array_equal(wheel, [8.0, -8.0])
        with self.assertRaises(ValueError):
            map_stand_offset_action(np.zeros(5))
        with self.assertRaises(ValueError):
            map_stand_offset_action(np.asarray([0, 0, 0, 0, np.nan, 0]))

    def test_actuator_targets_do_not_depend_on_feedback_state(self):
        actuator = PurePPOStandActuator(EnvParams())
        action = np.asarray([0.1, -0.2, 0.2, -0.1, 0.05, -0.05])
        expected_action, expected_wheel, expected_joint = map_stand_offset_action(action)
        first = actuator.compute(
            action,
            joint_pos=Q_REF,
            joint_vel=np.zeros(4),
            wheel_vel=np.zeros(2),
        )
        second = actuator.compute(
            action,
            joint_pos=Q_REF + np.asarray([0.1, -0.1, 0.05, -0.05]),
            joint_vel=np.asarray([0.2, -0.2, 0.1, -0.1]),
            wheel_vel=np.asarray([1.0, -1.0]),
        )
        np.testing.assert_array_equal(first["action"], expected_action)
        np.testing.assert_array_equal(second["action"], expected_action)
        np.testing.assert_allclose(first["wheel_target"], expected_wheel)
        np.testing.assert_allclose(second["wheel_target"], expected_wheel)
        np.testing.assert_allclose(first["joint_target"], expected_joint)
        np.testing.assert_allclose(second["joint_target"], expected_joint)
        self.assertFalse(np.allclose(first["wheel_current"], second["wheel_current"]))
        self.assertFalse(np.allclose(first["joint_torque"], second["joint_torque"]))

    def test_checkpoint_contract_accepts_current_protocol_and_rejects_legacy(self):
        contract = contract_payload(seed=0, config_sha256="test")
        require_compatible_contract(contract)
        legacy = dict(contract)
        legacy["action_protocol"] = "legacy_common_differential_v0"
        with self.assertRaises(ValueError):
            require_compatible_contract(legacy)
        missing = dict(contract)
        del missing["actor_observation_layout_sha256"]
        with self.assertRaises(ValueError):
            require_compatible_contract(missing)


if __name__ == "__main__":
    unittest.main()
