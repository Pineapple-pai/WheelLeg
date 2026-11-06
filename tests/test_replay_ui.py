"""Replay controls and isolated native screen overlay integration."""
import os
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from uz05.replay_controls import KeyboardMailbox, KeyboardCommandState
from uz05.replay_viewer import ReplayViewer


class ReplayControlsTest(unittest.TestCase):
    def test_stand_key_on_both_keyboards(self):
        mailbox = KeyboardMailbox()
        for key in (55, 327):  # top-row 7 and keypad 7
            mailbox.on_key(key)
            self.assertEqual(mailbox.drain(), [('mode', 'stand')])
        for key in (48, 320):
            mailbox.on_key(key)
            self.assertEqual(mailbox.drain(), [])

    def test_stand_preserves_height_and_adjusts_it(self):
        state = KeyboardCommandState(mode='reverse', height=.30, speed=2)
        bounds = ((-2, 2), (-1.5, 1.5), (.22, .32))
        state.handle('mode', 'stand', *bounds)
        self.assertEqual(state.command(), [0, 0, 0, .30, 0])
        state.handle('adjust', 1, *bounds)
        self.assertAlmostEqual(state.command()[3], .31)
        self.assertEqual(state.command()[:3], [0, 0, 0])


@unittest.skipUnless(os.environ.get('DISPLAY'), 'native viewer requires DISPLAY')
class NativeReplayTest(unittest.TestCase):
    def test_ui_round_trip_does_not_advance_physics(self):
        runtime = Path(__file__).resolve().parents[1] / '.cache/mujoco-native-ui-3.3.7'
        if not runtime.exists():
            self.skipTest('isolated UI runtime not installed')
        version = mujoco.__version__
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        qpos, qvel, timestamp = data.qpos.copy(), data.qvel.copy(), data.time
        viewer = ReplayViewer(model, data)
        try:
            viewer.cam.distance = 2
            viewer.track([0, 0, 1])
            viewer.set_hud(['Pitch 0 deg   Yaw 0 deg   Roll 0 deg',
                            'Vx actual 0 / target 0   Yaw rate actual 0 / target 0',
                            'Height actual 1 / target 1   Mode stand'])
            for _ in range(5):
                viewer.sync()
            self.assertTrue(viewer.is_running())
            np.testing.assert_array_equal(data.qpos, qpos)
            np.testing.assert_array_equal(data.qvel, qvel)
            self.assertEqual(data.time, timestamp)
            self.assertEqual(mujoco.__version__, version)
        finally:
            viewer.close()
        self.assertEqual(viewer._process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
