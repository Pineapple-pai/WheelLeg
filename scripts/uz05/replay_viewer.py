"""Native MuJoCo replay with screen-space telemetry and unchanged physics."""

import numpy as np
from scipy.spatial.transform import Rotation


def telemetry_lines(env, mode):
    sim = env.sim
    q = sim.base_quat
    roll, pitch, yaw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz', degrees=True)
    vx = float(sim.body_frame(sim.base_lin_vel_world)[0])
    yaw_rate = float(sim.body_frame(sim.base_ang_vel_world)[2])
    height = float(sim.base_pos[2] - sim.support_height(float(sim.base_pos[0]), float(sim.base_pos[1])))
    posture = sim.support_posture()
    axle = sim.wheel_axle_geometry()
    leg_diff = float(np.diff(sim.leg_lengths())[0])
    control = env._previous_control
    target_line = 'Joint targets: awaiting policy step'
    if control is not None:
        target = control['joint_target']
        angles = np.degrees(np.arctan2(np.sin(target), np.cos(target)))
        difference = target[:2] - target[2:]
        pair = np.degrees(np.arctan2(np.sin(difference), np.cos(difference)))
        target_line = (f'Joint target deg L[{angles[0]:+.1f},{angles[1]:+.1f}] '
                       f'R[{angles[2]:+.1f},{angles[3]:+.1f}]   L-R[{pair[0]:+.1f},{pair[1]:+.1f}]')
    return [
        f'Pitch {pitch:+7.2f} deg   Yaw {yaw:+7.2f} deg   Roll {roll:+7.2f} deg',
        f'Vx actual {vx:+.3f} / target {env.command[0]:+.3f} m/s   '
        f'Yaw rate actual {yaw_rate:+.3f} / target {env.command[2]:+.3f} rad/s',
        f'Height actual {height:.3f} / target {env.command[3]:.3f} m   '
        f'Mode {mode}   t={env.steps * env.params.control_dt:.2f}s',
        f'COM lean {np.degrees(posture["support_com_lean_rad"]):+.2f} deg (forward +)   '
        f'Chassis/COM error {np.degrees(posture["support_pitch_error_rad"]):+.2f} deg',
        f'Axle offset {axle["axle_line_offset_m"]*1000:.1f} mm   '
        f'Axis angle {axle["axle_parallel_error_deg"]:.2f} deg   Leg R-L {leg_diff*1000:+.1f} mm',
        target_line,
    ]


class ReplayViewer:
    """Native UI in a display-only process; the caller retains its physics engine.

    MuJoCo 3.2 lacks screen overlays. The isolated UI runtime uses set_texts,
    which renders on the native render thread, including mouse-driven redraws.
    No simulation steps or policy inference run in that process.
    """
    def __init__(self, model, data, key_callback=None):
        import os
        import socket
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from multiprocessing.connection import Connection
        import mujoco

        root = Path(__file__).resolve().parents[2]
        runtime = root / '.cache/mujoco-native-ui-3.3.7'
        if not (runtime / 'mujoco/viewer.py').is_file():
            raise RuntimeError(
                'Missing native UI runtime. Run: conda run -n sim python -m pip '
                'install --no-deps --target .cache/mujoco-native-ui-3.3.7 mujoco==3.3.7')
        self.model, self.data = model, data
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self._key_callback = key_callback
        self._hud = []
        self._position = None
        self._running = True
        self._temp = tempfile.TemporaryDirectory(prefix='wheelleg-native-ui-')
        xml = Path(self._temp.name) / 'scene.xml'
        mujoco.mj_saveLastXML(str(xml), model)
        parent_socket, child_socket = socket.socketpair()
        self._connection = Connection(parent_socket.detach())
        child_fd = child_socket.fileno()
        child_env = os.environ.copy()
        child_env['PYTHONPATH'] = str(runtime) + os.pathsep + child_env.get('PYTHONPATH', '')
        self._process = None
        try:
            self._process = subprocess.Popen(
                [sys.executable, str(Path(__file__).with_name('replay_native_ui.py')),
                 str(child_fd), str(xml)], pass_fds=(child_fd,), env=child_env)
        except BaseException:
            self._connection.close()
            self._temp.cleanup()
            raise
        finally:
            child_socket.close()
        try:
            if not self._connection.poll(60):
                raise RuntimeError('Native MuJoCo UI did not start within 60 seconds')
            response = self._connection.recv()
            if response.get('error'):
                raise RuntimeError(response['error'])
        except BaseException:
            self.close()
            raise

    def set_hud(self, lines):
        self._hud = list(lines)

    def track(self, position):
        self._position = np.asarray(position, dtype=float).copy()

    def is_running(self):
        return self._running and self._process.poll() is None

    def sync(self, force=False):
        if not self.is_running():
            return
        # Transfer render state only. Native perturbations return to the original
        # engine just as with launch_passive; the UI never advances physics.
        packet = {name: getattr(self.data, name).copy() for name in
                  ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat')}
        packet.update(time=self.data.time, hud=self._hud, position=self._position,
                      camera={name: float(getattr(self.cam, name)) for name in
                              ('distance', 'azimuth', 'elevation')})
        try:
            self._connection.send(packet)
            if not self._connection.poll(10):
                raise RuntimeError('Native MuJoCo UI stopped responding')
            response = self._connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self._running = False
            return
        if response.get('error'):
            raise RuntimeError(response['error'])
        self._running = response['running']
        for key in response.get('keys', ()):
            if self._key_callback is not None:
                self._key_callback(key)
        if 'xfrc_applied' in response:
            self.data.xfrc_applied[:] = response['xfrc_applied']

    def close(self):
        import subprocess
        self._running = False
        try:
            if self._process is not None and self._process.poll() is None:
                try:
                    self._connection.send(None)
                    self._process.wait(timeout=3)
                except (OSError, EOFError, subprocess.TimeoutExpired):
                    self._process.terminate()
                    self._process.wait(timeout=3)
        finally:
            self._connection.close()
            self._temp.cleanup()
