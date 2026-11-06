"""Display-only native viewer worker, launched by replay_viewer.ReplayViewer.

Its MuJoCo import comes from an isolated UI runtime, never the training runtime.
State and key events travel over a private inherited socket. Do not mj_step here.
"""
import sys
from multiprocessing.connection import Connection
from queue import Empty, SimpleQueue

import mujoco
import mujoco.viewer
def main():
    connection = Connection(int(sys.argv[1]))
    viewer = None
    try:
        model = mujoco.MjModel.from_xml_path(sys.argv[2])
        data = mujoco.MjData(model)
        keys = SimpleQueue()
        viewer = mujoco.viewer.launch_passive(
            model, data, key_callback=keys.put,
            show_left_ui=True, show_right_ui=True)
        connection.send({'ready': True})
        previous_position = previous_camera = None
        while True:
            packet = connection.recv()
            if packet is None:
                break
            if not viewer.is_running():
                connection.send({'running': False})
                break
            with viewer.lock():
                for name in ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat'):
                    getattr(data, name)[:] = packet[name]
                data.time = packet['time']
                mujoco.mj_forward(model, data)
                position = packet['position']
                if position is not None:
                    if previous_position is None:
                        viewer.cam.lookat[:] = position
                    else:
                        viewer.cam.lookat[:] += position - previous_position
                    previous_position = position.copy()
                # Apply programmatic camera settings only when changed, so
                # mouse orbit/zoom remain entirely under native viewer control.
                camera = packet['camera']
                if camera != previous_camera:
                    for name, value in camera.items():
                        setattr(viewer.cam, name, value)
                    previous_camera = camera.copy()
            # set_texts owns screen-space placement on every native redraw.
            # Use short rows so the telemetry fits between both sidebars.
            lines = packet['hud']
            if lines:
                height, *status = lines[2].split('   ')
                compact = [' '.join(lines[0].replace(' deg', '').split()) + ' (deg)',
                           *lines[1].split('   '), height, ' '.join(status)]
                viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL,
                                  mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  '\n'.join(compact), ''))
            viewer.sync()
            pending = []
            while True:
                try:
                    pending.append(keys.get_nowait())
                except Empty:
                    break
            connection.send({'running': viewer.is_running(), 'keys': pending,
                             'xfrc_applied': data.xfrc_applied.copy()})
    except (EOFError, BrokenPipeError, ConnectionResetError):
        pass
    except Exception as error:
        try:
            connection.send({'error': f'Native UI: {error}'})
        except OSError:
            pass
    finally:
        if viewer is not None:
            viewer.close()
        connection.close()


if __name__ == '__main__':
    main()
