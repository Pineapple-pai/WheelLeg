#!/usr/bin/env python3
"""MuJoCo native viewer or MP4 replay for the joint tracking PPO policy."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
MESHES = Path('/home/p/下载/UZ05_MJCF_real/UZ05_MJCF_real_params/meshes/stl')
CASES = {
    'stand': (0.0, 0.0, 0.27),
    'forward': (0.3, 0.0, 0.27),
    'reverse': (-0.3, 0.0, 0.27),
    'yaw': (0.0, 1.0, 0.27),
    'yaw_right': (0.0, -1.0, 0.27),
    'mixed_left': (0.3, 1.0, 0.27),
    'mixed': (0.3, -1.0, 0.27),
    'height': (0.0, 0.0, 0.245),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/joint_tracking_pos_v2/checkpoint')
    parser.add_argument('--case', choices=(*CASES, 'suite', 'random'), default='suite')
    parser.add_argument('--nominal', action='store_true', help='disable domain randomization and scheduled pushes')
    parser.add_argument('--moving-starts', action='store_true', help='启用训练中的运动初态分布；默认从近静止状态回放')
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--duration', type=float, default=12.0)
    parser.add_argument('--speed', type=float, default=.3, help='initial translation speed magnitude (m/s)')
    parser.add_argument('--yaw-rate', type=float, default=1.0, help='initial turning speed magnitude (rad/s)')
    parser.add_argument('--height', type=float, default=None, help='held base height above ground (m)')
    parser.add_argument('--video', type=Path, help='Render to MP4 instead of opening the native viewer')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--mesh-dir', type=Path, default=MESHES)
    parser.add_argument('--no-mesh', action='store_true', help='Use collision geometry if visual meshes are unavailable')
    parser.add_argument('--keyboard', action='store_true', help='数字键盘交互控制；切换模式不重置机器人')
    parser.add_argument('--once', action='store_true', help='Exit after one pass through the selected cases')
    args = parser.parse_args()
    if args.duration < 0.008 or args.fps < 1 or args.fps > 125:
        parser.error('duration must be >= 0.008 seconds and fps must be in [1,125]')
    if args.keyboard and args.video:
        parser.error('--keyboard requires the native viewer; omit --video')
    if args.video:
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['UZ05_VISUAL'] = '0' if args.no_mesh else '1'
    os.environ['UZ05_MESH_DIR'] = str(args.mesh_dir.expanduser())
    os.environ['UZ05_REPLAY_SCENE'] = 'checker'

    import mujoco
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from train_uz05 import require_compatible_contract
    from uz05.env import UZ05Env
    from uz05.spec import EnvParams, env_params_for_stage
    from uz05.replay_viewer import ReplayViewer, telemetry_lines

    checkpoint = args.checkpoint.expanduser().resolve().with_suffix('')
    contract = json.loads(Path(f'{checkpoint}_contract.json').read_text())
    stage = contract['stage']
    require_compatible_contract(checkpoint, stage)
    params = env_params_for_stage(stage)
    params.domain_randomization.enabled = contract['domain_randomization'] and not args.nominal
    if args.nominal:
        params.robust.pushes_enabled = False
    env = UZ05Env(stage, params=params)
    if not args.moving_starts:
        env.stage = replace(env.stage, moving_start_probability=0.0)
    if not np.isfinite(args.speed) or not 0 <= args.speed <= max(abs(v) for v in env.stage.vx_range):
        parser.error('--speed is outside this model training range')
    if not np.isfinite(args.yaw_rate) or not 0 <= args.yaw_rate <= max(abs(v) for v in env.stage.yaw_range):
        parser.error('--yaw-rate is outside this model training range')
    if args.height is not None and (not np.isfinite(args.height) or not env.stage.base_height_range[0] <= args.height <= env.stage.base_height_range[1]):
        parser.error('--height is outside this model training range')
    case_commands = {key: ((args.speed if vx > 0 else -args.speed) if vx else 0,
                          (args.yaw_rate if yaw > 0 else -args.yaw_rate) if yaw else 0,
                          args.height if args.height is not None else h)
                     for key, (vx, yaw, h) in CASES.items()}
    if env.stage.transition_tracking and args.height is None:
        case_commands['height'] = (0, 0, env.stage.base_height_range[0])
    normalized = VecNormalize.load(f'{checkpoint}_vecnormalize.pkl', DummyVecEnv([lambda: env]))
    normalized.training = False
    normalized.norm_reward = False
    model = PPO.load(checkpoint, device='cpu')
    cases = ['stand', 'forward', 'yaw', 'mixed', 'height'] if args.case == 'suite' else [args.case]
    viewer = renderer = writer = None
    summaries = []
    from uz05.replay_controls import KeyboardMailbox, KeyboardCommandState, KEYBOARD_HELP
    keyboard = KeyboardMailbox()
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        hud_font = ImageFont.truetype('DejaVuSansMono.ttf', 13)
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.video), fps=args.fps, codec='libx264', macro_block_size=2)
        env.sim.model.vis.global_.offwidth = 800
        env.sim.model.vis.global_.offheight = 500
        renderer = mujoco.Renderer(env.sim.model, width=800, height=500)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance, camera.azimuth, camera.elevation = 1.8, 100, -10
    else:
        viewer = ReplayViewer(env.sim.model, env.sim.data,
                              key_callback=keyboard.on_key if args.keyboard else None)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 100, -10

    def frame(case, info, reason):
        camera.lookat[:] = [env.sim.base_pos[0], env.sim.base_pos[1], 0.28]
        renderer.update_scene(env.sim.data, camera=camera)
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        lines = telemetry_lines(env, case)
        draw.rectangle((0, 0, 800, 10 + 20*len(lines)), fill=(15, 20, 26))
        for index, line in enumerate(lines):
            draw.text((8, 6 + index * 20), line, font=hud_font,
                      fill='white' if index == 0 else (140, 235, 220))
        return np.asarray(image)

    def run_keyboard():
        mode = args.case if args.case in CASES or args.case == 'random' else 'stand'
        state = KeyboardCommandState(mode=mode, height=case_commands[mode][2] if mode in case_commands else .27,
                                     speed=args.speed, turn_speed=args.yaw_rate)
        episode = 0

        def apply_command():
            if state.mode != 'random':
                env.command[:] = state.command()
                env.command_target[:] = env.command
                env.command_steps_left = env.stage.episode_steps + 1
            return env._obs(current_action=env.previous_action)

        print(KEYBOARD_HELP, flush=True)
        while viewer.is_running():
            obs, _ = env.reset(seed=args.seed + episode)
            obs = apply_command()
            print(f'[键盘回放] mode={state.mode} vx={env.command[0]:+.2f} '
                  f'yaw={env.command[2]:+.2f} height={state.height:.3f}m episode={episode+1}', flush=True)
            while viewer.is_running():
                started = time.perf_counter()
                reset = False
                for kind, value in keyboard.drain():
                    if kind == 'reset':
                        reset = True
                    else:
                        state.handle(kind, value, env.stage.vx_range, env.stage.yaw_range,
                                     env.stage.base_height_range)
                        obs = apply_command()
                        print(f'[命令] mode={state.mode} vx={env.command[0]:+.2f} '
                              f'yaw={env.command[2]:+.2f} height={env.command[3]:.3f}', flush=True)
                if reset:
                    break
                action, _ = model.predict(normalized.normalize_obs(obs), deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                viewer.track([env.sim.base_pos[0], env.sim.base_pos[1], 0.28])
                viewer.set_hud(telemetry_lines(env, state.mode))
                viewer.sync()
                time.sleep(max(0, params.control_dt - (time.perf_counter() - started)))
                if terminated or truncated:
                    print(f'[回合结束] {info["termination_reason"]}', flush=True)
                    if args.once:
                        return
                    break
            episode += 1

    try:
        if args.keyboard:
            run_keyboard()
            return
        while True:
            for case in cases:
                obs, _ = env.reset(seed=args.seed)
                if case != 'random':
                    vx, yaw, height = case_commands[case]
                    env.command[:] = [vx, 0, yaw, height, 0]
                    env.command_target[:] = env.command
                    env.command_steps_left = int(args.duration / params.control_dt) + 1
                    obs = env._obs()
                next_frame = 0
                reason = 'running'
                for step in range(min(env.stage.episode_steps, int(args.duration / params.control_dt))):
                    started = time.perf_counter()
                    if viewer is not None and not viewer.is_running():
                        return
                    # A held zero-speed height change, with no extra stabilizer.
                    if case == 'height' and step == int(4 / params.control_dt):
                        env.command[3] = env.command_target[3] = (env.stage.base_height_range[1] if env.stage.transition_tracking else .295)
                        obs = env._obs(current_action=env.previous_action)
                    action, _ = model.predict(normalized.normalize_obs(obs), deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    reason = info['termination_reason']
                    if writer is not None and step * params.control_dt * args.fps >= next_frame:
                        writer.append_data(frame(case, info, reason))
                        next_frame += 1
                    if viewer is not None:
                        viewer.track([env.sim.base_pos[0], env.sim.base_pos[1], 0.28])
                        viewer.set_hud(telemetry_lines(env, case))
                        viewer.sync()
                        time.sleep(max(0, params.control_dt - (time.perf_counter() - started)))
                    if terminated or truncated:
                        break
                summary = {'case': case, 'seed': args.seed, 'steps': env.steps, 'reason': reason}
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
                if writer is not None:
                    last_frame = frame(case, info, reason)
                    for _ in range(args.fps):
                        writer.append_data(last_frame)
            if args.video or args.once:
                break
    finally:
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.close()
            args.video.with_suffix('.json').write_text(json.dumps(summaries, indent=2) + '\n')
        normalized.close()


if __name__ == '__main__':
    main()
