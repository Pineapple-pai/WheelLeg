#!/usr/bin/env python3
"""Run a local browser replay of a trained UZ-05 policy."""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_DIR = Path("/home/p/下载/UZ05_MJCF_real/UZ05_MJCF_real_params/meshes/stl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints/stand_negative_rate_refine_v1/checkpoint",
        help="SB3 checkpoint prefix or .zip file",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--stand-level", type=int, default=2)
    parser.add_argument("--fps", type=int, default=18, help="web preview refresh rate")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--once", action="store_true", help="停在每回合结束处，不自动循环")
    parser.add_argument("--mesh-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint.suffix == ".zip":
        checkpoint = checkpoint.with_suffix("")
    mesh_dir = (args.mesh_dir or Path(os.environ.get("UZ05_MESH_DIR", DEFAULT_MESH_DIR))).expanduser().resolve()
    if not mesh_dir.is_dir():
        raise SystemExit(f"找不到 MJCF 可视化网格目录：{mesh_dir}（可用 --mesh-dir 指定）")

    # The visual mesh references are resolved when uz05.spec is imported.
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["UZ05_VISUAL"] = "1"
    os.environ["UZ05_MESH_DIR"] = str(mesh_dir)
    os.environ["UZ05_REPLAY_CAMERA"] = "side"

    import mujoco
    import numpy as np
    from PIL import Image
    from stable_baselines3 import PPO

    from uz05.env import UZ05Env
    from uz05.policy import AsymmetricActorCriticPolicy  # noqa: F401 - registers checkpoint policy class
    from train_uz05 import require_compatible_contract

    require_compatible_contract(checkpoint)
    contract = json.loads(Path(f"{checkpoint}_contract.json").read_text())
    model = PPO.load(str(checkpoint), device=args.device)
    env = UZ05Env(stage="stand", seed=args.seed, stand_level=args.stand_level)
    # Match the dynamics used to train this checkpoint. Otherwise the browser
    # replay can silently randomize friction/mass/actuator gain for a nominal-
    # dynamics policy and show a behavior the saved policy was never trained on.
    env.params.domain_randomization.enabled = bool(
        contract.get("domain_randomization", True)
    )
    obs, _ = env.reset(seed=args.seed)
    renderer = [None]
    state_lock = threading.RLock()
    stop = threading.Event()
    state = {
        "paused": False,
        "done": False,
        "step": 0,
        "reward": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "yaw_error": 0.0,
        "vx": 0.0,
        "command_vx": 0.0,
        "height": float(env.sim.base_pos[2]),
        "drift": 0.0,
        "max_drift": 0.0,
        "reason": "",
        "episode": 1,
        "fps": args.fps,
        "checkpoint": checkpoint.parent.name,
        "frame": None,
    }
    camera_home = {
        "lookat": [0.0, 0.0, 0.33],
        "distance": 1.75,
        "azimuth": 90.0,
        "elevation": -12.0,
    }
    camera_state = {
        "lookat": list(camera_home["lookat"]),
        "distance": camera_home["distance"],
        "azimuth": camera_home["azimuth"],
        "elevation": camera_home["elevation"],
    }
    free_camera = [None]
    current_obs = [obs]
    last_reward = [0.0]
    last_info = [{}]
    control_period = float(env.params.control_dt)
    image_period = 1.0 / max(1, args.fps)

    def apply_camera_state() -> None:
        cam = free_camera[0]
        if cam is None:
            return
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = camera_state["lookat"]
        cam.distance = float(camera_state["distance"])
        cam.azimuth = float(camera_state["azimuth"])
        cam.elevation = float(camera_state["elevation"])

    def reset_camera() -> None:
        camera_state.update({
            "lookat": list(camera_home["lookat"]),
            "distance": camera_home["distance"],
            "azimuth": camera_home["azimuth"],
            "elevation": camera_home["elevation"],
        })
        apply_camera_state()

    def render_jpeg() -> bytes:
        renderer[0].update_scene(env.sim.data, camera=free_camera[0])
        rgb = renderer[0].render()
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=86, optimize=False)
        return buffer.getvalue()

    def worker() -> None:
        last_render = 0.0
        done_at = None
        next_step = time.monotonic()
        try:
            # EGL contexts are thread-local on this host: create, use, and close
            # the renderer on this same worker thread.
            renderer[0] = mujoco.Renderer(env.sim.model, height=args.height, width=args.width)
            free_camera[0] = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(free_camera[0])
            with state_lock:
                reset_camera()
            while not stop.is_set():
                now = time.monotonic()
                with state_lock:
                    if not state["paused"] and not state["done"]:
                        action, _ = model.predict(current_obs[0], deterministic=True)
                        current_obs[0], reward, terminated, truncated, info = env.step(action)
                        last_reward[0], last_info[0] = float(reward), info
                        state.update({
                            "step": int(info.get("episode_steps", env.steps)),
                            "reward": float(reward),
                            "pitch": float(info.get("pitch", 0.0)),
                            "roll": float(info.get("roll", 0.0)),
                            "yaw_error": float(info.get("yaw_error", 0.0)),
                            "vx": float(info.get("body_vx", 0.0)),
                            "command_vx": float(info.get("command_vx", 0.0)),
                            "height": float(info.get("base_height", env.sim.base_pos[2])),
                            "drift": float(info.get("station_error", 0.0)),
                            "max_drift": float(info.get("station_max_abs", 0.0)),
                        })
                        if terminated or truncated:
                            state["done"] = True
                            state["paused"] = True
                            state["reason"] = str(info.get("termination_reason") or "episode 完成")
                            done_at = now
                        next_step = now + control_period

                    if now - last_render >= image_period or state["frame"] is None:
                        state["frame"] = render_jpeg()
                        last_render = now
                    if state["done"] and not args.once and done_at is not None and now - done_at >= 1.2:
                        reset_replay()
                        done_at = None

                if state["paused"] or state["done"]:
                    stop.wait(0.02)
                else:
                    stop.wait(max(0.0, next_step - time.monotonic()))
        except Exception as exc:
            with state_lock:
                state["reason"] = f"回放异常：{exc}"
                state["paused"] = True
        finally:
            if renderer[0] is not None:
                renderer[0].close()

    def reset_replay() -> None:
        nonlocal_seed = args.seed + int(state["episode"])
        new_obs, _ = env.reset(seed=nonlocal_seed)
        current_obs[0] = new_obs
        last_reward[0], last_info[0] = 0.0, {}
        state.update({
            "paused": False, "done": False, "step": 0, "reward": 0.0,
            "pitch": 0.0, "roll": 0.0, "yaw_error": 0.0, "vx": 0.0,
            "command_vx": 0.0, "height": float(env.sim.base_pos[2]),
            "drift": 0.0, "max_drift": 0.0, "reason": "",
            "episode": int(state["episode"]) + 1,
        })

    class Handler(BaseHTTPRequestHandler):
        server_version = "UZ05Replay/1.0"

        def log_message(self, fmt, *values):
            return

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/" or route == "/index.html":
                page = (ROOT / "web_replay/index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if route == "/state":
                with state_lock:
                    payload = {key: value for key, value in state.items() if key != "frame"}
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if route == "/frame.jpg":
                with state_lock:
                    frame = state["frame"]
                if not frame:
                    self.send_error(503, "frame is not ready")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                return
            if route == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not stop.is_set():
                        with state_lock:
                            frame = state["frame"]
                        if frame:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                                + frame + b"\r\n"
                            )
                        time.sleep(image_period)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                return
            self.send_error(404)

        def do_POST(self):
            route = urlparse(self.path).path
            if route == "/camera":
                length = min(int(self.headers.get("Content-Length", "0")), 2048)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    action = body.get("action")
                    dx = float(body.get("dx", 0.0))
                    dy = float(body.get("dy", 0.0))
                    if not np.isfinite(dx) or not np.isfinite(dy):
                        raise ValueError("non-finite camera delta")
                    dx = float(np.clip(dx, -2000.0, 2000.0))
                    dy = float(np.clip(dy, -2000.0, 2000.0))
                except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                    self.send_error(400, "invalid camera command")
                    return

                with state_lock:
                    if action == "reset":
                        reset_camera()
                    elif action == "rotate":
                        camera_state["azimuth"] = (camera_state["azimuth"] - dx * 0.45) % 360.0
                        camera_state["elevation"] = float(np.clip(
                            camera_state["elevation"] - dy * 0.35, -85.0, 85.0
                        ))
                        apply_camera_state()
                    elif action == "pan":
                        azimuth = np.deg2rad(camera_state["azimuth"])
                        elevation = np.deg2rad(camera_state["elevation"])
                        right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
                        up = np.array([
                            -np.sin(elevation) * np.cos(azimuth),
                            -np.sin(elevation) * np.sin(azimuth),
                            np.cos(elevation),
                        ])
                        scale = camera_state["distance"] * 0.0022
                        lookat = np.asarray(camera_state["lookat"], dtype=float)
                        camera_state["lookat"] = (
                            lookat + scale * (-dx * right + dy * up)
                        ).tolist()
                        apply_camera_state()
                    elif action == "zoom":
                        camera_state["distance"] = float(np.clip(
                            camera_state["distance"] * np.exp(dy * 0.0015), 0.45, 8.0
                        ))
                        apply_camera_state()
                    else:
                        self.send_error(400, "unknown camera action")
                        return
                payload = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if route != "/control":
                self.send_error(404)
                return
            length = min(int(self.headers.get("Content-Length", "0")), 2048)
            try:
                command = json.loads(self.rfile.read(length) or b"{}").get("command")
            except (json.JSONDecodeError, AttributeError):
                self.send_error(400, "invalid JSON")
                return
            with state_lock:
                if command == "reset":
                    reset_replay()
                elif command == "toggle" and not state["done"]:
                    state["paused"] = not state["paused"]
                else:
                    self.send_error(400, "unknown command")
                    return
                payload = json.dumps({"ok": True, "paused": state["paused"]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    worker_thread = threading.Thread(target=worker, name="uz05-replay", daemon=True)
    worker_thread.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"网页回放：http://{args.host}:{args.port}/", flush=True)
    print(f"策略：{checkpoint}.zip | MJCF：{ROOT / 'uz05_修正后(3).xml'}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        worker_thread.join(timeout=2)
        env.close()


if __name__ == "__main__":
    main()
