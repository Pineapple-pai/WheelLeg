"""Local replay for the direct PPO target interface."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("UZ05_VISUAL", "1")

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_uz05 import AsymmetricActorCriticPolicy  # noqa: E402
from uz05.env import UZ05Env  # noqa: E402
from uz05.spec import ACTION_DIM, ACTION_SPEC  # noqa: E402

WEB_ROOT = ROOT / "web_replay"


def finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


class ReplaySession:
    def __init__(self, checkpoint: Path | None, stand_level: int, seed: int,
                 init_scale: float, deployment_mode: bool = False):
        self.lock = threading.RLock()
        self.checkpoint = checkpoint
        self.seed = int(seed)
        self.stand_level = int(stand_level)
        self.env = UZ05Env(stage="stand", stand_level=stand_level,
                           seed=seed, init_scale=init_scale,
                           deployment_mode=deployment_mode)
        self.model = None if checkpoint is None else PPO.load(
            str(checkpoint), custom_objects={"policy_class": AsymmetricActorCriticPolicy},
            device="cpu"
        )
        self.obs, _ = self.env.reset(seed=seed)
        self.playing = False
        self.done = False
        self.episode = 0
        self.total_steps = 0
        self.speed = 1.0
        self.motion = {"wheel_speed_rad_s": 0.0, "yaw_rate_rad_s": 0.0}
        self.history: deque[dict[str, float]] = deque(maxlen=2000)
        self.last_info: dict[str, Any] = {}
        self.last_reward = 0.0
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.camera = {"azimuth": 135.0, "elevation": -18.0,
                       "distance": 1.15, "lookat": [0.0, 0.0, 0.12]}
        # MuJoCo Renderer takes (height, width); keep height within the
        # model's default 480 px offscreen framebuffer.
        self.renderer = mujoco.Renderer(self.env.sim.model, 420, 640)
        self.render_camera = mujoco.MjvCamera()
        self.render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._apply_camera()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        with self.lock:
            self.renderer.close()
            self.env.close()

    def _apply_camera(self) -> None:
        self.render_camera.azimuth = self.camera["azimuth"]
        self.render_camera.elevation = self.camera["elevation"]
        self.render_camera.distance = self.camera["distance"]
        self.render_camera.lookat[:] = self.camera["lookat"]

    def _manual_action(self) -> np.ndarray:
        # Manual replay uses the same six normalized targets as PPO.  It is a
        # UI input mapper, not a second feedback law.
        wheel = np.clip(self.motion["wheel_speed_rad_s"] / ACTION_SPEC[0][2], -1.0, 1.0)
        yaw = np.clip(self.motion["yaw_rate_rad_s"] / 3.0, -1.0, 1.0)
        return np.array([wheel - 0.25 * yaw, wheel + 0.25 * yaw, 0, 0, 0, 0], dtype=np.float32)

    def _step_locked(self) -> None:
        if self.done:
            self.playing = False
            return
        if self.model is None:
            action = self._manual_action()
        else:
            action, _ = self.model.predict(self.obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = action.copy()
        self.last_reward = float(reward)
        self.last_info = dict(info)
        self.total_steps += 1
        self.history.append({
            "step": float(self.total_steps),
            "drift_cm": 100.0 * finite(info.get("station_error")),
            "pitch_deg": np.degrees(finite(info.get("pitch"))),
            "height_m": finite(info.get("base_height")),
            "wheel_target_rad_s": finite(info.get("wheel_target_abs")),
            "leg_error_mm": finite(info.get("leg_length_error_mm")),
        })
        if terminated:
            self.done = True
            self.playing = False
        elif truncated:
            self.env.steps = 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self.lock:
                active = self.playing and not self.done
                if active:
                    self._step_locked()
                    speed = max(0.05, float(self.speed))
                else:
                    speed = 1.0
            self._stop.wait(self.env.params.control_dt / speed if active else 0.04)

    def reset(self, seed: int | None = None) -> None:
        with self.lock:
            if seed is not None:
                self.seed = int(seed)
            self.episode += 1
            self.obs, _ = self.env.reset(seed=self.seed + self.episode)
            self.playing = False
            self.done = False
            self.total_steps = 0
            self.history.clear()
            self.last_info = {}
            self.last_reward = 0.0
            self.last_action[:] = 0.0

    def command(self, payload: dict[str, Any]) -> None:
        name = str(payload.get("command", ""))
        with self.lock:
            if name == "play":
                self.playing = not self.done
            elif name == "pause":
                self.playing = False
            elif name == "step":
                self.playing = False
                for _ in range(max(1, int(payload.get("count", 1)))):
                    self._step_locked()
            elif name == "reset":
                self.reset(payload.get("seed"))
            elif name == "speed":
                self.speed = float(np.clip(payload.get("value", 1.0), 0.05, 4.0))
            elif name == "motion":
                self.motion["wheel_speed_rad_s"] = float(np.clip(payload.get("wheel_speed_rad_s", 0.0), -8.0, 8.0))
                self.motion["yaw_rate_rad_s"] = float(np.clip(payload.get("yaw_rate_rad_s", 0.0), -3.0, 3.0))
            elif name == "camera":
                for key in ("azimuth", "elevation", "distance"):
                    if key in payload:
                        self.camera[key] = float(payload[key])
                self._apply_camera()
            elif name == "follow":
                pass

    def state(self) -> dict[str, Any]:
        with self.lock:
            info = self.last_info
            lengths = [finite(info.get("leg_length_left")), finite(info.get("leg_length_right"))]
            currents = [finite(info.get("wheel_current_left")), finite(info.get("wheel_current_right"))]
            pitch_values = [abs(point["pitch_deg"]) for point in self.history]
            drift_values = [abs(point["drift_cm"]) for point in self.history]
            return {
                "playing": bool(self.playing), "done": bool(self.done),
                "episode": int(self.episode), "step": int(self.total_steps),
                "sim_time_s": float(self.total_steps * self.env.params.control_dt),
                "checkpoint": str(self.checkpoint or "manual targets"),
                "mode": "PPO direct targets" if self.model is not None else "manual direct targets",
                "termination": str(info.get("termination_reason", "ready")),
                "reward": float(self.last_reward), "drift_cm": finite(info.get("station_error")) * 100.0,
                "peak_drift_cm": max(drift_values, default=0.0),
                "pitch_deg": finite(info.get("pitch")) * 180.0 / np.pi,
                "pitch_rms_deg": float(np.sqrt(np.mean(np.square(pitch_values)))) if pitch_values else 0.0,
                "height_m": finite(info.get("base_height")), "body_vx": finite(info.get("body_vx_after")),
                "wheel_current_a": currents, "wheel_target_rad_s": [
                    finite(info.get("wheel_target_left")), finite(info.get("wheel_target_right"))
                ],
                "leg_lengths_m": lengths, "leg_length_target_m": finite(info.get("leg_length_target")),
                "leg_length_error_mm": finite(info.get("leg_length_error_mm")),
                "motion": dict(self.motion), "history": list(self.history),
            }

    def frame_jpeg(self) -> bytes:
        with self.lock:
            self.renderer.update_scene(self.env.sim.data, camera=self.render_camera)
            image = Image.fromarray(self.renderer.render())
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85)
            return output.getvalue()


class Handler(BaseHTTPRequestHandler):
    session: ReplaySession

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: Any, status: int = 200) -> None:
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json", status)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/state":
            self.send_json(self.session.state())
        elif path == "/api/frame.jpg":
            self.send_bytes(self.session.frame_jpeg(), "image/jpeg")
        elif path in ("/", "/index.html", "/app.js", "/styles.css"):
            name = "index.html" if path == "/" else path[1:]
            target = WEB_ROOT / name
            self.send_bytes(target.read_bytes(), {
                "index.html": "text/html", "app.js": "text/javascript",
                "styles.css": "text/css",
            }[name])
        else:
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/control":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            self.session.command(json.loads(self.rfile.read(length) or b"{}"))
            self.send_json({"ok": True})
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stand-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--init-scale", type=float, default=0.0)
    parser.add_argument("--deployment-mode", action="store_true",
                        help="Use deployment timing, quantization, and actuator loops")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8087)
    args = parser.parse_args()
    checkpoint = None if args.checkpoint is None else Path(args.checkpoint).expanduser()
    if checkpoint is not None and checkpoint.suffix != ".zip":
        checkpoint = Path(f"{checkpoint}.zip")
    if checkpoint is not None and not checkpoint.exists():
        parser.error(f"checkpoint not found: {checkpoint}")
    # MuJoCo's EGL context is thread-affine.  A single-threaded server keeps
    # frame rendering on the same thread that created the Renderer.
    server = HTTPServer((args.host, args.port), Handler)
    session = ReplaySession(checkpoint, args.stand_level, args.seed, args.init_scale,
                            deployment_mode=args.deployment_mode)
    Handler.session = session
    print(f"replay_mode: {'PPO direct targets' if checkpoint else 'manual direct targets'}", flush=True)
    print(f"deployment_mode: {args.deployment_mode}", flush=True)
    print(f"replay_url: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        session.close()


if __name__ == "__main__":
    main()
