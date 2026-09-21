"""Local interactive web replay for the UZ-05 balance controllers.

Two modes:

* **协同控制器模式（推荐，`--controller-only`）**：不加载任何 checkpoint，
  策略动作恒为 0，由环境内置的「腿 + 轮协同平衡控制器」驱动。这是本次修复的
  成果：零指令下 pitch RMS ≈0.15°、漂移峰值 ≈2.4 cm。
* **策略模式**：加载 PPO checkpoint，与协同控制器叠加（`coord_residual_scale`
  决定策略残差权限）。

Start from the repository root with::

    MUJOCO_GL=egl conda run --no-capture-output -n sim \
      python -u scripts/web_replay_uz05.py --controller-only

The server intentionally uses the same UZ05Env as training/evaluation.
It is a single-user local tool: simulation state and MuJoCo rendering are
protected by one lock, while the HTTP server remains responsive to controls.
"""

from __future__ import annotations

import argparse
import base64
import errno
import json
import os
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Keep the visual model in this one process. The training path strips expensive
# mesh geoms, but a replay should show the actual robot appearance.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("UZ05_VISUAL", "1")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_uz05 import AsymmetricActorCriticPolicy  # noqa: E402
from uz05.env import UZ05Env  # noqa: E402

WEB_ROOT = REPO_ROOT / "web_replay"
# 「站着最稳」的旧 checkpoint（仍有 1.62 Hz 点头、pitch RMS 3.68°），放在回放里
# 正好能和协同控制器做对照。默认走 --controller-only，不需要 checkpoint。
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/ppo_stand_s2_pitchquiet_v1/checkpoint_iter_40.zip"
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


class ReplaySession:
    """Thread-safe policy rollout, camera and state snapshot."""

    def __init__(self, checkpoint: Path | None, stand_level: int, seed: int,
                 init_scale: float = 0.0,
                 lock_stand_differential: bool = True,
                 coord_mix: float = 1.0,
                 coord_residual_scale: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 height_rate_limit: float = 0.90,
                 height_rate_gain: float = 50.0,
                 leg_feedforward_scale: float = 1.0,
                 leg_target_rate: float = 0.65,
                 domain_randomization: bool = False):
        self.lock = threading.RLock()
        # ``checkpoint=None`` ⇒ 纯协同控制器模式（策略动作恒为 0）
        self.controller_only = checkpoint is None
        self.checkpoint = (
            Path(REPO_ROOT / "web_replay" / "coordinated-controller") if checkpoint is None
            else checkpoint.resolve()
        )
        self.stand_level = int(stand_level)
        self.seed = int(seed)
        self.init_scale = float(np.clip(init_scale, 0.0, 1.0))
        self.lock_stand_differential = bool(lock_stand_differential)
        self.episode = 0
        self.playing = False
        self.done = False
        # ``UZ05Env`` reports the configured episode horizon as ``truncated``.
        # Replay should treat that bookkeeping boundary as transparent, so keep
        # a separate monotonically increasing counter for the browser.
        self.total_steps = 0
        self.time_limit_rollovers = 0
        self.speed = 1.0
        # The browser slider is a user request, not a physically instant
        # reference.  Slewing the internal reference prevents a quick drag or
        # a sequence of button clicks from asking the legs to jump 100+ mm in
        # one control tick.  This is deliberately web-only; training still
        # receives its original command curriculum.
        self.leg_target_rate_m_s = max(0.005, float(leg_target_rate))
        self._wall_elapsed_s = 0.0
        self._run_started_s: float | None = None
        self._next_deadline_s: float | None = None
        self.last_action = np.zeros(6, dtype=np.float32)
        self.last_policy_action = np.zeros(6, dtype=np.float32)
        self.last_reward = 0.0
        self.last_info: dict[str, Any] = {}
        # 与验收统计（_pitch_sq 等）保持**同一个窗口长度**，否则"漂移峰值/   
        # pitch 峰值"和"RMS"口径不一致 —— 初始瞬态会被短窗口丢掉（踩过：
        # 800 点窗口时显示 drift_peak=0.001 cm，而同一时刻 RMS 统计里
        # 还留着 2.4 cm 的起始瞬态）。
        self.history: deque[dict[str, float]] = deque(maxlen=3000)
        # 验收统计（滚动窗口）：与 accept_stand.py 同口径
        self._pitch_sq: deque[float] = deque(maxlen=3000)
        self._prate_sq: deque[float] = deque(maxlen=3000)
        self._current_sq: deque[float] = deque(maxlen=3000)
        self._current_delta_sq: deque[float] = deque(maxlen=3000)
        self._prev_current: float | None = None
        self.camera = {
            "azimuth": 135.0,
            "elevation": -18.0,
            "distance": 1.15,
            "lookat": [0.0, 0.0, 0.12],
            "follow": True,
        }
        self._frame_bytes = b""
        self._frame_dirty = True
        self._frame_ready = threading.Event()
        self._render_wakeup = threading.Event()

        # ★ 协同控制器：coord_mix 是它的权限（1 = 全权）。控制器在 env.step
        #   内部叠加，所以纯控制器模式下只要把策略动作喂 0 即可。
        self.env = UZ05Env(stage="stand", stand_level=self.stand_level,
                           assist_scale=0.0, seed=self.seed,
                           init_scale=self.init_scale,
                           coord_mix=float(np.clip(coord_mix, 0.0, 1.0)),
                           coord_residual_scale=tuple(coord_residual_scale),
                           height_rate_limit=height_rate_limit,
                           height_rate_gain=height_rate_gain,
                           height_retract_rate_damping=0.65,
                           height_low_target_brake_damping_scale=1.0,
                           height_reference_jump_reset_m=0.020,
                           height_brake_error_m=0.015,
                           height_target_rate_feedforward_scale=0.35,
                           height_hold_error_m=0.006,
                           height_hold_rate_limit=0.30,
                           height_hold_rate_gain=40.0,
                           height_filter_alpha=0.75,
                           leg_feedforward_scale=leg_feedforward_scale)
        # A replay is meant to expose a responsive manual height control.  The
        # main extension limit stays below the training limit, while the
        # retract profile slows the final approach to avoid body drop/overshoot.
        # Keep replay aligned with the validated dynamic-height profile.  The
        # old web-only override used full retract feed-forward (1.0), which is
        # unsafe after a 0.27 -> 0.15 m direction reversal: the mechanism's
        # inertia carries it through the lower limit before feedback can brake.
        # A small retract feed-forward keeps the transition responsive while
        # leaving most of the braking authority to measured leg velocity.
        # Replay-only fast retract profile: increase the response speed after
        # the no-snap fix, while adding damping and reducing retract feedforward
        # so the faster path does not recreate the low-height overshoot.
        self.env.coordinated.p.height_retract_rate_limit = 1.10
        self.env.coordinated.p.height_retract_slow_rate_limit = 0.90
        # Start braking 20 mm before the low target.  The extra 5 mm compared
        # with the old web profile is intentional: it absorbs the mechanism's
        # reflected inertia before the 0.15 m lower boundary.
        self.env.coordinated.p.height_retract_slow_error_m = 0.020
        self.env.coordinated.p.height_retract_rate_gain = 70.0
        self.env.coordinated.p.height_retract_feedforward_scale = 0.10
        self.env.coordinated.p.height_rate_damping = 0.5
        self.env.coordinated.p.height_retract_rate_damping = 0.65
        self.env.coordinated.p.height_low_target_brake_damping_scale = 1.0
        self.env.coordinated.p.height_rate_brake_threshold = 0.005
        self.env.coordinated.p.height_reference_jump_reset_m = 0.020
        self.env.coordinated.p.height_brake_error_m = 0.015
        self.env.coordinated.p.height_target_rate_feedforward_scale = 0.35
        self.env.coordinated.p.height_hold_error_m = 0.006
        self.env.coordinated.p.height_hold_rate_limit = 0.30
        self.env.coordinated.p.height_hold_rate_gain = 40.0
        self.env.coordinated.p.height_filter_alpha = 0.75
        # 回放要验证当前机器人/控制器本身；训练用的域随机化会把单次
        # 网页切换变成不同机构参数，造成不可复现的急降和漂移。
        self.env.params.domain_randomization.enabled = bool(domain_randomization)
        nominal_leg = float(self.env.coordinated.p.leg_length_ref_default)
        self.motion = {
            "leg_length_m": nominal_leg,
            "wheel_speed_rad_s": 0.0,
            "yaw_rate_rad_s": 0.0,
        }
        # The browser can have several POST requests in flight while a range
        # input is dragged.  Keep a monotonic sequence so a delayed packet
        # cannot overwrite a newer slider value and make the robot appear to
        # move backwards.  Older clients without a sequence remain supported.
        self._last_motion_seq = -1
        self.model = None
        if not self.controller_only:
            self.model = PPO.load(str(self.checkpoint), custom_objects={
                "policy_class": AsymmetricActorCriticPolicy,
            }, device="cpu")
        self.obs, _ = self.env.reset(seed=self.seed)
        # reset() samples and pre-positions a leg-length target.  The UI
        # command must start from that actual pose; resetting it to the
        # nominal 0.184 m would create a fake first-step height jump.
        self.motion["leg_length_m"] = float(np.mean(self.env.sim.leg_lengths()))
        self._leg_control_target_m = float(self.motion["leg_length_m"])
        self._apply_motion_locked()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="uz05-replay", daemon=True)
        self._thread.start()
        self._render_thread = threading.Thread(
            target=self._render_loop, name="uz05-render", daemon=True
        )
        self._render_thread.start()
        self._frame_ready.wait(timeout=10.0)

    def close(self) -> None:
        self._stop.set()
        self._render_wakeup.set()
        self._thread.join(timeout=2.0)
        self._render_thread.join(timeout=3.0)
        with self.lock:
            self.env.close()

    def _rpy(self) -> np.ndarray:
        quat = self.env.sim.base_quat
        return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz")

    def _step_locked(self) -> None:
        if self.done:
            self._set_playing_locked(False)
            return
        # 手动只改腿长时，腿长过渡必须由同一套闭环独立完成；即使加载了
        # checkpoint，也不要让策略在过渡期间叠加隐藏的腿/轮动作。
        # 这样“网页改腿长”的响应与 controller-only 验证完全一致。
        if self.model is None or self._manual_leg_motion_active_locked():
            action = np.zeros(6, dtype=np.float32)
        else:
            action, _ = self.model.predict(self.obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
        self.last_policy_action = action.copy()
        self._apply_motion_locked()
        if self.lock_stand_differential and not self.env.command_active and not self._manual_motion_active_locked():
            # Mirror the environment's zero-command mode: expose the actual
            # applied action in the replay instead of showing raw unused leg
            # and differential outputs from the policy.
            action[1] = 0.0
            action[2:] = 0.0
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.last_action = action.copy()
        self.last_reward = float(reward)
        self.last_info = dict(info)
        # ---- 验收统计（与 accept_stand.py 同口径的滚动窗口）----
        self._pitch_sq.append(float(info.get("pitch", 0.0)) ** 2)
        self._prate_sq.append(float(info.get("pitch_rate", 0.0)) ** 2)
        current = _finite(info.get("wheel_current_left"))
        self._current_sq.append(current * current)
        if self._prev_current is not None:
            self._current_delta_sq.append((current - self._prev_current) ** 2)
        self._prev_current = current
        if terminated:
            # A real physical termination remains terminal and stops playback.
            self.done = True
        elif truncated:
            # The training horizon is not a physical failure. Preserve qpos,
            # qvel, contacts, phase, history, and policy observation, then
            # start a fresh accounting window inside the same simulation.
            self.env.steps = 0
            self.time_limit_rollovers += 1
            self.last_info["termination_reason"] = "running"
            self.last_info["episode_steps"] = 0
            self.done = False
        else:
            self.done = False
        self._frame_dirty = True
        rpy = self._rpy()
        self.history.append({
            "step": float(self.total_steps),
            "drift_cm": 100.0 * abs(float(info.get("station_error", 0.0))),
            "pitch_deg": np.degrees(float(rpy[1])),
            "height_m": float(self.env.sim.data.qpos[2]),
            # 腿偏置（rad）与轮电流（A）：用来在时间轴上直接看"分工"
            "leg_offset_rad": _finite(info.get("coord_leg_offset")),
            "wheel_current_a": _finite(info.get("wheel_current_left")),
        })
        if self.done:
            self.playing = False

    def _manual_motion_active_locked(self) -> bool:
        nominal = float(self.env.params.robot.nominal_leg_length)
        return bool(
            abs(self.motion["wheel_speed_rad_s"]) > 0.01
            or abs(self.motion["yaw_rate_rad_s"]) > 0.01
            or abs(self.motion["leg_length_m"] - nominal) > 0.002
        )

    def _manual_leg_motion_active_locked(self) -> bool:
        nominal = float(self.env.params.robot.nominal_leg_length)
        return abs(self.motion["leg_length_m"] - nominal) > 0.002

    def _apply_motion_locked(self) -> None:
        """把回放控件映射到训练接口。

        * 腿长：直接走 ``env.set_leg_length_command()``（腿长环的目标），
          而不是只改高度命令 —— 腿长环打开时这样才真正驱动机构伸缩。
        * 轮速/偏航：映射成机体速度命令与偏航命令。
        """
        robot = self.env.params.robot
        rng = self.env.leg_length_range
        lo = rng[0] if rng else robot.leg_length_min
        hi = rng[1] if rng else robot.leg_length_max
        requested = float(np.clip(self.motion["leg_length_m"], lo, hi))
        self.motion["leg_length_m"] = requested
        current = float(np.clip(getattr(self, "_leg_control_target_m", requested), lo, hi))
        max_step = self.leg_target_rate_m_s * float(self.env.params.control_dt)
        self._leg_control_target_m = current + float(np.clip(
            requested - current, -max_step, max_step
        ))
        self.env.set_leg_length_command(self._leg_control_target_m)
        # The checkpoint command is body velocity [m/s], while the UI exposes
        # wheel angular speed [rad/s]. Convert using the measured wheel radius.
        vx = float(self.motion["wheel_speed_rad_s"]) * float(robot.wheel_radius)
        self.env.command_target[0] = float(np.clip(vx, -0.6, 0.6))
        self.env.command_target[2] = float(np.clip(self.motion["yaw_rate_rad_s"], -3.0, 3.0))

    def _set_playing_locked(self, value: bool) -> None:
        value = bool(value) and not self.done
        now = time.perf_counter()
        if value and not self.playing:
            self._run_started_s = now
            self._next_deadline_s = now
        elif not value and self.playing and self._run_started_s is not None:
            self._wall_elapsed_s += max(0.0, now - self._run_started_s)
            self._run_started_s = None
            self._next_deadline_s = None
        self.playing = value

    def _loop(self) -> None:
        control_dt = float(self.env.params.control_dt)
        while not self._stop.is_set():
            with self.lock:
                active = self.playing and not self.done
                if active:
                    self._step_locked()
                speed = max(0.05, self.speed)
            if active:
                # Deadline pacing prevents inference/render overhead from
                # accumulating as one full extra period on every step.
                interval = control_dt / speed
                with self.lock:
                    now = time.perf_counter()
                    if self._next_deadline_s is None:
                        self._next_deadline_s = now
                    self._next_deadline_s += interval
                    delay = self._next_deadline_s - now
                    if delay < -4.0 * interval:
                        self._next_deadline_s = now
                        delay = 0.0
                if delay > 0.0:
                    self._stop.wait(delay)
            else:
                with self.lock:
                    self._next_deadline_s = None
                self._stop.wait(0.04)

    def reset(self, seed: int | None = None) -> None:
        with self.lock:
            if seed is not None:
                self.seed = int(seed)
            self.episode += 1
            self.obs, _ = self.env.reset(seed=self.seed + self.episode)
            self.last_action[:] = 0.0
            self.last_policy_action[:] = 0.0
            self.last_reward = 0.0
            self.last_info = {}
            self.history.clear()
            self._pitch_sq.clear()
            self._prate_sq.clear()
            self._current_sq.clear()
            self._current_delta_sq.clear()
            self._prev_current = None
            self.total_steps = 0
            self.time_limit_rollovers = 0
            self._wall_elapsed_s = 0.0
            self._run_started_s = None
            self._next_deadline_s = None
            # Keep the slider target continuous with the pose produced by
            # reset().  The next manual slider change is then a real relative
            # height transition from the current robot, not from nominal.
            self.motion["leg_length_m"] = float(np.mean(self.env.sim.leg_lengths()))
            self._leg_control_target_m = float(self.motion["leg_length_m"])
            self.motion["wheel_speed_rad_s"] = 0.0
            self.motion["yaw_rate_rad_s"] = 0.0
            self._apply_motion_locked()
            self.done = False
            self._set_playing_locked(False)
            self._frame_dirty = True
            self._render_wakeup.set()

    def command(self, payload: dict[str, Any]) -> None:
        command = str(payload.get("command", ""))
        with self.lock:
            if command == "play":
                self._set_playing_locked(True)
            elif command == "pause":
                self._set_playing_locked(False)
            elif command == "step":
                self.playing = False
                for _ in range(max(1, min(100, int(payload.get("count", 1))))):
                    self._step_locked()
                    if self.done:
                        break
            elif command == "reset":
                self.reset(payload.get("seed"))
            elif command == "speed":
                self.speed = float(np.clip(float(payload.get("value", 1.0)), 0.05, 8.0))
            elif command == "motion":
                if "motion_seq" in payload:
                    try:
                        motion_seq = int(payload["motion_seq"])
                    except (TypeError, ValueError):
                        motion_seq = None
                    if motion_seq is not None:
                        if motion_seq <= self._last_motion_seq:
                            return
                        self._last_motion_seq = motion_seq
                robot = self.env.params.robot
                if "leg_length_m" in payload:
                    self.motion["leg_length_m"] = float(np.clip(
                        _finite(payload.get("leg_length_m"), robot.nominal_leg_length),
                        robot.leg_length_min, robot.leg_length_max,
                    ))
                if "wheel_speed_rad_s" in payload:
                    self.motion["wheel_speed_rad_s"] = float(np.clip(
                        _finite(payload.get("wheel_speed_rad_s")), -8.0, 8.0
                    ))
                if "yaw_rate_rad_s" in payload:
                    self.motion["yaw_rate_rad_s"] = float(np.clip(
                        _finite(payload.get("yaw_rate_rad_s")), -3.0, 3.0
                    ))
                self._apply_motion_locked()
            elif command == "motion_reset":
                self.motion["leg_length_m"] = float(np.mean(self.env.sim.leg_lengths()))
                self._leg_control_target_m = float(self.motion["leg_length_m"])
                self.motion["wheel_speed_rad_s"] = 0.0
                self.motion["yaw_rate_rad_s"] = 0.0
                self._apply_motion_locked()
            elif command == "camera":
                self._update_camera_locked(payload)
                self._frame_dirty = True
                self._render_wakeup.set()
            elif command == "follow":
                self.camera["follow"] = bool(payload.get("value", True))
                self._frame_dirty = True
                self._render_wakeup.set()

    def _update_camera_locked(self, payload: dict[str, Any]) -> None:
        for name, lo, hi in (("azimuth", -360.0, 360.0),
                             ("elevation", -80.0, 25.0),
                             ("distance", 0.45, 3.0)):
            if name in payload:
                self.camera[name] = float(np.clip(float(payload[name]), lo, hi))
        if "preset" in payload:
            presets = {
                "three-quarter": (135.0, -18.0, 1.15),
                "side": (90.0, -12.0, 1.05),
                "front": (180.0, -10.0, 1.05),
                "top": (135.0, -55.0, 1.35),
            }
            if payload["preset"] in presets:
                self.camera["azimuth"], self.camera["elevation"], self.camera["distance"] = presets[payload["preset"]]

    def _render_loop(self) -> None:
        """Own the EGL context and render copied simulation state.

        MuJoCo's renderer is thread-affine. Copying into a private MjData keeps
        the HTTP and simulation threads independent from OpenGL calls.
        """
        renderer = None
        try:
            # The supplied MJCF keeps the offscreen framebuffer at 640 px wide.
            renderer = mujoco.Renderer(self.env.sim.model, height=420, width=640)
            render_data = mujoco.MjData(self.env.sim.model)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            from PIL import Image
            import io
            while not self._stop.is_set():
                with self.lock:
                    active = self.playing and not self.done
                    dirty = self._frame_dirty
                    if not active and not dirty:
                        wait = True
                    else:
                        wait = False
                        # This MuJoCo Python build does not expose mj_copyData;
                        # qpos/qvel/ctrl plus forward is sufficient for rendering.
                        render_data.qpos[:] = self.env.sim.data.qpos
                        render_data.qvel[:] = self.env.sim.data.qvel
                        render_data.ctrl[:] = self.env.sim.data.ctrl
                        mujoco.mj_forward(self.env.sim.model, render_data)
                        camera.azimuth = float(self.camera["azimuth"])
                        camera.elevation = float(self.camera["elevation"])
                        camera.distance = float(self.camera["distance"])
                        lookat = np.asarray(self.camera["lookat"], dtype=np.float64).copy()
                        if self.camera["follow"]:
                            lookat[0] = float(self.env.sim.data.qpos[0])
                        camera.lookat[:] = lookat
                        self._frame_dirty = False
                if wait:
                    self._render_wakeup.wait(timeout=0.5)
                    self._render_wakeup.clear()
                    continue
                renderer.update_scene(render_data, camera=camera)
                pixels = renderer.render()
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="JPEG", quality=86, optimize=True)
                with self.lock:
                    self._frame_bytes = buffer.getvalue()
                self._frame_ready.set()
                # Cap visual refresh to 10 FPS while allowing simulation to run
                # at its configured playback speed.
                self._render_wakeup.wait(timeout=0.10)
                self._render_wakeup.clear()
        finally:
            if renderer is not None:
                renderer.close()

    def frame_jpeg(self) -> bytes:
        with self.lock:
            if not self._frame_bytes:
                raise RuntimeError("render frame is not ready")
            return self._frame_bytes

    def state(self) -> dict[str, Any]:
        with self.lock:
            sim = self.env.sim
            rpy = self._rpy()
            tilt_abs_deg = float(np.degrees(max(abs(float(rpy[0])), abs(float(rpy[1])))))
            tilt_limit_deg = float(np.degrees(self.env.tilt_limit))
            info = self.last_info
            qpos = np.asarray(sim.data.qpos[:3], dtype=np.float64)
            station = qpos[:2] - self.env.nominal_xy
            current = [
                _finite(info.get("wheel_current_left")),
                _finite(info.get("wheel_current_right")),
            ]
            wheel_vel = [
                _finite(info.get("wheel_vel_left")),
                _finite(info.get("wheel_vel_right")),
            ]
            leg_lengths = [
                _finite(info.get("leg_length_left"), float(sim.leg_lengths()[0])),
                _finite(info.get("leg_length_right"), float(sim.leg_lengths()[1])),
            ]
            body_contact = bool(_finite(info.get("body_contact")) > 0.5)
            airborne = bool(_finite(info.get("airborne")) > 0.5)
            leg_mean = float(np.mean(leg_lengths))
            leg_floor = float(self.env.params.robot.nominal_leg_length * self.env.params.leg_length_fail_ratio)
            height_ok = bool(float(qpos[2]) >= 0.80 * self.env.params.robot.nominal_stand_height)
            upright_ok = bool(tilt_abs_deg <= tilt_limit_deg)
            physical_failed = bool(
                body_contact or not height_ok or not upright_ok or leg_mean < leg_floor
            )
            grid_count = sum(
                1 for geom_id in range(self.env.sim.model.ngeom)
                if (mujoco.mj_id2name(
                    self.env.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                ) or "").startswith("replay_tile_")
            )
            wall_elapsed = self._wall_elapsed_s
            if self._run_started_s is not None:
                wall_elapsed += max(0.0, time.perf_counter() - self._run_started_s)
            sim_time = float(self.total_steps * self.env.params.control_dt)
            return {
                "checkpoint": str(self.checkpoint),
                "playing": bool(self.playing),
                "done": bool(self.done),
                "speed": float(self.speed),
                "episode": int(self.episode),
                "step": int(self.total_steps),
                "sim_time_s": sim_time,
                "wall_time_s": float(wall_elapsed),
                "real_time_factor": float(sim_time / wall_elapsed) if wall_elapsed > 1e-6 else 0.0,
                "time_limit_rollovers": int(self.time_limit_rollovers),
                "position": [float(v) for v in qpos],
                "station_error": [float(v) for v in station],
                "drift_cm": 100.0 * float(np.linalg.norm(station)),
                "tail_drift_cm": 100.0 * _finite(info.get("station_tail_mean")),
                "peak_drift_cm": 100.0 * _finite(info.get("station_max_abs")),
                "roll_deg": float(np.degrees(rpy[0])),
                "pitch_deg": float(np.degrees(rpy[1])),
                "yaw_deg": float(np.degrees(rpy[2])),
                "height_m": float(qpos[2]),
                "tilt_abs_deg": tilt_abs_deg,
                "tilt_limit_deg": tilt_limit_deg,
                "upright_ok": upright_ok,
                "height_ok": height_ok,
                "body_contact": body_contact,
                "airborne": airborne,
                "leg_mean_m": leg_mean,
                "leg_floor_m": leg_floor,
                "failed": physical_failed,
                "failure_reason": (
                    "body_contact" if body_contact else
                    "height_limit" if not height_ok else
                    "tilt_limit" if not upright_ok else
                    "leg_length_limit" if leg_mean < leg_floor else ""
                ),
                "grid_geom_count": int(grid_count),
                "body_vx": _finite(info.get("body_vx")),
                "body_vy": _finite(info.get("body_vy")),
                "wheel_current_a": current,
                "wheel_velocity": wheel_vel,
                "leg_lengths_m": leg_lengths,
                "action": [float(v) for v in self.last_action],
                "reward": float(self.last_reward),
                "termination": str(info.get("termination_reason", "ready")),
                "assist": _finite(info.get("assist_scale")),
                "history": list(self.history),
                "camera": dict(self.camera),
                "init_scale": float(self.env.init_scale),
                "stand_differential_locked": bool(self.lock_stand_differential),
                "policy_action": [float(v) for v in self.last_policy_action],
                "motion": dict(self.motion),
                "height_controller": {
                    "target_rate_m_s": float(self.leg_target_rate_m_s),
                    "diff_limit_margin_scale": float(
                        self.env.coordinated.p.height_diff_limit_margin_scale),
                    "diff_limit_action": float(
                        self.env.coordinated.p.height_diff_limit),
                    "target_rate_feedforward_scale": float(
                        self.env.coordinated.p.height_target_rate_feedforward_scale),
                    "extension_rate_limit_action_s": float(
                        self.env.coordinated.p.height_rate_limit),
                    "retract_rate_limit_action_s": float(
                        self.env.coordinated.p.height_retract_rate_limit),
                    "retract_slow_rate_limit_action_s": float(
                        self.env.coordinated.p.height_retract_slow_rate_limit),
                    "retract_slow_error_m": float(
                        self.env.coordinated.p.height_retract_slow_error_m),
                    "brake_error_m": float(
                        self.env.coordinated.p.height_brake_error_m),
                    "hold_error_m": float(
                        self.env.coordinated.p.height_hold_error_m),
                    "hold_rate_limit_action_s": float(
                        self.env.coordinated.p.height_hold_rate_limit),
                    "hold_rate_gain": float(
                        self.env.coordinated.p.height_hold_rate_gain),
                    "height_filter_alpha": float(
                        self.env.coordinated.p.height_filter_alpha),
                    "reference_jump_reset_m": float(
                        self.env.coordinated.p.height_reference_jump_reset_m),
                    "retract_rate_gain": float(
                        self.env.coordinated.p.height_retract_rate_gain),
                    "retract_rate_damping": float(
                        self.env.coordinated.p.height_retract_rate_damping),
                    "low_target_brake_damping_scale": float(
                        self.env.coordinated.p.height_low_target_brake_damping_scale),
                    "retract_feedforward_scale": float(
                        self.env.coordinated.p.height_retract_feedforward_scale),
                },
                # ---- 协同控制器状态（本项目的核心）----
                "controller_only": bool(self.controller_only),
                "coord_mix": float(self.env.coord_mix),
                "coord_leg_offset_rad": _finite(info.get("coord_leg_offset")),
                "coord_current_a": _finite(info.get("coord_current")),
                "coord_residual_scale": list(self.env.coord_residual_scale),
                # Keep the public command equal to the requested UI target so
                # the displayed error answers the user's question.  The
                # slower internal reference is exposed separately for tuning.
                "leg_length_cmd_m": float(self.motion["leg_length_m"]),
                "leg_length_control_m": float(self.env.leg_length_command),
                "leg_length_error_requested_mm": float(
                    (leg_mean - self.motion["leg_length_m"]) * 1000.0
                ),
                "leg_length_error_control_mm": float(
                    (leg_mean - self.env.leg_length_command) * 1000.0
                ),
                "leg_target_rate_m_s": float(self.leg_target_rate_m_s),
                "leg_length_range_m": list(self.env.leg_length_range or []),
                "leg_length_meas_m": float(np.mean(self.env.sim.leg_lengths())),
                # ---- 验收指标（滚动窗口，与 accept_stand.py 同口径）----
                "window_steps": len(self._pitch_sq),
                "pitch_rms_deg": float(np.degrees(np.sqrt(np.mean(self._pitch_sq))))
                if self._pitch_sq else 0.0,
                "pitch_rate_rms": float(np.sqrt(np.mean(self._prate_sq)))
                if self._prate_sq else 0.0,
                "current_rms_a": float(np.sqrt(np.mean(self._current_sq)))
                if self._current_sq else 0.0,
                "current_delta_rms_a": float(np.sqrt(np.mean(self._current_delta_sq)))
                if self._current_delta_sq else 0.0,
                "pitch_peak_deg": float(max(
                    (abs(v) for v in (d["pitch_deg"] for d in self.history)), default=0.0)),
                "drift_peak_cm": float(max(
                    (d["drift_cm"] for d in self.history), default=0.0)),
            }


class ReplayHandler(BaseHTTPRequestHandler):
    session: ReplaySession

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep the terminal useful: only API errors are printed by the caller.
        if self.path.startswith("/api/") and self.command not in ("GET",):
            print(f"[web] {self.command} {self.path}")

    def _send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        except BrokenPipeError:
            # Browsers routinely cancel an in-flight JPEG when the next frame
            # is ready.  That is normal polling behavior, not a replay error.
            return

    def _send_json(self, payload: Any, status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(content, "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            target = WEB_ROOT / "index.html"
            self._send_bytes(target.read_bytes(), "text/html; charset=utf-8")
        elif path == "/app.js":
            target = WEB_ROOT / "app.js"
            self._send_bytes(target.read_bytes(), "text/javascript; charset=utf-8")
        elif path == "/styles.css":
            target = WEB_ROOT / "styles.css"
            self._send_bytes(target.read_bytes(), "text/css; charset=utf-8")
        elif path == "/api/state":
            self._send_json(self.session.state())
        elif path == "/api/frame.jpg":
            try:
                self._send_bytes(self.session.frame_jpeg(), "image/jpeg")
            except Exception as exc:  # pragma: no cover - browser-visible error path
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/control":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self.session.command(payload)
            self._send_json({"ok": True})
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


class ReplayHTTPServer(ThreadingHTTPServer):
    # Allow a quick restart after the previous process has closed. An actively
    # listening process still correctly reports EADDRINUSE below.
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UZ-05 交互式网页回放（腿 + 轮协同平衡控制器 / PPO 策略）")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="PPO checkpoint；--controller-only 时忽略")
    parser.add_argument("--controller-only", action="store_true",
                        help="★ 不加载 checkpoint，纯协同平衡控制器驱动（推荐）。"
                             "零指令下 pitch RMS ≈0.15°、漂移峰值 ≈2.4 cm。")
    parser.add_argument("--coord-mix", type=float, default=1.0,
                        help="协同控制器权限（1 = 全权；0 = 完全靠策略）")
    parser.add_argument("--coord-residual-scale", type=float, nargs=3,
                        default=(0.0, 0.0, 0.0), metavar=("WHEEL", "DIFF", "LEG"),
                        help="策略残差通道缩放（仅策略模式生效）；手动改腿长时默认全由控制器跟踪")
    parser.add_argument("--height-rate-limit", type=float, default=0.90,
                        help="网页腿长差模速率；默认 0.90，快速回放档")
    parser.add_argument("--height-rate-gain", type=float, default=50.0,
                        help="网页腿长反馈增益；默认 50.0，快速回放档")
    parser.add_argument("--leg-feedforward-scale", type=float, default=1.0,
                        help="网页腿长前馈比例；默认 1.0，配合速度阻尼快速到位")
    parser.add_argument("--leg-target-rate", type=float, default=0.65,
                        help="网页请求目标的内部参考斜率（m/s）；默认 0.65，快速回放档")
    parser.add_argument("--domain-randomization", action="store_true",
                        help="回放时启用域随机化（默认关闭，便于复现当前机构）")
    parser.add_argument("--stand-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--init-scale", type=float, default=0.0,
                        help="初始扰动比例；0 = 额定静止站姿")
    parser.add_argument(
        "--no-lock-stand-differential", dest="lock_stand_differential",
        action="store_false", default=True,
        help="replay the raw differential wheel action (diagnostic; may yaw-spin)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8087)
    args = parser.parse_args()
    checkpoint: Path | None = None
    if not args.controller_only:
        checkpoint = Path(args.checkpoint).expanduser()
        if checkpoint.suffix != ".zip":
            checkpoint = Path(f"{checkpoint}.zip")
        if not checkpoint.exists():
            parser.error(f"checkpoint not found: {checkpoint}\n"
                         f"（或加 --controller-only 用协同控制器回放，不需要 checkpoint）")

    # Bind before loading the visual model. This makes a duplicate launch fail
    # immediately and avoids leaving a MuJoCo/render thread behind on error.
    try:
        server = ReplayHTTPServer((args.host, args.port), ReplayHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"port {args.port} is already in use; open the existing replay at "
                f"http://{args.host}:{args.port}/ or choose another port with --port"
            ) from exc
        raise
    session = ReplaySession(
        checkpoint, args.stand_level, args.seed, args.init_scale,
        args.lock_stand_differential, args.coord_mix,
        tuple(args.coord_residual_scale),
        args.height_rate_limit, args.height_rate_gain,
        args.leg_feedforward_scale,
        args.leg_target_rate,
        args.domain_randomization,
    )
    ReplayHandler.session = session
    mode = "协同平衡控制器（无策略）" if args.controller_only else f"PPO 策略 + 协同控制器"
    print(f"replay_mode: {mode}", flush=True)
    print(f"replay_checkpoint: {session.checkpoint}", flush=True)
    print(f"replay_coord_mix: {args.coord_mix}", flush=True)
    print(f"replay_url: http://{args.host}:{args.port}/", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
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
