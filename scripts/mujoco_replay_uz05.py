"""Native MuJoCo viewer replay for the direct PPO UZ-05 policy."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("UZ05_VISUAL", "1")

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_uz05 import AsymmetricActorCriticPolicy  # noqa: E402
from uz05.env import UZ05Env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/direct_ppo_wheelpi_stand_s2_finetune_v1/checkpoint.zip",
    )
    parser.add_argument("--stand-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=15000)
    parser.add_argument("--init-scale", type=float, default=0.40)
    parser.add_argument(
        "--deployment-mode", action=argparse.BooleanOptionalAction, default=True,
        help="Use deployment delays/quantization (default: enabled)",
    )
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    if checkpoint.suffix != ".zip":
        checkpoint = Path(f"{checkpoint}.zip")
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        parser.error(f"checkpoint not found: {checkpoint}")

    env = UZ05Env(
        stage="stand", stand_level=args.stand_level, seed=args.seed,
        init_scale=args.init_scale, deployment_mode=args.deployment_mode,
    )
    model = PPO.load(
        str(checkpoint), custom_objects={"policy_class": AsymmetricActorCriticPolicy},
        device="cpu",
    )
    obs, _ = env.reset(seed=args.seed)
    state = {"obs": obs, "playing": True, "reset": False, "episode": 0}

    def on_key(key: int) -> None:
        # MuJoCo's native viewer sends GLFW key codes.
        if key == 32:  # space: pause/resume
            state["playing"] = not state["playing"]
            print(f"viewer: {'playing' if state['playing'] else 'paused'}", flush=True)
        elif key in (ord("R"), ord("r")):
            state["playing"] = False
            state["reset"] = True

    def key_callback(key: int, action: int, _mods: int) -> None:
        if action == 1:  # GLFW_PRESS
            on_key(key)

    print(f"checkpoint: {checkpoint}", flush=True)
    print(
        f"MuJoCo native viewer | stand-level={args.stand_level} "
        f"init-scale={args.init_scale:.2f} deployment-mode={args.deployment_mode}",
        flush=True,
    )
    print("Controls: Space = pause/resume, R = reset episode", flush=True)

    last_report = time.monotonic()
    try:
        with mujoco.viewer.launch_passive(
            env.sim.model, env.sim.data, key_callback=key_callback,
        ) as viewer:
            while viewer.is_running():
                loop_start = time.monotonic()
                if state["reset"]:
                    state["episode"] += 1
                    state["obs"], _ = env.reset(seed=args.seed + state["episode"])
                    state["reset"] = False
                    viewer.sync()

                if state["playing"]:
                    state["obs"], _reward, terminated, truncated, info = env.step(
                        np.asarray(model.predict(state["obs"], deterministic=True)[0],
                                   dtype=np.float32).reshape(-1)
                    )
                    if terminated or truncated:
                        state["playing"] = False
                        print(
                            f"episode ended: {info.get('termination_reason', 'truncated')} "
                            f"at step {info.get('episode_steps', 0)}; press R to reset",
                            flush=True,
                        )
                    viewer.sync()

                    if time.monotonic() - last_report >= 2.0:
                        print(
                            "step={episode_steps} height={base_height:.4f}m "
                            "drift={station_error_cm:.2f}cm pitch={pitch_deg:.2f}deg "
                            "leg_error={leg_length_error_mm:.1f}mm airborne={airborne:.0f}".format(
                                episode_steps=info.get("episode_steps", 0),
                                base_height=info.get("base_height", 0.0),
                                station_error_cm=100.0 * info.get("station_error", 0.0),
                                pitch_deg=np.degrees(info.get("pitch", 0.0)),
                                leg_length_error_mm=info.get("leg_length_error_mm", 0.0),
                                airborne=info.get("airborne", 0.0),
                            ),
                            flush=True,
                        )
                        last_report = time.monotonic()
                else:
                    viewer.sync()
                    time.sleep(0.02)

                if args.realtime and state["playing"]:
                    remaining = env.params.control_dt - (time.monotonic() - loop_start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    main()
