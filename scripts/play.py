import logging
import os
import time
import weakref

import hydra
import torch

from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import RenderCallback
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
)
from omni_drones.utils.torchrl import EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


FILE_PATH = os.path.dirname(__file__)
CHECKPOINT_ROOT = os.environ.get(
    "WHEELLEG_CHECKPOINT_ROOT",
    os.path.abspath(os.path.join(FILE_PATH, "..", "checkpoints")),
)


def _latest_checkpoint_path(version: str):
    import glob
    version_dir = os.path.join(CHECKPOINT_ROOT, version)
    final_path = os.path.join(version_dir, "checkpoint_final.pt")
    if os.path.exists(final_path):
        return final_path
    candidates = glob.glob(os.path.join(version_dir, "checkpoint_*.pt"))
    if not candidates:
        return None
    def _frame_num(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return -1
    candidates.sort(key=_frame_num)
    return candidates[-1]


class KeyboardCommandCallback:
    def __init__(
        self,
        base_env,
        forward_speed: float,
        smoothing: float = 0.15,
        render_interval: int = 1,
        command_file: str = "",
    ):
        import carb
        import omni.appwindow

        self.base_env = base_env
        self.forward_speed = abs(float(forward_speed))
        self.smoothing = float(smoothing)
        self.render = RenderCallback(interval=render_interval)
        self.target_vx = 0.0
        self.current_vx = 0.0
        self.command_file = command_file
        self._last_command_text = None
        self.carb = carb

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        print(
            f"Keyboard control enabled: W/UP/NUMPAD_8 forward({-self.forward_speed:.3f}), "
            f"S/DOWN/NUMPAD_2 backward({self.forward_speed:.3f}), release to stop."
        )
        if self.command_file:
            print(
                f"Command-file fallback enabled: echo -{self.forward_speed:.3f} > {self.command_file} "
                f"for forward, echo 0 > {self.command_file} to stop."
            )

    def close(self):
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __call__(self, env, *args):
        self._read_command_file()
        self.current_vx += self.smoothing * (self.target_vx - self.current_vx)
        self.base_env.command[..., 0] = self.current_vx
        self.base_env.command[..., 1] = 0.0
        return self.render(env, *args)

    def _on_keyboard_event(self, event, *args, **kwargs):
        key = event.input.name if hasattr(event.input, "name") else str(event.input)
        if event.type == self.carb.input.KeyboardEventType.KEY_PRESS:
            if key in ("W", "UP", "NUMPAD_8"):
                self.target_vx = -self.forward_speed
            elif key in ("S", "DOWN", "NUMPAD_2"):
                self.target_vx = self.forward_speed
        elif event.type == self.carb.input.KeyboardEventType.KEY_RELEASE:
            if key in ("W", "S", "UP", "DOWN", "NUMPAD_8", "NUMPAD_2"):
                self.target_vx = 0.0
        return True

    def _read_command_file(self):
        if not self.command_file:
            return
        try:
            with open(self.command_file) as f:
                text = f.read().strip()
        except FileNotFoundError:
            return
        if not text or text == self._last_command_text:
            return
        try:
            vx = float(text.split()[0])
        except ValueError:
            return
        self._last_command_text = text
        self.target_vx = max(-self.forward_speed, min(self.forward_speed, vx))

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    setproctitle(cfg.task.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # if cfg.task.get("history", False):
    #     # transforms.append(History([("info", "drone_state"), ("info", "prev_action")]))
    #     transforms.append(History([("agents", "observation")]))

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    if cfg.algo.get("checkpoint_path", None) is None:
        checkpoint_version = str(cfg.get("checkpoint_version", "v1"))
        cfg.algo.checkpoint_path = _latest_checkpoint_path(checkpoint_version)
    print(f"Using checkpoint: {cfg.algo.checkpoint_path}")

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    base_env.enable_render(not cfg.headless)
    base_env.eval()
    env.eval()
    keyboard_control = bool(cfg.get("keyboard_control", False))
    if keyboard_control and cfg.headless:
        raise ValueError("keyboard_control=true requires headless=false")
    command_callback = None
    if keyboard_control:
        teleop_speed = cfg.get("teleop_speed", abs(float(cfg.task.eval_command_vx)))
        teleop_smoothing = cfg.get("teleop_smoothing", 0.15)
        teleop_command_file = cfg.get("teleop_command_file", "/tmp/twowheel_cmd.txt")
        command_callback = KeyboardCommandCallback(
            base_env,
            forward_speed=teleop_speed,
            smoothing=teleop_smoothing,
            render_interval=1,
            command_file=teleop_command_file,
        )
        render_callback = command_callback
    else:
        render_callback = None if cfg.headless else RenderCallback(interval=2)
    total_frames = cfg.get("total_frames", base_env.max_episode_length)

    try:
        pbar = tqdm(total=total_frames)
        frames = 0
        start_time = time.time()
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            while frames < total_frames:
                rollout_steps = min(base_env.max_episode_length, total_frames - frames)
                rollout_kwargs = dict(
                    max_steps=rollout_steps,
                    policy=policy,
                    auto_reset=True,
                    break_when_any_done=False,
                    return_contiguous=False,
                )
                if render_callback is not None:
                    rollout_kwargs["callback"] = render_callback
                data = env.rollout(**rollout_kwargs)
                frames += env.num_envs * rollout_steps
                elapsed = max(time.time() - start_time, 1e-6)
                pbar.update(env.num_envs * rollout_steps)

                info = {"env_frames": frames, "rollout_fps": frames / elapsed}
                episode_stats.add(data.to_tensordict())

                if len(episode_stats) >= base_env.num_envs:
                    stats = {
                        "eval/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                        for k, v in episode_stats.pop().items(True, True)
                    }
                    info.update(stats)

                print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

                pbar.set_postfix({"rollout_fps": frames / elapsed, "frames": frames})
        pbar.close()
    finally:
        if command_callback is not None:
            command_callback.close()

    simulation_app.close()


if __name__ == "__main__":
    main()
