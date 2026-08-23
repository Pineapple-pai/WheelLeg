import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb

from torch.func import vmap
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

CHECKPOINT_ROOT = os.environ.get(
    "WHEELLEG_CHECKPOINT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints")),
)


TRACKED_TRAIN_KEYS = (
    "stats.return",
    "stats.episode_len",
    "reward",
    "stats.uprightness",
    "stats.roll_error",
    "stats.pitch_error",
    "stats.yaw_error",
    "stats.height",
    "stats.height_error",
    "stats.terminated_roll",
    "stats.terminated_pitch",
    "stats.terminated_height",
    "stats.terminated_xy",
    "stats.forward_velocity",
    "stats.velocity_error",
    "stats.forward_progress",
    "stats.displacement_progress",
    "stats.position_error",
    "stats.velocity_alignment",
    "stats.quiet_state",
    "stats.stand_reward",
    "stats.settle_reward",
    "stats.leg_neutral_reward",
    "stats.command_vx",
    "stats.command_yaw_rate",
)

TRACKED_UPDATE_KEYS = (
    "policy_loss",
    "value_loss",
    "entropy",
    "actor_grad_norm",
    "critic_grad_norm",
    "explained_var",
)

TRACKED_EVAL_KEYS = (
    "eval/stats.return",
    "eval/stats.episode_len",
    "eval/stats.uprightness",
    "eval/stats.height",
    "eval/stats.height_error",
    "eval/stats.terminated_roll",
    "eval/stats.terminated_pitch",
    "eval/stats.terminated_height",
    "eval/stats.terminated_xy",
    "eval/stats.forward_velocity",
    "eval/stats.velocity_error",
    "eval/stats.quiet_state",
    "eval/stats.leg_neutral_reward",
    "eval/forward/stats.episode_len",
    "eval/forward/stats.uprightness",
    "eval/forward/stats.height",
    "eval/forward/stats.terminated_roll",
    "eval/forward/stats.terminated_pitch",
    "eval/forward/stats.terminated_height",
    "eval/forward/stats.terminated_xy",
    "eval/forward/stats.forward_velocity",
    "eval/forward/stats.velocity_error",
    "eval/backward/stats.episode_len",
    "eval/backward/stats.uprightness",
    "eval/backward/stats.height",
    "eval/backward/stats.terminated_roll",
    "eval/backward/stats.terminated_pitch",
    "eval/backward/stats.terminated_height",
    "eval/backward/stats.terminated_xy",
    "eval/backward/stats.forward_velocity",
    "eval/backward/stats.velocity_error",
)

TRACKED_EVAL_STAND_KEYS = (
    "eval/stats.return",
    "eval/stats.episode_len",
    "eval/stats.uprightness",
    "eval/stats.height",
    "eval/stats.height_error",
    "eval/stats.terminated_roll",
    "eval/stats.terminated_pitch",
    "eval/stats.terminated_height",
    "eval/stats.terminated_xy",
    "eval/stats.forward_velocity",
    "eval/stats.velocity_error",
)

TRACKED_EVAL_FORWARD_KEYS = (
    "eval/forward/stats.episode_len",
    "eval/forward/stats.uprightness",
    "eval/forward/stats.height",
    "eval/forward/stats.terminated_roll",
    "eval/forward/stats.terminated_pitch",
    "eval/forward/stats.terminated_height",
    "eval/forward/stats.terminated_xy",
    "eval/forward/stats.forward_velocity",
    "eval/forward/stats.velocity_error",
)

TRACKED_EVAL_BACKWARD_KEYS = (
    "eval/backward/stats.episode_len",
    "eval/backward/stats.uprightness",
    "eval/backward/stats.height",
    "eval/backward/stats.terminated_roll",
    "eval/backward/stats.terminated_pitch",
    "eval/backward/stats.terminated_height",
    "eval/backward/stats.terminated_xy",
    "eval/backward/stats.forward_velocity",
    "eval/backward/stats.velocity_error",
)

OBS_STATE_INDICES = {
    "roll": 0,
    "pitch": 1,
    "yaw": 2,
    "height": 3,
    "vx": 4,
    "vy": 5,
    "vz": 6,
    "roll_rate": 7,
    "pitch_rate": 8,
    "yaw_rate": 9,
    "command_vx": 14,
    "command_yaw_rate": 15,
}


def _scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().item()
    elif isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, int)):
        return value
    return None


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_block(title, items):
    print(title)
    for key, value in items:
        print(f"  {key}: {_format_value(value)}")
    print(flush=True)


def _compact_summary(cfg, total_frames, max_iters, eval_interval, save_interval, frames_per_batch):
    checkpoint_version = str(cfg.get("checkpoint_version", "v1"))
    fields = [
        ("task", cfg.task.name),
        ("algo", cfg.algo.name),
        ("headless", cfg.headless),
        ("num_envs", cfg.task.env.num_envs),
        ("checkpoint_version", checkpoint_version),
        ("frames_per_batch", frames_per_batch),
        ("total_frames", total_frames),
        ("max_iters", max_iters),
        ("eval_interval", eval_interval),
        ("save_interval", save_interval),
        ("wandb", str(cfg.wandb.mode)),
    ]
    _print_block("run", fields)


def _checkpoint_dir(cfg):
    version = str(cfg.get("checkpoint_version", "v1"))
    return os.path.join(CHECKPOINT_ROOT, version)


def _checkpoint_path(cfg, name):
    return os.path.join(_checkpoint_dir(cfg), name)


def _select_fields(info, keys):
    items = []
    for key in keys:
        if key not in info:
            continue
        value = _mean_tensor(info[key])
        if value is None:
            continue
        items.append((key, value))
    return items


def _mean_tensor(value):
    if isinstance(value, torch.Tensor) and value.numel() > 1:
        return value.detach().float().mean().item()
    return _scalar(value)


def _current_state_from_obs(obs: torch.Tensor):
    obs = obs.detach().float()
    flat = obs.reshape(-1, obs.shape[-1])
    mean_obs = flat.mean(dim=0)
    return [(name, mean_obs[idx].item()) for name, idx in OBS_STATE_INDICES.items() if idx < mean_obs.numel()]


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    os.makedirs(_checkpoint_dir(cfg), exist_ok=True)

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

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

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
    _compact_summary(cfg, total_frames, max_iters, eval_interval, save_interval, frames_per_batch)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE,
        command_vx: float=None,
        prefix: str="eval/stats",
    ):

        old_eval_command_vx = float(base_env.eval_command_vx)
        if command_vx is not None:
            base_env.eval_command_vx = float(command_vx)
        base_env.enable_render(not cfg.headless)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = None if cfg.headless else RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            rollout_kwargs = dict(
                max_steps=base_env.max_episode_length,
                policy=policy,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
            if render_callback is not None:
                rollout_kwargs["callback"] = render_callback
            trajs = env.rollout(**rollout_kwargs)
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            prefix + "." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        if render_callback is not None:
            info["recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        base_env.eval_command_vx = old_eval_command_vx
        return info

    def evaluate_commands():
        info = evaluate(command_vx=float(cfg.task.eval_command_vx), prefix="eval/stats")
        move_vx = abs(float(cfg.task.get("eval_move_command_vx", 0.0)))
        if move_vx > 0.0:
            info.update(evaluate(seed=1, command_vx=-move_vx, prefix="eval/forward/stats"))
            info.update(evaluate(seed=2, command_vx=move_vx, prefix="eval/backward/stats"))
        return info

    env.train()
    start_time = time.time()
    for i, data in enumerate(collector):
        elapsed = max(time.time() - start_time, 1e-6)
        info = {
            "env_frames": collector._frames,
            "rollout_fps": collector._fps,
            "elapsed_s": elapsed,
            "train_iter": i + 1,
        }
        info["reward"] = _mean_tensor(data[("next", "agents", "reward")])
        info.update({f"state.{k}": v for k, v in _current_state_from_obs(data[("next", "agents", "observation")])})
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        update_info = policy.train_op(data.to_tensordict())
        info.update(update_info)

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate_commands())
            env.train()
            base_env.train()

        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = _checkpoint_path(cfg, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        run.log(info)
        state_items = [
            ("iter", f"{i + 1}/{max_iters}" if max_iters > 0 else i + 1),
            ("frames", f"{collector._frames}/{total_frames}" if total_frames > 0 else collector._frames),
            ("fps", f"{collector._fps:.2f}"),
            ("elapsed_s", f"{elapsed:.1f}"),
        ]
        state_items.extend(_select_fields(info, [
            "state.roll",
            "state.pitch",
            "state.yaw",
            "state.height",
            "state.vx",
            "state.vy",
            "state.vz",
            "state.roll_rate",
            "state.pitch_rate",
            "state.yaw_rate",
            "state.command_vx",
            "state.command_yaw_rate",
        ]))
        _print_block("state", state_items)

        train_items = _select_fields(info, TRACKED_TRAIN_KEYS)
        if train_items:
            _print_block("train", train_items)

        update_items = _select_fields(info, TRACKED_UPDATE_KEYS)
        if update_items:
            _print_block("update", update_items)

        eval_stand_items = _select_fields(info, TRACKED_EVAL_STAND_KEYS)
        if eval_stand_items:
            _print_block("eval_stand", eval_stand_items)

        eval_forward_items = _select_fields(info, TRACKED_EVAL_FORWARD_KEYS)
        if eval_forward_items:
            _print_block("eval_forward", eval_forward_items)

        eval_backward_items = _select_fields(info, TRACKED_EVAL_BACKWARD_KEYS)
        if eval_backward_items:
            _print_block("eval_backward", eval_backward_items)

        if max_iters > 0 and i >= max_iters - 1:
            break

    info = {"env_frames": collector._frames}
    if eval_interval > 0:
        logging.info(f"Final Eval at {collector._frames} steps.")
        info.update(evaluate_commands())
    run.log(info)
    final_items = [("env_frames", info["env_frames"])]
    final_items.extend(_select_fields(info, TRACKED_EVAL_KEYS))
    _print_block("final", final_items)

    try:
        ckpt_path = _checkpoint_path(cfg, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        if str(cfg.wandb.mode).lower() != "disabled":
            wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
