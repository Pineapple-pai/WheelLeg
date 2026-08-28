import logging
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

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


TRACKED_STATE_STATS_KEYS = (
    "stats.return",
    "stats.episode_len",
    "reward",
    "stats.roll_error",
    "stats.height_error",
    "stats.done",
    "stats.forward_velocity",
    "stats.velocity_error",
    "stats.velocity_alignment",
    "stats.forward_progress",
    "stats.wheel_action_magnitude",
    "stats.translation_wheel_diff_penalty",
    "stats.translation_action_diff_penalty",
    "stats.yaw_rate_drift_penalty",
    "stats.balance_baseline_action",
    "stats.balance_feedforward_action",
    "stats.policy_wheel_residual",
    "stats.command_action_alignment",
    "stats.tracking_lin_vel_soft",
    "stats.wheel_command_tracking",
    "stats.normalized_velocity_tracking",
    "stats.active_velocity_error_penalty",
    "stats.lin_vel_tracking_square_penalty",
    "stats.wheel_residual_scale",
    "stats.leg_policy_scale",
    "stats.leg_pos_error",
    "stats.leg_vel_error",
    "stats.leg_action_magnitude",
    "stats.stand_reward",
    "stats.command_vx",
    "stats.zero_command_fraction",
    "stats.translation_command_fraction",
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
    "eval/stats.terminated_nan",
    "eval/stats.done",
    "eval/stats.forward_velocity",
    "eval/stats.velocity_error",
    "eval/stats.position_error",
    "eval/stats.yaw_error",
    "eval/stats.yaw_rate_error",
    "eval/stats.balance_baseline_action",
    "eval/stats.balance_yaw_action",
    "eval/stats.policy_wheel_residual",
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
    "eval/stats.position_error",
    "eval/stats.yaw_error",
    "eval/stats.yaw_rate_error",
    "eval/stats.balance_baseline_action",
    "eval/stats.balance_yaw_action",
    "eval/stats.policy_wheel_residual",
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

TRACKED_EVAL_ROBUST_KEYS = (
    "eval/robust/stats.return",
    "eval/robust/stats.episode_len",
    "eval/robust/stats.uprightness",
    "eval/robust/stats.height",
    "eval/robust/stats.roll_error",
    "eval/robust/stats.yaw_error",
    "eval/robust/stats.forward_velocity",
    "eval/robust/stats.position_error",
    "eval/robust/stats.balance_baseline_action",
    "eval/robust/stats.balance_yaw_action",
    "eval/robust/stats.policy_wheel_residual",
)

TRACKED_EVAL_TURN_KEYS = (
    "eval/turn_left/stats.episode_len",
    "eval/turn_left/stats.uprightness",
    "eval/turn_left/stats.height",
    "eval/turn_left/stats.yaw_rate_error",
    "eval/turn_left/stats.terminated_roll",
    "eval/turn_left/stats.terminated_height",
    "eval/turn_right/stats.episode_len",
    "eval/turn_right/stats.uprightness",
    "eval/turn_right/stats.height",
    "eval/turn_right/stats.yaw_rate_error",
    "eval/turn_right/stats.terminated_roll",
    "eval/turn_right/stats.terminated_height",
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
        ("train_stage", cfg.task.get("train_stage", "stand")),
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
    state = [
        (name, mean_obs[idx].item())
        for name, idx in OBS_STATE_INDICES.items()
        if idx < mean_obs.numel()
    ]
    for name in ("command_vx", "command_yaw_rate"):
        idx = OBS_STATE_INDICES[name]
        if idx < flat.shape[-1]:
            state.append((f"command_abs_{name.removeprefix('command_')}", flat[:, idx].abs().mean().item()))
    return state


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs import resolve_env_class

    run = init_wandb(cfg)
    setproctitle(run.name)
    os.makedirs(_checkpoint_dir(cfg), exist_ok=True)

    env_class = resolve_env_class(cfg.task.name)
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
    # SyncDataCollector resets the environment during construction. Initialize
    # curricula first so the first episode does not receive full disturbances.
    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(0)
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
        command_yaw_rate: float=None,
        prefix: str="eval/stats",
        disturbances: bool=False,
    ):

        old_eval_command_vx = float(base_env.eval_command_vx)
        old_eval_command_yaw_rate = float(base_env.eval_command_yaw_rate)
        if command_vx is not None:
            base_env.eval_command_vx = float(command_vx)
        if command_yaw_rate is not None:
            base_env.eval_command_yaw_rate = float(command_yaw_rate)
        base_env.enable_render(not cfg.headless)
        base_env.eval()
        env.eval()
        env.set_seed(seed)
        old_disturbances_in_eval = bool(base_env.disturbances_in_eval)
        base_env.disturbances_in_eval = bool(disturbances)

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
        base_env.eval_command_yaw_rate = old_eval_command_yaw_rate
        base_env.disturbances_in_eval = old_disturbances_in_eval
        return info

    def evaluate_commands():
        info = {}
        if bool(cfg.task.get("eval_nominal", True)):
            info.update(
                evaluate(command_vx=float(cfg.task.eval_command_vx), prefix="eval/stats")
            )
        move_vx = abs(float(cfg.task.get("eval_move_command_vx", 0.0)))
        if hasattr(base_env, "_current_lin_vel_range"):
            current_vx_range = base_env._current_lin_vel_range()
            move_vx = min(
                move_vx,
                max(abs(float(current_vx_range[0])), abs(float(current_vx_range[1]))),
            )
        if move_vx > 0.0:
            info.update(evaluate(seed=1, command_vx=move_vx, prefix="eval/forward/stats"))
            info.update(evaluate(seed=2, command_vx=-move_vx, prefix="eval/backward/stats"))
        turn_rate = abs(float(cfg.task.get("eval_turn_command_yaw_rate", 0.0)))
        if hasattr(base_env, "_current_max_yaw_rate"):
            turn_rate = min(turn_rate, abs(float(base_env._current_max_yaw_rate())))
        if turn_rate > 0.0:
            info.update(evaluate(
                seed=4,
                command_vx=0.0,
                command_yaw_rate=turn_rate,
                prefix="eval/turn_left/stats",
            ))
            info.update(evaluate(
                seed=5,
                command_vx=0.0,
                command_yaw_rate=-turn_rate,
                prefix="eval/turn_right/stats",
            ))
        if bool(cfg.task.get("eval_disturbances", False)):
            info.update(
                evaluate(
                    seed=3,
                    command_vx=0.0,
                    prefix="eval/robust/stats",
                    disturbances=True,
                )
            )
        return info

    env.train()
    if hasattr(base_env, "set_training_progress"):
        base_env.set_training_progress(0)
    start_time = time.time()
    last_eval_frames = -1
    for i, data in enumerate(collector):
        if hasattr(base_env, "set_training_progress"):
            base_env.set_training_progress(collector._frames)
        elapsed = max(time.time() - start_time, 1e-6)
        info = {
            "env_frames": collector._frames,
            "rollout_fps": collector._fps,
            "elapsed_s": elapsed,
            "train_iter": i + 1,
        }
        info["reward"] = _mean_tensor(data[("next", "agents", "reward")])
        if hasattr(base_env, "_current_wheel_residual_scale"):
            info["state.wheel_residual_scale"] = float(
                base_env._current_wheel_residual_scale
            )
        if hasattr(base_env, "_current_leg_policy_scale"):
            info["state.leg_policy_scale"] = float(base_env._current_leg_policy_scale)
        if hasattr(base_env, "_current_disturbance_scale"):
            info["state.disturbance_scale"] = float(
                base_env._current_disturbance_scale
            )
        if hasattr(base_env, "_current_translation_position_hold_scale"):
            info["state.translation_position_hold_scale"] = float(
                base_env._current_translation_position_hold_scale
            )
        if hasattr(base_env, "_current_lin_vel_range"):
            command_vx_range = base_env._current_lin_vel_range()
            info["state.command_vx_min"] = float(command_vx_range[0])
            info["state.command_vx_max"] = float(command_vx_range[1])
        if hasattr(base_env, "_current_max_yaw_rate"):
            info["state.command_yaw_rate_max"] = float(
                base_env._current_max_yaw_rate()
            )
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

        if save_interval > 0 and (i + 1) % save_interval == 0:
            try:
                ckpt_path = _checkpoint_path(cfg, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        if eval_interval > 0 and (i + 1) % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate_commands())
            last_eval_frames = collector._frames
            env.train()
            base_env.train()

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
            "state.height",
            "state.vy",
            "state.yaw_rate",
            "state.command_vx",
            "state.command_abs_vx",
            "state.command_vx_min",
            "state.command_vx_max",
            "state.wheel_residual_scale",
            "state.leg_policy_scale",
        ]))
        state_items.extend(_select_fields(info, TRACKED_STATE_STATS_KEYS))
        _print_block("state", state_items)

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

        eval_robust_items = _select_fields(info, TRACKED_EVAL_ROBUST_KEYS)
        if eval_robust_items:
            _print_block("eval_robust", eval_robust_items)

        eval_turn_items = _select_fields(info, TRACKED_EVAL_TURN_KEYS)
        if eval_turn_items:
            _print_block("eval_turn", eval_turn_items)

        if max_iters > 0 and i >= max_iters - 1:
            break

    final_info = {"env_frames": collector._frames}
    if eval_interval > 0 and last_eval_frames != collector._frames:
        logging.info(f"Final Eval at {collector._frames} steps.")
        final_info.update(evaluate_commands())
    elif last_eval_frames == collector._frames:
        final_info.update({k: v for k, v in info.items() if k.startswith("eval/")})
    info = final_info
    run.log(info)
    final_items = [("env_frames", info["env_frames"])]
    final_items.extend(_select_fields(info, TRACKED_EVAL_KEYS))
    final_items.extend(_select_fields(info, TRACKED_EVAL_ROBUST_KEYS))
    final_items.extend(_select_fields(info, TRACKED_EVAL_TURN_KEYS))
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
