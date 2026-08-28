import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import hydra
import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from omni_drones import init_simulation_app
from omni_drones.learning import ALGOS
from omni_drones.utils.torchrl.transforms import ravel_composite


FILE_PATH = os.path.dirname(__file__)


@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    from omni_drones.envs import resolve_env_class


    env_class = resolve_env_class(cfg.task.name)
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]
    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    env = TransformedEnv(base_env, Compose(*transforms)).eval()
    env.set_seed(cfg.seed)

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )

    steps = int(cfg.get("diagnose_steps", 180))
    print_every = int(cfg.get("diagnose_print_every", 10))
    td = env.reset()

    print(
        "config, "
        f"checkpoint_version={cfg.checkpoint_version}, "
        f"roll_target={float(cfg.task.get('roll_target', 0.0)):.4f}, "
        f"termination_roll={float(cfg.task.get('termination_roll', 0.0)):.4f}, "
        f"min_height={float(cfg.task.get('min_height', 0.0)):.4f}, "
        f"base_height_target={float(cfg.task.get('base_height_target', 0.0)):.4f}, "
        f"usd_path={cfg.task.robot.get('usd_path', '')}",
        flush=True,
    )
    print(
        "step, command_vx, roll, pitch, yaw, height, raw_vy, signed_forward_v, lateral_vx, wz, wheel_l, wheel_r, "
        "wheel_l_rad_s, wheel_r_rad_s, expected_no_slip_v, slip_v, slip_ratio, "
        "act_l, act_r, leg_pos_norm, leg_vel_norm, leg_act_norm, reward, done",
        flush=True,
    )
    wheel_radius = float(cfg.task.robot.get("wheel_radius", 0.06))
    wheel_rolling_sign = float(cfg.task.robot.get("wheel_rolling_sign", -1.0))
    forward_axis_sign = float(cfg.task.get("forward_axis_sign", 1.0))
    prev_wheel = None
    prev_action = None
    prev_action_delta = None
    max_abs_roll = 0.0
    sum_wheel_delta = 0.0
    sum_wheel_sign_flip = 0.0
    sum_action_accel = 0.0
    sum_abs_slip = 0.0
    sum_slip_ratio = 0.0
    measured_steps = 0
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for step in range(steps):
            td = policy(td)
            action = td[("agents", "action")].detach().clone()
            td = env.step(td)

            obs = td[("next", "agents", "observation")][0, 0].detach().cpu()
            action_vec = action[0, 0].detach().cpu()
            reward = td[("next", "agents", "reward")][0, 0, 0].detach().cpu().item()
            done = td[("next", "done")][0, 0].detach().cpu().item()
            leg_pos_norm = base_env.robot.leg_pos[0, 0].norm().item()
            leg_vel_norm = base_env.robot.leg_vel[0, 0].norm().item()
            leg_act_norm = action_vec[2:].norm().item() if action_vec.numel() > 2 else 0.0
            wheel_rad_s = base_env.robot.wheel_vel[0, 0].detach().cpu()
            expected_no_slip_v = wheel_rolling_sign * wheel_radius * wheel_rad_s.mean().item()
            signed_forward_v = forward_axis_sign * obs[5].item()
            slip_v = expected_no_slip_v - signed_forward_v
            slip_ratio = abs(slip_v) / max(abs(expected_no_slip_v), 0.05)
            sum_abs_slip += abs(slip_v)
            sum_slip_ratio += slip_ratio
            values = {
                "command_vx": base_env.command[0, 0].detach().cpu().item(),
                "roll": obs[0].item(),
                "pitch": obs[1].item(),
                "yaw": obs[2].item(),
                "height": obs[3].item(),
                "raw_vy": obs[5].item(),
                "signed_forward_v": signed_forward_v,
                "lateral_vx": obs[4].item(),
                "wz": obs[9].item(),
                "wheel_l": obs[10].item(),
                "wheel_r": obs[11].item(),
                "wheel_l_rad_s": wheel_rad_s[0].item(),
                "wheel_r_rad_s": wheel_rad_s[1].item(),
                "expected_no_slip_v": expected_no_slip_v,
                "slip_v": slip_v,
                "slip_ratio": slip_ratio,
                "act_l": action_vec[0].item(),
                "act_r": action_vec[1].item(),
                "leg_pos_norm": leg_pos_norm,
                "leg_vel_norm": leg_vel_norm,
                "leg_act_norm": leg_act_norm,
                "reward": reward,
                "done": bool(done),
            }
            wheel = torch.tensor([values["wheel_l"], values["wheel_r"]])
            wheel_action = action_vec[:2].clone()
            max_abs_roll = max(max_abs_roll, abs(values["roll"]))
            if prev_wheel is not None:
                wheel_delta = torch.linalg.vector_norm(wheel - prev_wheel).item()
                sign_flip = (
                    ((wheel * prev_wheel) < 0.0)
                    & ((wheel.abs() + prev_wheel.abs()) > 0.035)
                ).float().sum().item()
                sum_wheel_delta += wheel_delta
                sum_wheel_sign_flip += sign_flip
            if prev_action is not None:
                action_delta = wheel_action - prev_action
                if prev_action_delta is not None:
                    sum_action_accel += torch.linalg.vector_norm(action_delta - prev_action_delta).item()
                prev_action_delta = action_delta
            prev_wheel = wheel
            prev_action = wheel_action
            measured_steps += 1
            if step % print_every == 0 or done:
                print(
                    f"{step}, {values['command_vx']:.4f}, "
                    f"{values['roll']:.4f}, {values['pitch']:.4f}, {values['yaw']:.4f}, "
                    f"{values['height']:.4f}, {values['raw_vy']:.4f}, "
                    f"{values['signed_forward_v']:.4f}, {values['lateral_vx']:.4f}, {values['wz']:.4f}, "
                    f"{values['wheel_l']:.4f}, {values['wheel_r']:.4f}, "
                    f"{values['wheel_l_rad_s']:.4f}, {values['wheel_r_rad_s']:.4f}, "
                    f"{values['expected_no_slip_v']:.4f}, {values['slip_v']:.4f}, "
                    f"{values['slip_ratio']:.4f}, "
                    f"{values['act_l']:.4f}, {values['act_r']:.4f}, "
                    f"{values['leg_pos_norm']:.4f}, {values['leg_vel_norm']:.4f}, "
                    f"{values['leg_act_norm']:.4f}, "
                    f"{values['reward']:.4f}, {values['done']}",
                    flush=True,
                )
            td = td["next"]
            if done:
                break

    denom = max(measured_steps - 1, 1)
    accel_denom = max(measured_steps - 2, 1)
    print(
        "summary, "
        f"steps={measured_steps}, max_abs_roll={max_abs_roll:.4f}, "
        f"mean_wheel_delta={sum_wheel_delta / denom:.4f}, "
        f"mean_wheel_sign_flip={sum_wheel_sign_flip / denom:.4f}, "
        f"mean_wheel_action_accel={sum_action_accel / accel_denom:.4f}, "
        f"mean_abs_slip_v={sum_abs_slip / max(measured_steps, 1):.4f}, "
        f"mean_slip_ratio={sum_slip_ratio / max(measured_steps, 1):.4f}",
        flush=True,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
