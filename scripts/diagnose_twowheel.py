import os

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

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
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
        "step, roll, pitch, yaw, height, vx, wz, wheel_l, wheel_r, "
        "act_l, act_r, leg_pos_norm, leg_vel_norm, leg_act_norm, reward, done",
        flush=True,
    )
    prev_wheel = None
    prev_action = None
    prev_action_delta = None
    max_abs_roll = 0.0
    sum_wheel_delta = 0.0
    sum_wheel_sign_flip = 0.0
    sum_action_accel = 0.0
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
            leg_pos_norm = obs[16:20].norm().item() if obs.numel() >= 24 else 0.0
            leg_vel_norm = obs[20:24].norm().item() if obs.numel() >= 24 else 0.0
            leg_act_norm = action_vec[2:].norm().item() if action_vec.numel() > 2 else 0.0
            values = {
                "roll": obs[0].item(),
                "pitch": obs[1].item(),
                "yaw": obs[2].item(),
                "height": obs[3].item(),
                "vx": obs[4].item(),
                "wz": obs[9].item(),
                "wheel_l": obs[10].item(),
                "wheel_r": obs[11].item(),
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
                    f"{step}, {values['roll']:.4f}, {values['pitch']:.4f}, {values['yaw']:.4f}, "
                    f"{values['height']:.4f}, {values['vx']:.4f}, {values['wz']:.4f}, "
                    f"{values['wheel_l']:.4f}, {values['wheel_r']:.4f}, "
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
        f"mean_wheel_action_accel={sum_action_accel / accel_denom:.4f}",
        flush=True,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
