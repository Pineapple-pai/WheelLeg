import os
import sys

import hydra
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.transforms import Compose, InitTracker, TransformedEnv


FILE_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_PATH, ".."))
sys.path.insert(0, PROJECT_ROOT)


@hydra.main(config_path=os.path.join(FILE_PATH, "../cfg"), config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    target_action = float(cfg.get("actuator_test", {}).get("target_action", 0.25))
    steps = int(cfg.get("actuator_test", {}).get("steps", 160))
    print_every = int(cfg.get("actuator_test", {}).get("print_every", 20))

    from omni_drones import init_simulation_app

    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs import resolve_env_class
    from omni_drones.utils.torchrl import RenderCallback

    env_class = resolve_env_class(cfg.task.name)
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(cfg.seed)
    base_env.enable_render(not cfg.headless)
    render_callback = None if cfg.headless else RenderCallback(interval=1)

    td = env.reset()
    device = env.device
    num_envs = base_env.num_envs
    action_dim = base_env.robot.action_spec.shape[-1]
    action = torch.zeros(num_envs, 1, action_dim, device=device)
    action[:, 0, 0] = target_action
    action[:, 0, 1] = target_action
    action_td = TensorDict({"agents": {"action": action}}, batch_size=[num_envs], device=device)

    print(
        "step, action_l, action_r, wheel_l_rad_s, wheel_r_rad_s, forward_v, y, roll, pitch, height, done",
        flush=True,
    )
    with torch.no_grad():
        for step in range(steps):
            td = env.step(action_td)
            if render_callback is not None:
                render_callback(env)

            done = td[("next", "done")]
            if step % print_every == 0 or bool(done[0, 0].detach().cpu().item()):
                wheel_vel = base_env.robot.wheel_vel[0, 0].detach().cpu()
                forward_v = (
                    float(cfg.task.get("forward_axis_sign", 1.0))
                    * base_env.robot.vel_b[0, 0, 1].detach().cpu().item()
                )
                y = base_env.robot.pos[0, 0, 1].detach().cpu().item()
                rpy = base_env.robot.rpy[0, 0].detach().cpu()
                height = base_env.robot.pos[0, 0, 2].detach().cpu().item()
                print(
                    f"{step}, {action[0, 0, 0].item():.4f}, {action[0, 0, 1].item():.4f}, "
                    f"{wheel_vel[0].item():.4f}, {wheel_vel[1].item():.4f}, "
                    f"{forward_v:.4f}, {y:.4f}, {rpy[0].item():.4f}, "
                    f"{rpy[1].item():.4f}, {height:.4f}, {bool(done[0, 0].detach().cpu().item())}",
                    flush=True,
                )

            td = td["next"]
            if done.any():
                env.reset()

    simulation_app.close()


if __name__ == "__main__":
    main()
