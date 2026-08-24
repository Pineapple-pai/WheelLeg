import os

import hydra
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from omni_drones import init_simulation_app


FILE_PATH = os.path.dirname(__file__)


@hydra.main(config_path=FILE_PATH, config_name="stand_sweep", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    cfg.task.env.num_envs = 1
    cfg.env.num_envs = 1
    cfg.task.robot.lock_leg_actions = True
    cfg.task.max_init_roll = 0.0
    cfg.task.max_init_pitch = 0.0
    cfg.task.min_height = 0.0
    cfg.task.terminate_on_body_contact = False
    cfg.task.termination_roll = 3.14
    cfg.task.termination_pitch = 3.14
    cfg.task.max_xy = 100.0

    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs.isaac_env import IsaacEnv

    env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=cfg.headless).eval()
    env.reset()

    steps = int(cfg.get("contact_diag_steps", 240))
    roll_kp = float(cfg.get("diag_roll_kp", 0.0))
    roll_kd = float(cfg.get("diag_roll_kd", 0.0))
    forward_kd = float(cfg.get("diag_forward_kd", 0.0))
    action_bias = float(cfg.get("diag_action_bias", 0.0))
    action = torch.zeros(env.num_envs, 1, env.robot.action_spec.shape[-1], device=env.device)
    td = TensorDict({"agents": {"action": action}}, batch_size=[env.num_envs], device=env.device)
    print_interval = int(cfg.get("diag_print_interval", 25))
    with torch.no_grad():
        for step in range(steps):
            env.robot.get_state()
            common_action = (
                action_bias
                + roll_kp * env.robot.rpy[..., 0]
                + roll_kd * env.robot.vel_b[..., 3]
                + forward_kd * env.robot.vel_b[..., 1]
            ).clamp(-1.0, 1.0)
            action[..., 0] = common_action
            action[..., 1] = common_action
            env.step(td)
            if step % print_interval == 0 or step == steps - 1:
                env.robot.get_state()
                print(
                    f"step={step:04d}, "
                    f"roll={env.robot.rpy[0, 0, 0].item():+.4f}, "
                    f"height={env.robot.pos[0, 0, 2].item():.4f}, "
                    f"vy={env.robot.vel_b[0, 0, 1].item():+.4f}, "
                    f"roll_rate={env.robot.vel_b[0, 0, 3].item():+.4f}, "
                    f"action={action[0, 0, 0].item():+.4f}",
                    flush=True,
                )

    env.robot.get_state()
    print(
        "pose, "
        f"roll={env.robot.rpy[0, 0, 0].item():.4f}, "
        f"pitch={env.robot.rpy[0, 0, 1].item():.4f}, "
        f"yaw={env.robot.rpy[0, 0, 2].item():.4f}, "
        f"height={env.robot.pos[0, 0, 2].item():.4f}, "
        f"vy={env.robot.vel_b[0, 0, 1].item():.4f}, "
        f"roll_rate={env.robot.vel_b[0, 0, 3].item():.4f}, "
        f"action={action[0, 0, 0].item():.4f}",
        flush=True,
    )

    get_contacts = getattr(env.robot._view, "get_net_contact_forces", None)
    if get_contacts is None:
        print("contact_forces_unavailable", flush=True)
        simulation_app.close()
        return

    forces = get_contacts(clone=True)[0, 0].detach().cpu()
    names = list(getattr(env.robot._view, "_body_names", []))
    print("contacts", flush=True)
    for idx, force in enumerate(forces):
        norm = float(force.norm())
        if norm <= 0.05:
            continue
        name = names[idx] if idx < len(names) else str(idx)
        values = ", ".join(f"{float(v):.4f}" for v in force.tolist())
        print(f"  {idx}: {name}: [{values}], norm={norm:.4f}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
