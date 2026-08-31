import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

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

@hydra.main(config_path=os.path.join(FILE_PATH, "../cfg"), config_name="train", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)

    from omni_drones import init_simulation_app
    simulation_app = init_simulation_app(cfg)

    from omni_drones.envs import resolve_env_class
    from omni_drones.learning import ALGOS
    from omni_drones.utils.torchrl import RenderCallback
    from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
    from torchrl.envs.utils import set_exploration_type, ExplorationType

    env_class = resolve_env_class(cfg.task.name)
    base_env = env_class(cfg, headless=cfg.headless)
    env = TransformedEnv(base_env, Compose(InitTracker())).train()
    env.set_seed(cfg.seed)

    if cfg.algo.get("checkpoint_path", None) is None:
        checkpoint_version = str(cfg.get("checkpoint_version", "v1"))
        cfg.algo.checkpoint_path = _latest_checkpoint_path(checkpoint_version)
    print(f"Using checkpoint: {cfg.algo.checkpoint_path}")

    policy = ALGOS[cfg.algo.name.lower()](
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device
    )

    base_env.enable_render(not cfg.headless)
    base_env.eval()
    env.eval()
    render_callback = None if cfg.headless else RenderCallback(interval=1)

    total_frames = cfg.get("total_frames", 20000)
    pbar = tqdm(total=total_frames)
    frames = 0
    start_time = time.time()

    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        td = env.reset()
        while frames < total_frames:
            td = policy(td)
            # 提取动作张量 (形状可能是 (1,6) 或 (1,1,6))
            actions = td[("agents", "action")]
            # 确保形状为 (num_envs, 6)
            if actions.dim() == 3:
                actions = actions.squeeze(1)  # (num_envs, 6)
            # 反转轮子动作（索引0和1）
            actions[:, 0] = -actions[:, 0]
            actions[:, 1] = -actions[:, 1]

            # 直接使用普通 Tensor 执行 step（与 test_wheel.py 一致）
            td = env.step(actions)  # 注意：直接传张量

            frames += env.num_envs
            pbar.update(env.num_envs)
            if render_callback is not None:
                render_callback(env)
            if td[("next", "done")].any():
                td = env.reset()
            else:
                td = td["next"]
            if frames % 500 == 0:
                elapsed = time.time() - start_time
                pbar.set_postfix({"fps": frames / elapsed, "frames": frames})

    pbar.close()
    simulation_app.close()

if __name__ == "__main__":
    main()