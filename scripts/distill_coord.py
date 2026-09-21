"""把腿+轮协同平衡控制器的行为「蒸馏」进 PPO 策略网络（行为克隆）。

动机
----
这个任务对"动作均值"极其敏感：腿的 CoP 权限约 30 (rad/s²)/rad，任何非零的
策略均值都会被放大成宏观漂移。标准 PPO（熵系数 + 高斯探索）会持续把均值
推离 0，实测把标定好的协同控制器逐步推坏（存活率 1.0 → 0.14）。

做法：先用 MSE 监督学习拟合协同控制器（确定性、可复现），得到一个
"接口一致、能直接部署"的策略 checkpoint；再选做极低学习率 + 零熵的微调。

用法::

    python scripts/distill_coord.py --samples 400000 --epochs 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uz05.env import UZ05Env  # noqa: E402
from uz05.spec import ACTOR_OBS_DIM, OBS_DIM, REPO_ROOT  # noqa: E402

from train_uz05 import AsymmetricActorCriticPolicy  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def collect(samples: int, seed: int = 0, max_envs: int = 1):
    """用协同控制器驱动环境，采集 (actor_obs, action) 对。"""
    env = UZ05Env(stage="stand", stand_level=2, assist_scale=0.0, seed=seed,
                  init_scale=1.0, lock_stand_leg_actions=False,
                  stand_leg_action_limit=1.0, coord_mix=1.0)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    ep = 0
    while len(obs_list) < samples:
        action = np.zeros(6)
        _, _, term, trunc, _ = env.step(action)
        # 协同控制器写的动作在 env 内部，这里直接读它下发的归一化动作
        obs_list.append(obs[:ACTOR_OBS_DIM].copy())
        act_list.append(env.previous_action.copy())
        obs = env._obs()
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=seed + ep * 7919)
    env.close()
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--version", default="ppo_stand_coord_bc_v1")
    a = ap.parse_args()

    print(f"[bc] 采集 {a.samples} 样本 ...", flush=True)
    t0 = time.time()
    obs, act = collect(a.samples, seed=a.seed)
    print(f"[bc] 采集完成 {obs.shape} 用时 {time.time() - t0:.1f}s "
          f"action_mean={act.mean(0)} action_std={act.std(0)}", flush=True)

    out = REPO_ROOT / "checkpoints" / a.version / "checkpoint"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([lambda: UZ05Env(stage="stand", stand_level=2, assist_scale=0.0,
                                       seed=a.seed, init_scale=1.0,
                                       lock_stand_leg_actions=False,
                                       stand_leg_action_limit=1.0, coord_mix=1.0)])
    model = PPO(
        AsymmetricActorCriticPolicy, env,
        n_steps=512, batch_size=512, learning_rate=1e-4, n_epochs=1,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        policy_kwargs=dict(actor_dim=ACTOR_OBS_DIM,
                           net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
                           activation_fn=nn.ELU, log_std_init=-3.5),
        device="cpu", seed=a.seed,
    )
    with torch.no_grad():
        model.policy.log_std.data.fill_(-3.5)

    X = torch.as_tensor(obs, device="cpu")
    Y = torch.as_tensor(act, device="cpu")
    opt = torch.optim.Adam(model.policy.parameters(), lr=a.lr)
    n = X.shape[0]
    for epoch in range(a.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            pred = model.policy._predict(X[idx], deterministic=True)
            loss = nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            opt.step()
            tot += float(loss) * idx.numel()
        print(f"[bc] epoch {epoch + 1}/{a.epochs} mse={tot / n:.6f}", flush=True)
        model.save(str(out))

    model.save(str(out))
    (Path(f"{out}_contract.json")).write_text(
        '{\n  "contract_version": "uz05_iface_v2",\n  "stage": "stand",\n'
        f'  "version": "{a.version}",\n  "note": "行为克隆自腿+轮协同平衡控制器",\n'
        f'  "observation_shape": {OBS_DIM},\n  "action_shape": 6,\n'
        '  "interface_frozen": true\n}\n'
    )
    print(f"[bc] saved {out}.zip", flush=True)


if __name__ == "__main__":
    main()
