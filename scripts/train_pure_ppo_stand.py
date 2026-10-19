#!/usr/bin/env python3
"""Train a fresh, unassisted PPO actor on the fixed UZ-05 standing task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from uz05.policy import AsymmetricActorCriticPolicy
from uz05.pure_ppo_contract import (
    CONTRACT_VERSION,
    contract_payload,
    current_commit,
    model_xml_sha256,
    source_fingerprint,
)
from uz05.pure_ppo_eval import evaluate_episode, summarize_evaluation
from uz05.pure_ppo_stand import (
    ACTOR_OBS_DIM,
    EPISODE_STEPS,
    OBS_DIM,
    POLICY_DT_S,
    PurePPOStandEnv,
    pure_ppo_config,
)

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = REPO_ROOT / "artifacts" / CONTRACT_VERSION
RUN_ROOT = REPO_ROOT / "runs" / CONTRACT_VERSION
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / CONTRACT_VERSION
MAX_UPDATES = 800
ROLLOUT_STEPS = 512
VALIDATION_SEEDS = tuple(range(10000, 10050))

METRIC_FIELDS = [
    "update",
    "timesteps",
    "rollout_reward_mean",
    "completed_episodes",
    "episode_return_mean",
    "episode_length_mean",
    "terminated_episodes",
    "truncated_episodes",
    "termination_reasons_json",
    "pitch_rms_rad_mean",
    "pitch_peak_rad_max",
    "pitch_rate_rms_rad_s_mean",
    "drift_max_m_max",
    "wheel_left_contact_fraction_mean",
    "wheel_right_contact_fraction_mean",
    "body_contact_steps_sum",
    "action_mean_json",
    "action_std_json",
    "action_saturation_fraction_json",
    "wheel_current_saturation_fraction_json",
    "wheel_torque_saturation_fraction_json",
    "leg_torque_saturation_fraction_json",
    "wheel_saturation_fraction_json",
    "reward_components_mean_json",
    "reward_raw_mean_json",
    "approx_kl",
    "clip_fraction",
    "entropy_loss",
    "value_loss",
    "explained_variance",
    "policy_gradient_loss",
    "learning_rate",
    "seed",
    "commit",
    "config_sha256",
]


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _scalar(value: Any) -> float:
    if value is None:
        return float("nan")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    return float(array.mean()) if array.size else float("nan")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _check_training_gates() -> dict[str, Any]:
    expected_model = model_xml_sha256()
    expected_source = source_fingerprint()
    reports = {}
    for name in ("preflight", "smoke", "dry_run"):
        path = GATE_DIR / f"{name}.json"
        if not path.is_file():
            raise RuntimeError(f"training gate is missing: {path}; complete preflight first")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise RuntimeError(f"training gate {name} did not pass: {report.get('status')}")
        if report.get("model_xml_sha256") != expected_model:
            raise RuntimeError(f"training gate {name} was produced for a different MJCF")
        if report.get("source_fingerprint_sha256") != expected_source:
            raise RuntimeError(
                f"training gate {name} is stale: audited control/model code has changed"
            )
        reports[name] = report
    return reports


def _make_env(seed: int, rank: int):
    def _init():
        return PurePPOStandEnv(seed=seed + rank * 100_003)
    return _init


def _make_vec_env(n_envs: int, seed: int, *, subproc: bool = True):
    env_fns = [_make_env(seed, rank) for rank in range(n_envs)]
    if n_envs == 1 or not subproc:
        vec = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns, start_method="fork")
    return VecMonitor(vec)


def _policy_kwargs() -> dict[str, Any]:
    return {
        "actor_dim": ACTOR_OBS_DIM,
        "net_arch": {"pi": [256, 128, 64], "vf": [256, 128, 64]},
        "activation_fn": nn.ELU,
        "log_std_init": -1.5,
    }


def _ppo_kwargs(*, seed: int, n_envs: int, n_steps: int, batch_size: int,
                n_epochs: int, device: str, tensorboard_log: str | None) -> dict[str, Any]:
    return {
        "learning_rate": 1.0e-4,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "target_kl": 0.015,
        "ent_coef": 0.0,
        "seed": int(seed),
        "device": device,
        "policy_kwargs": _policy_kwargs(),
        "verbose": 1,
        "tensorboard_log": tensorboard_log,
    }


class PurePPOAuditCallback(BaseCallback):
    """Record step/episode evidence and select best only with held-out eval."""

    def __init__(
        self,
        *,
        run_dir: Path,
        checkpoint_dir: Path,
        seed: int,
        updates: int,
        rollout_size: int,
        eval_interval: int,
        config_sha256: str,
        dry_run: bool = False,
    ):
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.updates = updates
        self.rollout_size = rollout_size
        self.eval_interval = eval_interval
        self.config_sha256 = config_sha256
        self.dry_run = dry_run
        self.metrics_path = run_dir / "metrics.csv"
        self.episodes_path = run_dir / "episodes.jsonl"
        self.validation_path = run_dir / "validation.jsonl"
        self.metrics_file = None
        self.metrics_writer = None
        self._last_logged_update = 0
        self._last_evaluated_update = 0
        self._best_rank: tuple[float, float, float] | None = None
        self._step_rewards: list[float] = []
        self._actions: list[np.ndarray] = []
        self._reward_components: dict[str, list[float]] = {}
        self._reward_raw: dict[str, list[float]] = {}
        self._episodes: list[dict[str, Any]] = []
        self._termination_reasons: Counter[str] = Counter()

    def _on_training_start(self) -> None:
        new_file = not self.metrics_path.exists()
        self.metrics_file = self.metrics_path.open("a", newline="", encoding="utf-8")
        self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=METRIC_FIELDS)
        if new_file:
            self.metrics_writer.writeheader()
            self.metrics_file.flush()

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._step_rewards.extend(float(x) for x in np.asarray(rewards).reshape(-1))
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        for index, info in enumerate(infos):
            action = info.get("action_applied")
            if action is not None:
                self._actions.append(np.asarray(action, dtype=np.float64))
            terms = info.get("reward_components") or {}
            for key, value in terms.items():
                self._reward_components.setdefault(key, []).append(float(value))
            raw_terms = info.get("reward_raw") or {}
            for key, value in raw_terms.items():
                self._reward_raw.setdefault(key, []).append(float(value))
            if dones is not None and index < len(dones) and bool(dones[index]):
                metrics = info.get("episode_metrics")
                if metrics:
                    self._episodes.append(metrics)
                    self._termination_reasons[str(metrics["termination_reason"])] += 1
                    with self.episodes_path.open("a", encoding="utf-8") as f:
                        f.write(_json({"seed": self.seed, **metrics}) + "\n")
        return True

    def _logger_value(self, key: str) -> float:
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return float("nan")
        return _scalar(logger.name_to_value.get(key))

    def _write_update(self, update: int) -> None:
        if update <= self._last_logged_update or not self.metrics_writer:
            return
        episodes = self._episodes
        actions = np.asarray(self._actions, dtype=np.float64)
        action_mean = actions.mean(axis=0).tolist() if actions.size else [float("nan")] * 6
        action_std = actions.std(axis=0).tolist() if actions.size else [float("nan")] * 6
        action_sat = (
            np.mean(np.abs(actions) >= 0.99, axis=0).tolist()
            if actions.size else [float("nan")] * 6
        )
        reward_means = {k: _safe_mean(v) for k, v in self._reward_components.items()}
        reward_raw_means = {k: _safe_mean(v) for k, v in self._reward_raw.items()}
        wheel_current_sat = [x for x in episodes for x in x.get("wheel_current_saturation_fraction", [])]
        wheel_torque_sat = [x for x in episodes for x in x.get("wheel_torque_saturation_fraction", [])]
        wheel_sat = [x for x in episodes for x in x.get("wheel_saturation_fraction", [])]
        leg_torque_sat = [x for x in episodes for x in x.get("leg_torque_saturation_fraction", [])]
        row = {
            "update": update,
            "timesteps": self.model.num_timesteps,
            "rollout_reward_mean": _safe_mean(self._step_rewards),
            "completed_episodes": len(episodes),
            "episode_return_mean": _safe_mean([float(x["episode_return"]) for x in episodes]),
            "episode_length_mean": _safe_mean([float(x["steps"]) for x in episodes]),
            "terminated_episodes": sum(bool(x["terminated"]) for x in episodes),
            "truncated_episodes": sum(bool(x["truncated"]) for x in episodes),
            "termination_reasons_json": _json(dict(self._termination_reasons)),
            "pitch_rms_rad_mean": _safe_mean([float(x["pitch_rms_rad"]) for x in episodes]),
            "pitch_peak_rad_max": max((float(x["pitch_peak_rad"]) for x in episodes), default=float("nan")),
            "pitch_rate_rms_rad_s_mean": _safe_mean([float(x["pitch_rate_rms_rad_s"]) for x in episodes]),
            "drift_max_m_max": max((float(x["drift_max_m"]) for x in episodes), default=float("nan")),
            "wheel_left_contact_fraction_mean": _safe_mean([float(x["wheel_left_contact_fraction"]) for x in episodes]),
            "wheel_right_contact_fraction_mean": _safe_mean([float(x["wheel_right_contact_fraction"]) for x in episodes]),
            "body_contact_steps_sum": sum(int(x["body_contact_steps"]) for x in episodes),
            "action_mean_json": _json(action_mean),
            "action_std_json": _json(action_std),
            "action_saturation_fraction_json": _json(action_sat),
            "wheel_current_saturation_fraction_json": _json(
                np.mean(wheel_current_sat, axis=0).tolist() if wheel_current_sat else [None] * 2
            ),
            "wheel_torque_saturation_fraction_json": _json(
                np.mean(wheel_torque_sat, axis=0).tolist() if wheel_torque_sat else [None] * 2
            ),
            "wheel_saturation_fraction_json": _json(
                np.mean(wheel_sat, axis=0).tolist() if wheel_sat else [None] * 2
            ),
            "leg_torque_saturation_fraction_json": _json(
                np.mean(leg_torque_sat, axis=0).tolist() if leg_torque_sat else [None] * 4
            ),
            "reward_components_mean_json": _json(reward_means),
            "reward_raw_mean_json": _json(reward_raw_means),
            "approx_kl": self._logger_value("train/approx_kl"),
            "clip_fraction": self._logger_value("train/clip_fraction"),
            "entropy_loss": self._logger_value("train/entropy_loss"),
            "value_loss": self._logger_value("train/value_loss"),
            "explained_variance": self._logger_value("train/explained_variance"),
            "policy_gradient_loss": self._logger_value("train/policy_gradient_loss"),
            "learning_rate": self._logger_value("train/learning_rate"),
            "seed": self.seed,
            "commit": current_commit(),
            "config_sha256": self.config_sha256,
        }
        self.metrics_writer.writerow(row)
        self.metrics_file.flush()
        self._last_logged_update = update
        self._step_rewards.clear()
        self._actions.clear()
        self._reward_components.clear()
        self._reward_raw.clear()
        self._episodes.clear()
        self._termination_reasons.clear()

    def _save_checkpoint(self, stem: str) -> Path:
        path = self.checkpoint_dir / stem
        self.model.save(str(path))
        contract = contract_payload(
            seed=self.seed,
            config_sha256=self.config_sha256,
            commit=current_commit(),
        )
        target = self.checkpoint_dir / f"{stem}_contract.json"
        target.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return self.checkpoint_dir / f"{stem}.zip"

    def _evaluate_and_select_best(self, update: int) -> None:
        env = PurePPOStandEnv(seed=VALIDATION_SEEDS[0])
        episodes = [
            evaluate_episode(self.model, seed=seed, mode="id", env=env)
            for seed in VALIDATION_SEEDS
        ]
        env.close()
        summary = summarize_evaluation(episodes, "id")
        with self.validation_path.open("a", encoding="utf-8") as f:
            f.write(_json({"update": update, "summary": summary}) + "\n")
        val_episode_path = self.run_dir / "validation_episodes.jsonl"
        with val_episode_path.open("a", encoding="utf-8") as f:
            for episode in episodes:
                f.write(_json({"update": update, **episode}) + "\n")
        rank = (
            float(summary["acceptance_rate"]),
            -float(summary["pitch_rms_deg_mean"]),
            -float(summary["planar_drift_m_mean"]),
        )
        if self._best_rank is None or rank > self._best_rank:
            self._best_rank = rank
            best_zip = self._save_checkpoint("best")
            best_record = {
                "update": update,
                "checkpoint": str(best_zip.relative_to(REPO_ROOT)),
                "selection": "fixed 50-episode deterministic validation only",
                "rank": list(rank),
                "summary": summary,
            }
            (self.checkpoint_dir / "best_eval.json").write_text(
                json.dumps(best_record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        self._last_evaluated_update = update
        print(
            f"[validation] update={update} episodes={summary['episodes']} "
            f"accepted={summary['accepted_episodes']} rate={summary['acceptance_rate']:.3f} "
            f"pitch_rms_mean={summary['pitch_rms_deg_mean']:.3f}deg",
            flush=True,
        )

    def _on_rollout_start(self) -> None:
        update = int(self.model.num_timesteps // self.rollout_size)
        if update <= 0:
            return
        self._write_update(update)
        if not self.dry_run:
            self._save_checkpoint("last")
            if update % self.eval_interval == 0 and update <= self.updates:
                self._evaluate_and_select_best(update)

    def _on_training_end(self) -> None:
        update = int(self.model.num_timesteps // self.rollout_size)
        self._write_update(update)
        if not self.dry_run:
            self._save_checkpoint("last")
            if self._last_evaluated_update != update:
                self._evaluate_and_select_best(update)
        if self.metrics_file is not None:
            self.metrics_file.flush()
            self.metrics_file.close()


def _training_config(seed: int, updates: int, n_envs: int, device: str) -> dict[str, Any]:
    import importlib.metadata as metadata

    cfg = pure_ppo_config()
    cfg["ppo"] = {
        "actor_critic_arch": {"pi": [256, 128, 64], "vf": [256, 128, 64]},
        "activation": "ELU",
        "optimizer": "Adam (fresh SB3 default state)",
        "learning_rate": 1.0e-4,
        "learning_rate_schedule": "constant",
        "n_envs": n_envs,
        "n_steps_per_env_per_update": ROLLOUT_STEPS,
        "batch_size": 512,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "target_kl": 0.015,
        "ent_coef": 0.0,
        "initial_log_std": -1.5,
        "max_updates": MAX_UPDATES,
        "requested_updates": updates,
        "seed": seed,
        "checkpoint_initialization": None,
        "initialization": "fresh random PPO weights and optimizer",
        "validation_seed_list": list(VALIDATION_SEEDS),
    }
    versions = {}
    for package in ("mujoco", "stable-baselines3", "torch", "numpy", "gymnasium"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    cfg["runtime"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device_requested": device,
        "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": versions,
    }
    cfg["model_xml_sha256"] = model_xml_sha256()
    cfg["source_fingerprint_sha256"] = source_fingerprint()
    cfg["commit"] = current_commit()
    cfg["working_tree_status"] = "recorded in baseline; user changes preserved"
    cfg["run_command"] = " ".join(sys.argv)
    return cfg


def _write_config(path: Path, config: dict[str, Any]) -> str:
    blob = json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(blob, encoding="utf-8")
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_dry_run(args: argparse.Namespace) -> int:
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = GATE_DIR / f"dry_run_{source_fingerprint()[:12]}"
    if output_dir.exists():
        raise FileExistsError(f"dry-run artifacts already exist; preserving them: {output_dir}")
    output_dir.mkdir(parents=True)
    envs = 2
    n_steps = 64
    batch_size = 64
    run_dir = output_dir / "runs"
    ckpt_dir = output_dir / "checkpoints"
    run_dir.mkdir()
    ckpt_dir.mkdir()
    config = _training_config(seed=args.seed, updates=2, n_envs=envs, device=args.device)
    config["dry_run"] = True
    config["ppo"].update({"n_steps_per_env_per_update": n_steps, "batch_size": batch_size, "n_epochs": 2})
    config_sha = _write_config(run_dir / "config.json", config)
    vec_env = _make_vec_env(envs, args.seed, subproc=False)
    model = PPO(
        AsymmetricActorCriticPolicy,
        vec_env,
        **_ppo_kwargs(
            seed=args.seed,
            n_envs=envs,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=2,
            device=args.device,
            tensorboard_log=None,
        ),
    )
    callback = PurePPOAuditCallback(
        run_dir=run_dir,
        checkpoint_dir=ckpt_dir,
        seed=args.seed,
        updates=2,
        rollout_size=envs * n_steps,
        eval_interval=99,
        config_sha256=config_sha,
        dry_run=True,
    )
    model.learn(total_timesteps=2 * envs * n_steps, callback=callback, log_interval=1)
    vec_env.close()
    log = model.logger.name_to_value
    finite_keys = (
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/value_loss",
        "train/explained_variance",
        "train/policy_gradient_loss",
    )
    ppo_metrics = {key: _scalar(log.get(key)) for key in finite_keys}
    status = "PASS" if all(np.isfinite(value) for value in ppo_metrics.values()) else "FAIL"
    model.save(str(ckpt_dir / "dry_run"))
    report = {
        "status": status,
        "updates": 2,
        "timesteps": 2 * envs * n_steps,
        "n_envs": envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "ppo_metrics": ppo_metrics,
        "model_xml_sha256": model_xml_sha256(),
        "source_fingerprint_sha256": source_fingerprint(),
        "artifact": str((ckpt_dir / "dry_run.zip").relative_to(REPO_ROOT)),
        "device": str(model.device),
    }
    (GATE_DIR / "dry_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 2


def run_formal_training(args: argparse.Namespace) -> int:
    gates = _check_training_gates()
    if args.seed not in (0, 1, 2):
        raise ValueError("the first experiment accepts training seeds 0, 1, or 2 only")
    if not (1 <= args.updates <= MAX_UPDATES):
        raise ValueError(f"updates must be within [1, {MAX_UPDATES}]")
    if not (1 <= args.num_envs <= 8):
        raise ValueError("num_envs must be within [1, 8]; record reductions for comparison")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")

    run_dir = RUN_ROOT / f"seed{args.seed}"
    checkpoint_dir = CHECKPOINT_ROOT / f"seed{args.seed}"
    for path in (run_dir, checkpoint_dir):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing experiment data: {path}")
        path.mkdir(parents=True, exist_ok=False)
    config = _training_config(args.seed, args.updates, args.num_envs, args.device)
    config["training_gates"] = gates
    config_sha = _write_config(run_dir / "config.json", config)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "launch.json").write_text(
        json.dumps({"argv": sys.argv, "seed": args.seed}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    vec_env = _make_vec_env(args.num_envs, args.seed, subproc=not args.dummy_vec_env)
    vec_env.seed(args.seed)
    model = PPO(
        AsymmetricActorCriticPolicy,
        vec_env,
        **_ppo_kwargs(
            seed=args.seed,
            n_envs=args.num_envs,
            n_steps=ROLLOUT_STEPS,
            batch_size=512,
            n_epochs=5,
            device=args.device,
            tensorboard_log=str(run_dir / "tensorboard"),
        ),
    )
    callback = PurePPOAuditCallback(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        updates=args.updates,
        rollout_size=args.num_envs * ROLLOUT_STEPS,
        eval_interval=args.eval_interval,
        config_sha256=config_sha,
    )
    total_timesteps = args.updates * args.num_envs * ROLLOUT_STEPS
    print(
        f"[train] seed={args.seed} fresh PPO initialization; updates={args.updates}; "
        f"n_envs={args.num_envs}; timesteps={total_timesteps}; device={model.device}",
        flush=True,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=1,
        reset_num_timesteps=True,
        progress_bar=False,
    )
    vec_env.close()
    print(
        f"[complete] seed={args.seed} updates={model.num_timesteps // (args.num_envs * ROLLOUT_STEPS)} "
        f"timesteps={model.num_timesteps} run={run_dir} checkpoint={checkpoint_dir}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dummy-vec-env", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="run the two-update PPO gate only")
    args = parser.parse_args()
    return run_dry_run(args) if args.dry_run else run_formal_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
