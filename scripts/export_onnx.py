#!/usr/bin/env python3
"""Export the deterministic 38-D PPO actor to ONNX and verify parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from train_uz05 import AsymmetricActorCriticPolicy
from uz05.policy_compat import ensure_forward_positive_wheel_actions
from uz05.spec import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    MOTOR_CONTROL_RATE_HZ,
    MOTOR_TICKS_PER_POLICY,
    OBS_DIM,
    POLICY_RATE_HZ,
    WHEEL_ACTION_FRAME,
    WHEEL_ANGULAR_TO_BODY_X,
)


class DeterministicActor(torch.nn.Module):
    """Deployment-only actor: 38 sensor values in, six clipped targets out."""

    def __init__(self, policy: AsymmetricActorCriticPolicy):
        super().__init__()
        self.actor = policy.mlp_extractor.policy_net
        self.action_head = policy.action_net

    def forward(self, actor_observation: torch.Tensor) -> torch.Tensor:
        latent = self.actor(actor_observation)
        return torch.clamp(self.action_head(latent), -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--parity-samples", type=int, default=64)
    args = parser.parse_args()

    try:
        import onnx
    except ImportError as exc:
        raise SystemExit(
            "ONNX export dependency is missing; install requirements-deploy.txt"
        ) from exc

    model = PPO.load(
        args.checkpoint,
        custom_objects={"policy_class": AsymmetricActorCriticPolicy},
        device="cpu",
    )
    migrated_legacy_wheel_actions = ensure_forward_positive_wheel_actions(model)
    actor = DeterministicActor(model.policy).eval()
    example = torch.zeros(1, ACTOR_OBS_DIM, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        example,
        str(args.output),
        input_names=["actor_observation"],
        output_names=["action"],
        dynamic_axes={"actor_observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(args.output))
    onnx.checker.check_model(onnx_model)

    parity_max_abs = None
    try:
        import onnxruntime as ort

        rng = np.random.default_rng(0)
        actor_obs = rng.normal(0.0, 0.5, (args.parity_samples, ACTOR_OBS_DIM)).astype(np.float32)
        full_obs = np.zeros((args.parity_samples, OBS_DIM), dtype=np.float32)
        full_obs[:, :ACTOR_OBS_DIM] = actor_obs
        sb3_action, _ = model.predict(full_obs, deterministic=True)
        session = ort.InferenceSession(
            str(args.output), providers=["CPUExecutionProvider"]
        )
        onnx_action = session.run(["action"], {"actor_observation": actor_obs})[0]
        parity_max_abs = float(np.max(np.abs(sb3_action - onnx_action)))
        if parity_max_abs > 1e-5:
            raise RuntimeError(f"ONNX parity failed: max abs error {parity_max_abs:.3e}")
    except ImportError:
        pass

    contract = {
        "contract_version": "uz05_direct_ppo_mit_v4_forward_positive_wheels",
        "source_checkpoint": str(Path(args.checkpoint)),
        "legacy_wheel_action_migrated": migrated_legacy_wheel_actions,
        "wheel_action_frame": WHEEL_ACTION_FRAME,
        "wheel_action_to_joint_velocity_sign": WHEEL_ANGULAR_TO_BODY_X,
        "runtime": "onnxruntime_cpu",
        "input_name": "actor_observation",
        "input_shape": ["batch", ACTOR_OBS_DIM],
        "output_name": "action",
        "output_shape": ["batch", ACTION_DIM],
        "policy_rate_hz": POLICY_RATE_HZ,
        "motor_control_rate_hz": MOTOR_CONTROL_RATE_HZ,
        "motor_ticks_per_policy": MOTOR_TICKS_PER_POLICY,
        "deterministic": True,
        "deployment_accepted": False,
        "deployment_acceptance_note": (
            "Export parity only; pass multirate simulation and hardware bring-up separately"
        ),
        "output_clipped": [-1.0, 1.0],
        "parity_max_abs": parity_max_abs,
    }
    sidecar = args.output.with_suffix(".contract.json")
    sidecar.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"onnx: {args.output.resolve()}")
    print(f"contract: {sidecar.resolve()}")
    print(f"input: actor_observation float32 [batch, {ACTOR_OBS_DIM}]")
    print(f"output: action float32 [batch, {ACTION_DIM}]")
    print(f"parity_max_abs: {parity_max_abs}")


if __name__ == "__main__":
    main()
