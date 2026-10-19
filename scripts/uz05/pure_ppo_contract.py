"""Versioned file/weight contract for the isolated Pure PPO stand experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .pure_ppo_stand import (
    ACTION_ORDER,
    ACTION_PROTOCOL,
    LEG_POSITION_SCALE_RAD,
    MOTOR_RATE_HZ,
    POLICY_RATE_HZ,
    Q_REF,
    WHEEL_SIGN,
    WHEEL_SPEED_SCALE_RAD_S,
)
from .spec import (
    ACTOR_OBS_BLOCKS,
    OBS_BLOCKS,
    OBS_DIM,
    OBS_SCALE,
    OBS_SOURCE,
    PRIV_OBS_DIM,
    ACTOR_OBS_DIM,
    MODEL_XML,
    EnvParams,
)


CONTRACT_VERSION = "pure_ppo_stand_v1"


def model_xml_sha256() -> str:
    return hashlib.sha256(MODEL_XML.read_bytes()).hexdigest()


def source_fingerprint() -> str:
    """Hash the model/control path and Pure PPO entry points used by the gates."""
    root = MODEL_XML.parents[0]
    paths = [
        root / "scripts/uz05/model.py",
        root / "scripts/uz05/spec.py",
        root / "scripts/uz05/actuators.py",
        root / "scripts/uz05/observations.py",
        root / "scripts/uz05/policy.py",
        root / "scripts/uz05/env.py",
        root / "scripts/uz05/pure_ppo_stand.py",
        root / "scripts/uz05/pure_ppo_contract.py",
        root / "scripts/uz05/pure_ppo_eval.py",
        root / "scripts/train_pure_ppo_stand.py",
        root / "scripts/preflight_pure_ppo_stand.py",
        root / "scripts/smoke_pure_ppo_stand.py",
        root / "scripts/eval_pure_ppo_stand.py",
        root / "scripts/export_pure_ppo_stand_onnx.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def observation_layout_hash() -> str:
    layout = {
        "actor_blocks": list(ACTOR_OBS_BLOCKS),
        "critic_blocks": list(OBS_BLOCKS[len(ACTOR_OBS_BLOCKS) :]),
        "scales": OBS_SCALE,
        "sources": OBS_SOURCE,
    }
    blob = json.dumps(layout, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def contract_payload(
    *, seed: int, config_sha256: str, commit: str | None = None
) -> dict[str, Any]:
    params = EnvParams()
    actuator_config = {
        "joint": asdict(params.joint),
        "wheel": asdict(params.wheel),
        "control_dt_s": 0.008,
        "motor_dt_s": 0.002,
        "motor_ticks_per_policy_step": 4,
    }
    return {
        "contract_version": CONTRACT_VERSION,
        "task_id": CONTRACT_VERSION,
        "action_protocol": ACTION_PROTOCOL,
        "action_order": list(ACTION_ORDER),
        "action_shape": 6,
        "q_ref_rad": Q_REF.tolist(),
        "wheel_sign": WHEEL_SIGN.tolist(),
        "wheel_speed_scale_rad_s": WHEEL_SPEED_SCALE_RAD_S,
        "leg_position_scale_rad": LEG_POSITION_SCALE_RAD,
        "actor_observation_shape": ACTOR_OBS_DIM,
        "critic_observation_shape": PRIV_OBS_DIM,
        "observation_shape": OBS_DIM,
        "actor_observation_layout_sha256": observation_layout_hash(),
        "model_xml_sha256": model_xml_sha256(),
        "source_fingerprint_sha256": source_fingerprint(),
        "actuator_config": actuator_config,
        "policy_rate_hz": POLICY_RATE_HZ,
        "motor_rate_hz": MOTOR_RATE_HZ,
        "domain_randomization": False,
        "observation_noise": False,
        "seed": int(seed),
        "initialization": "fresh random SB3 PPO; no checkpoint loaded",
        "config_sha256": config_sha256,
        "commit": commit or current_commit(),
    }


def require_compatible_contract(contract: dict[str, Any]) -> None:
    """Reject legacy/missing contracts and any mismatch with current inference."""
    expected = contract_payload(
        seed=int(contract.get("seed", -1)),
        config_sha256=str(contract.get("config_sha256", "")),
        commit=str(contract.get("commit", "unknown")),
    )
    mismatches = {
        key: (contract.get(key), expected_value)
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            "checkpoint contract is missing, legacy, or incompatible with the "
            f"current Pure PPO stand interface: {mismatches}"
        )


def read_and_require_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    if not contract_path.is_file():
        raise FileNotFoundError(f"required Pure PPO contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    require_compatible_contract(contract)
    return contract
