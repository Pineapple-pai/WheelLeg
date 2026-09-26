"""Minimal ONNX Runtime actor used by the 125 Hz deployment loop."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .spec import ACTION_DIM, ACTOR_OBS_DIM, WHEEL_ACTION_FRAME


class OnnxPolicy:
    def __init__(self, model_path: str | Path):
        contract_path = Path(model_path).with_suffix(".contract.json")
        if not contract_path.is_file():
            raise ValueError(f"ONNX action convention is unknown: missing {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("wheel_action_frame") != WHEEL_ACTION_FRAME:
            raise ValueError("ONNX wheel actions use an incompatible direction convention")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is missing; install requirements-deploy.txt"
            ) from exc
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        if self.session.get_inputs()[0].name != "actor_observation":
            raise ValueError("unexpected ONNX input name")

    def predict(self, actor_observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(actor_observation, dtype=np.float32).reshape(1, -1)
        if obs.shape[1] != ACTOR_OBS_DIM:
            raise ValueError(
                f"expected {ACTOR_OBS_DIM} actor observations, got {obs.shape[1]}"
            )
        action = self.session.run(["action"], {"actor_observation": obs})[0][0]
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise RuntimeError("invalid ONNX action")
        return np.clip(action, -1.0, 1.0)
