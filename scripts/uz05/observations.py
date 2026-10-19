"""Named actor/critic observation layout and its checked vector encoder."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .spec import (
    ACTOR_OBS_BLOCKS,
    ACTOR_OBS_DIM,
    OBS_BLOCKS,
    OBS_DIM,
    OBS_SCALE,
    PRIV_OBS_DIM,
    PRIV_OBS_BLOCKS,
    observation_noise_vector,
)


class ObservationEncoder:
    """Scale, perturb, pack, and validate named observation blocks.

    The actor and critic share a flat PPO space, so the layout is explicit:
    one current actor frame followed by one clean privileged frame. The actor's
    sensor-like signals receive noise; the critic's privileged signals remain
    noise free. No observation history is stacked.
    """

    def __init__(self, noise, clip_observations: float):
        self.noise = observation_noise_vector(noise)
        self.clip_observations = float(clip_observations)
        self.block_widths = dict(OBS_BLOCKS)
        if set(self.noise) != set(self.block_widths):
            raise ValueError("observation noise keys must match the declared observation blocks")
        self._validate_layout()

    @staticmethod
    def _validate_layout() -> None:
        names = [name for name, _ in OBS_BLOCKS]
        if len(names) != len(set(names)):
            raise ValueError("observation block names must be unique across actor and critic")
        if set(names) != set(OBS_SCALE):
            raise ValueError("observation scales must match the declared observation blocks")
        actor_dim = sum(width for _, width in ACTOR_OBS_BLOCKS)
        privileged_dim = sum(width for _, width in PRIV_OBS_BLOCKS)
        if actor_dim != ACTOR_OBS_DIM:
            raise ValueError(
                f"actor blocks total {actor_dim}, but ACTOR_OBS_DIM is {ACTOR_OBS_DIM}"
            )
        if privileged_dim != PRIV_OBS_DIM:
            raise ValueError(
                f"privileged blocks total {privileged_dim}, but PRIV_OBS_DIM is {PRIV_OBS_DIM}"
            )
        if actor_dim + privileged_dim != OBS_DIM:
            raise ValueError("OBS_DIM does not match the declared actor/critic layout")

    def encode(
        self, values: Mapping[str, np.ndarray], rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        expected = set(self.block_widths)
        actual = set(values)
        if actual != expected:
            raise ValueError(
                f"observation block mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )

        raw: dict[str, np.ndarray] = {}
        scaled: dict[str, np.ndarray] = {}
        for name, width in OBS_BLOCKS:
            block = np.asarray(values[name], dtype=np.float64)
            if block.shape != (width,):
                raise ValueError(
                    f"observation block {name!r} has shape {block.shape}; expected ({width},)"
                )
            if not np.all(np.isfinite(block)):
                raise ValueError(f"observation block {name!r} contains NaN or infinity")
            raw[name] = block.copy()
            scaled[name] = block * np.asarray(OBS_SCALE.get(name, 1.0), dtype=np.float64)

        actor_frame = np.concatenate([scaled[name] for name, _ in ACTOR_OBS_BLOCKS])
        if actor_frame.shape != (ACTOR_OBS_DIM,):
            raise RuntimeError(
                f"actor frame has shape {actor_frame.shape}; expected ({ACTOR_OBS_DIM},)"
            )
        if self.clip_observations > 0.0:
            np.clip(actor_frame, -self.clip_observations, self.clip_observations, out=actor_frame)

        noise_widths = np.concatenate([
            np.full(width, float(self.noise.get(name, 0.0)), dtype=np.float64)
            for name, width in ACTOR_OBS_BLOCKS
        ])
        if np.any(noise_widths):
            actor_frame += rng.uniform(-1.0, 1.0, actor_frame.shape) * noise_widths

        privileged = np.concatenate([scaled[name] for name, _ in PRIV_OBS_BLOCKS])
        if privileged.shape != (PRIV_OBS_DIM,):
            raise RuntimeError(
                f"privileged frame has shape {privileged.shape}; expected ({PRIV_OBS_DIM},)"
            )
        observation = np.concatenate((actor_frame, privileged)).astype(np.float32)

        expected_shape = (OBS_DIM,)
        if observation.shape != expected_shape:
            raise RuntimeError(
                f"encoded observation has shape {observation.shape}; expected {expected_shape}"
            )
        return observation, raw, scaled
