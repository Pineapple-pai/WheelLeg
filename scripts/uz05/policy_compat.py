"""Migrate legacy PPO wheel actions to the body-forward-positive convention."""

from __future__ import annotations

import torch
from torch import nn

from .spec import ACTOR_OBS_DIM, OBS_HISTORY, OBS_SLICES, WHEEL_ACTION_FRAME


def ensure_forward_positive_wheel_actions(model) -> bool:
    """Change old joint-positive actions without changing the physical policy.

    Legacy checkpoints emitted native joint-positive wheel targets.  The new
    actuator boundary negates the first two actions.  Negating those output
    rows and the previous-wheel-action input columns preserves both the
    actor's physical output and the critic's value on equivalent histories.
    Returns True only when an old checkpoint was migrated.
    """
    frame = getattr(model, "wheel_action_frame", None)
    if frame == WHEEL_ACTION_FRAME:
        return False
    if frame not in (None, "joint_positive"):
        raise ValueError(f"unsupported wheel action frame: {frame!r}")
    if OBS_HISTORY != 1:
        raise ValueError("wheel-action checkpoint migration requires OBS_HISTORY=1")

    previous_wheels = OBS_SLICES["previous_action"].start
    actor = model.policy.mlp_extractor.policy_net[0]
    critic = model.policy.mlp_extractor.value_net[0]
    output = model.policy.action_net
    if not all(isinstance(layer, nn.Linear) for layer in (actor, critic, output)):
        raise TypeError("unsupported PPO architecture for wheel-action migration")
    if (actor.in_features != ACTOR_OBS_DIM
            or critic.in_features < previous_wheels + 2
            or output.out_features != 6):
        raise ValueError("checkpoint dimensions do not match wheel-action migration")

    with torch.no_grad():
        actor.weight[:, previous_wheels:previous_wheels + 2].mul_(-1)
        critic.weight[:, previous_wheels:previous_wheels + 2].mul_(-1)
        output.weight[:2].mul_(-1)
        output.bias[:2].mul_(-1)

    model.wheel_action_frame = WHEEL_ACTION_FRAME
    # Adam moments belong to the old parameter coordinates.  Rebuild them on
    # the next update while retaining all learned policy and value weights.
    model.policy.optimizer.state.clear()
    return True
