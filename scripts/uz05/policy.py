"""SB3 policy modules for asymmetric actor/critic observation inputs."""

from __future__ import annotations

import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class AsymmetricMlpExtractor(nn.Module):
    """The actor reads its observation prefix; the critic reads the full vector."""

    def __init__(self, actor_dim, observation_dim, net_arch, activation_fn):
        super().__init__()
        self.actor_dim = int(actor_dim)
        if self.actor_dim <= 0 or self.actor_dim > int(observation_dim):
            raise ValueError(
                f"actor_dim must be in [1, observation_dim], got "
                f"{self.actor_dim} / {observation_dim}"
            )
        pi_layers = list(net_arch.get("pi", [64, 64]))
        vf_layers = list(net_arch.get("vf", [64, 64]))
        self.policy_net = self._build(self.actor_dim, pi_layers, activation_fn)
        self.value_net = self._build(observation_dim, vf_layers, activation_fn)
        self.latent_dim_pi = pi_layers[-1]
        self.latent_dim_vf = vf_layers[-1]

    @staticmethod
    def _build(in_dim, layers, activation_fn):
        modules, last = [], int(in_dim)
        for hidden in layers:
            modules += [nn.Linear(last, hidden), activation_fn()]
            last = hidden
        return nn.Sequential(*modules)

    def forward(self, obs):
        return self.policy_net(obs[..., : self.actor_dim]), self.value_net(obs)

    def forward_actor(self, obs):
        return self.policy_net(obs[..., : self.actor_dim])

    def forward_critic(self, obs):
        return self.value_net(obs)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """SB3 PPO policy with privileged observations available only to the critic."""

    def __init__(self, *args, actor_dim=None, **kwargs):
        if actor_dim is None:
            raise ValueError("actor_dim is required for the asymmetric policy")
        self._actor_dim = int(actor_dim)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = AsymmetricMlpExtractor(
            self._actor_dim, self.features_dim, self.net_arch, self.activation_fn
        )
