"""
Custom actor distribution + policy for the neutral (bias-free) A2C agent in the
BBAPT reproduction (Charkhestani & Esfahanipour, 2026).

Design, per Section 3.2 / Figure 6 of the paper:
  - The actor outputs a diagonal Gaussian over RAW, pre-softmax scores (one per
    asset), NOT over portfolio weights directly.
  - Mean branch:  Linear -> ReLU -> clamp to [0, 1]   (bounded, non-negative mean)
  - Std branch:   Linear -> Softplus                   (state-dependent, > 0)
  - softmax(raw_action) -> portfolio weights is applied OUTSIDE this policy,
    inside the environment's step(), which is the only mathematically valid
    place for it (see conversation notes: softmax is non-invertible, so a
    closed-form log_prob only exists for the pre-softmax Gaussian sample, which
    is exactly what SB3's rollout collection stores and trains on).

Assumptions made explicit (flag if you want these changed):
  - "Scaling layer" in Fig. 6 is interpreted literally as ReLU followed by a
    hard clamp to an upper bound of 1.0. This is non-differentiable at the
    clamp boundary (zero gradient once saturated) -- kept as-is per instruction
    to stick with ReLU for now; a Sigmoid replacement is a one-line swap later
    (see `ClampedReLU` vs. a future `SigmoidBound`).
  - Softplus std has a small additive epsilon (STD_EPSILON) to keep sigma
    strictly positive for numerical stability in log_prob / entropy.
  - action_space must be gymnasium.spaces.Box(low=0.0, high=1.0, shape=(n,)).
    SB3 clips the sampled (unclipped) action to these bounds before it reaches
    env.step(); the UNCLIPPED sample is what's stored in the rollout buffer and
    used for log_prob/entropy in the A2C loss (confirmed against SB3 2.9.0
    source, common/on_policy_algorithm.py::collect_rollouts).

Usage:
    from portfolio_a2c_policy import PortfolioActorCriticPolicy
    from stable_baselines3 import A2C

    model = A2C(
        PortfolioActorCriticPolicy,
        env,
        policy_kwargs=dict(
            net_arch=dict(pi=[320, 160, 80], vf=[64, 16, 4]),  # paper's Table 5, optional
            activation_fn=nn.Tanh,                              # paper's shared-trunk activation
        ),
    )
    model.learn(total_timesteps=...)

    raw_action, _ = model.predict(obs, deterministic=False)
    weights = softmax(raw_action)   # do this in env.step(), and again here if
                                     # you want interpretable weights at inference
"""

from functools import partial

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Normal

from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import Schedule

STD_EPSILON = 1e-4  # floor added to softplus(std) so sigma never hits exactly 0


class ClampedReLU(nn.Module):
    """ReLU followed by a hard clamp at `max_value`. Bounds output to [0, max_value].

    Non-differentiable at the upper boundary (gradient is 0 once clamped) -- this
    is the literal reading of the paper's "ReLU + scaling layer" for the mean
    branch. Swap for nn.Sigmoid() * max_value if/when you want smooth gradients
    across the whole range instead.
    """

    def __init__(self, max_value: float = 1.0):
        super().__init__()
        self.max_value = max_value

    def forward(self, x: th.Tensor) -> th.Tensor:
        return th.clamp(th.relu(x), max=self.max_value)


class PortfolioGaussianDistribution(DiagGaussianDistribution):
    """
    Diagonal Gaussian whose mean and std are BOTH direct, independent network
    outputs (mean via ReLU+clamp, std via Softplus), instead of SB3's default
    (mean via Linear, std via a single free `log_std` parameter that gets
    exponentiated). log_prob / entropy / sample / mode are inherited unchanged
    from DiagGaussianDistribution -- once `self.distribution` is a valid
    `torch.distributions.Normal`, those closed-form formulas need no changes.
    """

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0):
        """
        Returns (mean_net, std_net) -- both nn.Module, both functions of the
        actor's latent features. `log_std_init` is accepted for interface
        compatibility with the base class but unused here (std is fully
        state-dependent via Softplus, there is no fixed initial value to set).
        """
        mean_net = nn.Sequential(
            nn.Linear(latent_dim, self.action_dim),
            ClampedReLU(max_value=1.0),
        )
        std_net = nn.Sequential(
            nn.Linear(latent_dim, self.action_dim),
            nn.Softplus(),
        )
        return mean_net, std_net

    def proba_distribution(self, mean_actions: th.Tensor, std: th.Tensor) -> "PortfolioGaussianDistribution":
        """
        `std` here is already the actual standard deviation (softplus output),
        NOT a log-std to be exponentiated -- this is the one line that removes
        SB3's default exp(log_std) convention.
        """
        self.distribution = Normal(mean_actions, std + STD_EPSILON)
        return self


class PortfolioActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy variant that forces PortfolioGaussianDistribution instead
    of SB3's default DiagGaussianDistribution, and wires the std branch as a
    genuine function of the latent features (state-dependent), which the base
    class's `_get_action_dist_from_latent` does not support out of the box
    (it passes `self.log_std` through as-is rather than calling it on
    latent_pi). Only `_build` and `_get_action_dist_from_latent` are
    overridden; forward / evaluate_actions / predict / get_distribution are
    all inherited unchanged since they route through these two methods.

    Shared trunk (mlp_extractor) and value_net are untouched -- configure them
    the normal SB3 way via `net_arch` / `activation_fn` in policy_kwargs.
    """

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Overwrite whatever make_proba_distribution() picked in __init__
        # (it always returns a plain DiagGaussianDistribution for Box spaces).
        self.action_dist = PortfolioGaussianDistribution(get_action_dim(self.action_space))
        self.action_net, self.std_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.std_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        mean_actions = self.action_net(latent_pi)
        std = self.std_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, std)