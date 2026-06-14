"""Conditional Flow Matching model for one-step Heston transitions."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar FM time `tau` in [0, 1]."""

    def __init__(self, embedding_dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.max_period = float(max_period)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 0:
            tau = tau[None]
        tau = tau.reshape(-1).float()
        half = self.embedding_dim // 2
        if half == 0:
            return tau[:, None]

        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=tau.device, dtype=tau.dtype)
            / max(half - 1, 1)
        )
        args = tau[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FiLMResidualBlock(nn.Module):
    """Residual MLP block modulated by condition and time embeddings."""

    def __init__(self, hidden_dim: int, context_dim: int, expansion: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(context_dim, hidden_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.SiLU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(context).chunk(2, dim=-1)
        modulated = self.norm(hidden) * (1.0 + scale) + shift
        return hidden + self.net(modulated)


class TransitionFM(nn.Module):
    """Conditional FM vector field for normalized one-step market states.

    Inputs:
        x_tau: noised next state, shape `[B, state_dim]`
        tau: FM interpolation time, shape `[B]` or `[B, 1]`
        condition: current state condition, shape `[B, condition_dim]`

    The default dimensions correspond to:
        condition = `(normalized log_v_t, normalized r_t)`
        target = `(normalized log_v_next, normalized r_next)`
    """

    def __init__(
        self,
        state_dim: int = 2,
        condition_dim: int = 2,
        hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or condition_dim <= 0:
            raise ValueError("state_dim and condition_dim must be positive")
        if hidden_dim <= 0 or time_embedding_dim <= 0:
            raise ValueError("hidden_dim and time_embedding_dim must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.state_dim = state_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.context_net = nn.Sequential(
            nn.Linear(condition_dim + time_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(hidden_dim=hidden_dim, context_dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x_tau: torch.Tensor, tau: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if x_tau.ndim != 2:
            raise ValueError("x_tau must have shape [batch, state_dim]")
        if condition.ndim != 2:
            raise ValueError("condition must have shape [batch, condition_dim]")
        if x_tau.shape[0] != condition.shape[0]:
            raise ValueError("x_tau and condition batch sizes must match")
        if x_tau.shape[1] != self.state_dim:
            raise ValueError(f"x_tau must have state_dim={self.state_dim}")
        if condition.shape[1] != self.condition_dim:
            raise ValueError(f"condition must have condition_dim={self.condition_dim}")

        tau_embedding = self.time_embedding(tau.to(device=x_tau.device))
        if tau_embedding.shape[0] != x_tau.shape[0]:
            raise ValueError("tau batch size must match x_tau batch size")

        context = self.context_net(torch.cat([condition, tau_embedding.to(dtype=condition.dtype)], dim=-1))
        hidden = self.input_proj(x_tau)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output(hidden)


def sample_conditional_flow_batch(
    target: torch.Tensor,
    time_eps: float = 1e-4,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the straight-line CFM training tuple for a target batch."""

    if target.ndim < 2:
        raise ValueError("target must have shape [batch, ...]")
    if not 0.0 <= time_eps < 0.5:
        raise ValueError("time_eps must be in [0, 0.5)")

    batch_size = target.shape[0]
    if noise is None:
        noise = torch.randn_like(target)
    elif noise.shape != target.shape:
        raise ValueError("noise must match target shape")

    tau = torch.rand(batch_size, device=target.device, dtype=target.dtype)
    if time_eps > 0:
        tau = tau * (1.0 - 2.0 * time_eps) + time_eps
    tau_view = tau.reshape(batch_size, *([1] * (target.ndim - 1)))
    x_tau = (1.0 - tau_view) * noise + tau_view * target
    velocity = target - noise
    return x_tau, tau, velocity, noise


def conditional_flow_matching_loss(
    model: TransitionFM,
    condition: torch.Tensor,
    target: torch.Tensor,
    time_eps: float = 1e-4,
    target_weights: torch.Tensor | None = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Compute conditional FM MSE loss for normalized transitions."""

    x_tau, tau, velocity, _ = sample_conditional_flow_batch(target=target, time_eps=time_eps)
    prediction = model(x_tau=x_tau, tau=tau, condition=condition)
    per_dim_loss = (prediction - velocity).pow(2)
    if target_weights is not None:
        if target_weights.ndim != 1 or target_weights.shape[0] != target.shape[-1]:
            raise ValueError("target_weights must have shape [target_dim]")
        weights = target_weights.to(device=target.device, dtype=target.dtype)
        if torch.any(weights <= 0):
            raise ValueError("target_weights must be positive")
        weights = weights / weights.mean().clamp_min(1e-12)
        per_dim_loss = per_dim_loss * weights.view(*([1] * (per_dim_loss.ndim - 1)), -1)
    if reduction == "mean":
        return per_dim_loss.mean()
    if reduction == "none":
        return per_dim_loss.flatten(start_dim=1).mean(dim=1)
    raise ValueError("reduction must be 'mean' or 'none'")


@torch.no_grad()
def euler_sample(
    model: TransitionFM,
    condition: torch.Tensor,
    n_steps: int = 20,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample a next-state target by Euler-integrating the learned vector field."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if condition.ndim != 2:
        raise ValueError("condition must have shape [batch, condition_dim]")
    if noise is None:
        x = torch.randn(condition.shape[0], model.state_dim, device=condition.device, dtype=condition.dtype)
    else:
        if noise.shape != (condition.shape[0], model.state_dim):
            raise ValueError("noise shape must be [batch, model.state_dim]")
        x = noise.to(device=condition.device, dtype=condition.dtype)

    dt = 1.0 / n_steps
    for step in range(n_steps):
        tau = torch.full((condition.shape[0],), step * dt, device=condition.device, dtype=condition.dtype)
        x = x + dt * model(x_tau=x, tau=tau, condition=condition)
    return x

