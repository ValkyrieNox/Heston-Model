"""Sampler abstractions over trained vol/ret models.

Three concrete samplers share the same ``sample(condition, *, noise=None) -> Tensor``
interface so the rollout module can plug any of them in:

- ``FMTeacherSampler``: multi-step Euler integration of an FM teacher.
- ``MeanFlowSampler``:  single-NFE call to a Mean Flow student.
- ``ConsistencySampler``: single-NFE call to a Consistency student.

A factory ``load_sampler_from_checkpoint`` inspects the saved checkpoint's
``stage`` / ``extra.kind`` fields and instantiates the right model + sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from finflow.models import ConsistencyStudent, MeanFlowStudent, TransitionFM
from finflow.training import load_checkpoint, resolve_device


class Sampler:
    """One-step conditional sampler interface."""

    state_dim: int
    condition_dim: int
    kind: str  # "fm" | "mf" | "cd"
    device: torch.device

    def sample(self, condition: torch.Tensor, *, noise: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


def _ensure_noise(noise, batch, state_dim, device, dtype) -> torch.Tensor:
    if noise is None:
        return torch.randn(batch, state_dim, device=device, dtype=dtype)
    if noise.shape != (batch, state_dim):
        raise ValueError(f"noise shape must be ({batch}, {state_dim}), got {tuple(noise.shape)}")
    return noise.to(device=device, dtype=dtype)


class FMTeacherSampler(Sampler):
    """Multi-step Euler ODE sampler over a trained Flow Matching teacher."""

    kind = "fm"

    def __init__(self, model: TransitionFM, n_steps: int = 20) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.model = model
        self.n_steps = n_steps
        self.state_dim = model.state_dim
        self.condition_dim = model.condition_dim
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, *, noise: torch.Tensor | None = None) -> torch.Tensor:
        condition = condition.to(self.device)
        batch = condition.shape[0]
        dtype = condition.dtype
        x = _ensure_noise(noise, batch, self.state_dim, self.device, dtype)
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            tau = torch.full((batch,), step * dt, device=self.device, dtype=dtype)
            x = x + dt * self.model(x_tau=x, tau=tau, condition=condition)
        return x


class MeanFlowSampler(Sampler):
    """1-NFE sampler that evaluates ``u(z, 0, 1, c)`` once.

    Mean Flow is distilled in the reversed ``data -> noise`` time convention,
    so the generated data endpoint is ``z - u(z, 0, 1, c)``.
    """

    kind = "mf"

    def __init__(self, model: MeanFlowStudent) -> None:
        self.model = model
        self.state_dim = model.state_dim
        self.condition_dim = model.condition_dim
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, *, noise: torch.Tensor | None = None) -> torch.Tensor:
        condition = condition.to(self.device)
        batch = condition.shape[0]
        dtype = condition.dtype
        x = _ensure_noise(noise, batch, self.state_dim, self.device, dtype)
        r = torch.zeros(batch, device=self.device, dtype=dtype)
        t = torch.ones(batch, device=self.device, dtype=dtype)
        u = self.model(x, r, t, condition)
        return x - u


class ConsistencySampler(Sampler):
    """1-NFE sampler that calls ``f(z, time_eps, c)``."""

    kind = "cd"

    def __init__(self, model: ConsistencyStudent, time_eps: float = 1e-3) -> None:
        if not 0.0 <= time_eps < 1.0:
            raise ValueError("time_eps must be in [0, 1)")
        self.model = model
        self.state_dim = model.state_dim
        self.condition_dim = model.condition_dim
        self.time_eps = float(time_eps)
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, *, noise: torch.Tensor | None = None) -> torch.Tensor:
        condition = condition.to(self.device)
        batch = condition.shape[0]
        dtype = condition.dtype
        x = _ensure_noise(noise, batch, self.state_dim, self.device, dtype)
        t = torch.full((batch,), self.time_eps, device=self.device, dtype=dtype)
        return self.model(x, t, condition)


@dataclass
class LoadedSampler:
    sampler: Sampler
    checkpoint: dict[str, Any]
    stage: str  # 'vol' | 'ret' (without the mf_/cd_ prefix)
    num_actions: int
    normalization: dict[str, float]


def _strip_stage(raw: str) -> str:
    if raw.startswith("mf_") or raw.startswith("cd_"):
        return raw[3:]
    return raw


def load_sampler_from_checkpoint(
    path: str | Path,
    device: str | torch.device = "auto",
    *,
    kind_override: str | None = None,
    fm_n_steps: int = 20,
    consistency_time_eps: float = 1e-3,
) -> LoadedSampler:
    """Load a sampler from a saved checkpoint, dispatching on its ``extra.kind``.

    ``kind_override`` forces a particular sampler kind regardless of the stored
    metadata (useful for legacy checkpoints).
    """

    resolved_device = resolve_device(device) if isinstance(device, str) else device
    ckpt = load_checkpoint(path, map_location=resolved_device)
    extra = ckpt.get("extra", {})
    raw_stage = ckpt.get("stage", "joint")
    stage = _strip_stage(raw_stage)
    num_actions = int(ckpt.get("num_actions", 1))
    normalization = ckpt.get("normalization", {})

    kind = kind_override or extra.get("kind")
    if kind is None:
        if raw_stage.startswith("mf_"):
            kind = "mean_flow"
        elif raw_stage.startswith("cd_"):
            kind = "consistency"
        else:
            kind = "fm"

    config = ckpt["model_config"]
    if kind == "mean_flow":
        model = MeanFlowStudent(
            state_dim=int(config["state_dim"]),
            condition_dim=int(config["condition_dim"]),
            hidden_dim=int(config.get("hidden_dim", 128)),
            time_embedding_dim=int(config.get("time_embedding_dim", 64)),
            num_blocks=int(config.get("num_blocks", 4)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(resolved_device).eval()
        sampler: Sampler = MeanFlowSampler(model)
    elif kind == "consistency":
        model = ConsistencyStudent(
            state_dim=int(config["state_dim"]),
            condition_dim=int(config["condition_dim"]),
            hidden_dim=int(config.get("hidden_dim", 128)),
            time_embedding_dim=int(config.get("time_embedding_dim", 64)),
            num_blocks=int(config.get("num_blocks", 4)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(resolved_device).eval()
        sampler = ConsistencySampler(model, time_eps=consistency_time_eps)
    elif kind == "fm":
        model = TransitionFM(
            state_dim=int(config["state_dim"]),
            condition_dim=int(config["condition_dim"]),
            hidden_dim=int(config.get("hidden_dim", 128)),
            time_embedding_dim=int(config.get("time_embedding_dim", 64)),
            num_blocks=int(config.get("num_blocks", 4)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(resolved_device).eval()
        sampler = FMTeacherSampler(model, n_steps=fm_n_steps)
    else:
        raise ValueError(f"unknown sampler kind '{kind}'")

    return LoadedSampler(
        sampler=sampler,
        checkpoint=ckpt,
        stage=stage,
        num_actions=num_actions,
        normalization=normalization,
    )
