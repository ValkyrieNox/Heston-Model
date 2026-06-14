"""Sampler abstractions over trained vol/ret models.

Three concrete samplers share the same ``sample(condition, *, noise=None) -> Tensor``
interface so the rollout module can plug any of them in:

- ``FMTeacherSampler``: multi-step ODE integration of an FM teacher.
- ``MeanFlowSampler``:  single-NFE call to a Mean Flow student.
- ``ConsistencySampler``: single-NFE call to a Consistency student.

A factory ``load_sampler_from_checkpoint`` inspects the saved checkpoint's
``stage`` / ``extra.kind`` fields and instantiates the right model + sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from finflow.models import ConsistencyStudent, MeanFlowStudent, TransitionFM
from finflow.training import load_checkpoint, resolve_device
from finflow.transforms import inverse_lambert_w_transform


class Sampler:
    """One-step conditional sampler interface."""

    state_dim: int
    condition_dim: int
    kind: str  # "fm" | "mf" | "cd"
    device: torch.device
    num_actions: int | None

    def sample(
        self,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        cfg_w: float = 0.0,
    ) -> torch.Tensor:
        raise NotImplementedError


def _ensure_noise(noise, batch, state_dim, device, dtype) -> torch.Tensor:
    if noise is None:
        return torch.randn(batch, state_dim, device=device, dtype=dtype)
    if noise.shape != (batch, state_dim):
        raise ValueError(f"noise shape must be ({batch}, {state_dim}), got {tuple(noise.shape)}")
    return noise.to(device=device, dtype=dtype)


def _unconditional_condition(
    condition: torch.Tensor,
    num_actions: int | None,
    cfg_w: float,
) -> torch.Tensor | None:
    if cfg_w == 0.0:
        return None
    if cfg_w < 0.0:
        raise ValueError("cfg_w must be non-negative")
    if num_actions is None or num_actions <= 0:
        raise ValueError("sampler needs num_actions to use CFG")
    if condition.shape[1] < num_actions:
        raise ValueError(
            f"condition dim {condition.shape[1]} is smaller than num_actions={num_actions}"
        )
    unconditional = condition.clone()
    unconditional[:, -num_actions:] = 0.0
    return unconditional


class FMTeacherSampler(Sampler):
    """Multi-step ODE sampler over a trained Flow Matching teacher."""

    kind = "fm"
    VALID_SOLVERS = {"euler", "heun"}

    def __init__(
        self,
        model: TransitionFM,
        n_steps: int = 20,
        num_actions: int | None = None,
        solver: str = "euler",
    ) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if solver not in self.VALID_SOLVERS:
            raise ValueError(f"solver must be one of {sorted(self.VALID_SOLVERS)}")
        self.model = model
        self.n_steps = n_steps
        self.solver = solver
        self.state_dim = model.state_dim
        self.condition_dim = model.condition_dim
        self.device = next(model.parameters()).device
        self.num_actions = num_actions

    def _velocity(
        self,
        x: torch.Tensor,
        tau: torch.Tensor,
        condition: torch.Tensor,
        unconditional: torch.Tensor | None,
        cfg_w: float,
    ) -> torch.Tensor:
        velocity = self.model(x_tau=x, tau=tau, condition=condition)
        if unconditional is not None:
            velocity_uncond = self.model(x_tau=x, tau=tau, condition=unconditional)
            velocity = (1.0 + cfg_w) * velocity - cfg_w * velocity_uncond
        return velocity

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        cfg_w: float = 0.0,
    ) -> torch.Tensor:
        condition = condition.to(self.device)
        unconditional = _unconditional_condition(condition, self.num_actions, cfg_w)
        batch = condition.shape[0]
        dtype = condition.dtype
        x = _ensure_noise(noise, batch, self.state_dim, self.device, dtype)
        dt = 1.0 / self.n_steps
        for step in range(self.n_steps):
            tau = torch.full((batch,), step * dt, device=self.device, dtype=dtype)
            velocity = self._velocity(x, tau, condition, unconditional, cfg_w)
            if self.solver == "euler":
                x = x + dt * velocity
                continue

            tau_next = torch.full(
                (batch,), min((step + 1) * dt, 1.0), device=self.device, dtype=dtype,
            )
            x_pred = x + dt * velocity
            velocity_next = self._velocity(x_pred, tau_next, condition, unconditional, cfg_w)
            x = x + 0.5 * dt * (velocity + velocity_next)
        return x


class MeanFlowSampler(Sampler):
    """1-NFE sampler that evaluates ``u(z, 0, 1, c)`` once.

    Mean Flow is distilled in the reversed ``data -> noise`` time convention,
    so the generated data endpoint is ``z - u(z, 0, 1, c)``.
    """

    kind = "mf"

    def __init__(self, model: MeanFlowStudent, num_actions: int | None = None) -> None:
        self.model = model
        self.state_dim = model.state_dim
        self.condition_dim = model.condition_dim
        self.device = next(model.parameters()).device
        self.num_actions = num_actions

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        cfg_w: float = 0.0,
    ) -> torch.Tensor:
        condition = condition.to(self.device)
        unconditional = _unconditional_condition(condition, self.num_actions, cfg_w)
        batch = condition.shape[0]
        dtype = condition.dtype
        x = _ensure_noise(noise, batch, self.state_dim, self.device, dtype)
        r = torch.zeros(batch, device=self.device, dtype=dtype)
        t = torch.ones(batch, device=self.device, dtype=dtype)
        u = self.model(x, r, t, condition)
        if unconditional is not None:
            u_uncond = self.model(x, r, t, unconditional)
            u = (1.0 + cfg_w) * u - cfg_w * u_uncond
        return x - u


class ConsistencySampler(Sampler):
    """1-NFE sampler that calls ``f(z, time_eps, c)``."""

    kind = "cd"

    def __init__(
        self,
        model: ConsistencyStudent,
        time_eps: float = 1e-3,
        num_actions: int | None = None,
    ) -> None:
        if not 0.0 <= time_eps < 1.0:
            raise ValueError("time_eps must be in [0, 1)")
        self.model = model
        self.state_dim = model.state_dim
        self.condition_dim = model.condition_dim
        self.time_eps = float(time_eps)
        self.device = next(model.parameters()).device
        self.num_actions = num_actions

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        cfg_w: float = 0.0,
    ) -> torch.Tensor:
        condition = condition.to(self.device)
        unconditional = _unconditional_condition(condition, self.num_actions, cfg_w)
        batch = condition.shape[0]
        dtype = condition.dtype
        x = _ensure_noise(noise, batch, self.state_dim, self.device, dtype)
        t = torch.full((batch,), self.time_eps, device=self.device, dtype=dtype)
        out = self.model(x, t, condition)
        if unconditional is not None:
            out_uncond = self.model(x, t, unconditional)
            out = (1.0 + cfg_w) * out - cfg_w * out_uncond
        return out


class LambertWInverseSampler(Sampler):
    """Wrap a sampler trained on a Lambert-W Gaussianized target.

    The wrapped model emits samples in the Gaussianized domain; we map them
    back to the standardized target domain with the inverse Lambert-W transform
    so the rest of the rollout (denormalization, exp -> variance) is unchanged.
    """

    def __init__(self, base: Sampler, delta: float) -> None:
        if delta <= 0.0:
            raise ValueError("delta must be positive to wrap a sampler")
        self._base = base
        self.delta = float(delta)
        self.state_dim = base.state_dim
        self.condition_dim = base.condition_dim
        self.kind = base.kind
        self.device = base.device
        self.num_actions = base.num_actions

    def sample(
        self,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        cfg_w: float = 0.0,
    ) -> torch.Tensor:
        gaussianized = self._base.sample(condition, noise=noise, cfg_w=cfg_w)
        arr = gaussianized.detach().cpu().numpy()
        restored = inverse_lambert_w_transform(arr, delta=self.delta)
        return torch.as_tensor(restored, dtype=gaussianized.dtype, device=gaussianized.device)


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
    fm_solver: str = "euler",
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
        sampler: Sampler = MeanFlowSampler(model, num_actions=num_actions)
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
        sampler = ConsistencySampler(model, time_eps=consistency_time_eps, num_actions=num_actions)
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
        sampler = FMTeacherSampler(
            model, n_steps=fm_n_steps, num_actions=num_actions, solver=fm_solver,
        )
    else:
        raise ValueError(f"unknown sampler kind '{kind}'")

    # Vol kernels trained with Lambert-W Gaussianized targets emit samples in
    # the Gaussianized domain; wrap so the inverse transform is applied before
    # the rollout denormalizes. Stored under extra.lambert_w_delta (0 = off).
    lambert_w_delta = float(extra.get("lambert_w_delta", 0.0) or 0.0)
    if lambert_w_delta > 0.0:
        sampler = LambertWInverseSampler(sampler, delta=lambert_w_delta)

    return LoadedSampler(
        sampler=sampler,
        checkpoint=ckpt,
        stage=stage,
        num_actions=num_actions,
        normalization=normalization,
    )
