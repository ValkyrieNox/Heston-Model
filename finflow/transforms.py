"""Lambert-W heavy-tail transforms (Goerg 2015), shared across the codebase.

The Quant GAN baseline (Wiese et al. 2020) Gaussianizes heavy-tailed *returns*
before training and re-injects tails at sampling. We reuse the exact same
transform for the Flow Matching *variance kernel* (vol stage): the per-step
Heston log-returns are conditionally Gaussian given the variance, so the heavy
tails come from the dispersion of the variance process. Gaussianizing the
log-variance target makes it easy for a Gaussian-prior flow to learn, and the
inverse transform restores the heavy upper tail of the variance at sampling.

Lives in its own module (no finflow imports) so both ``finflow.data`` and
``finflow.baselines`` can import it without a circular dependency.
"""

from __future__ import annotations

import numpy as np
import torch


def _lambertw_principal_nonnegative(z: np.ndarray, *, max_iter: int = 20) -> np.ndarray:
    """Principal Lambert W branch for non-negative real inputs."""

    z = np.asarray(z, dtype=np.float64)
    if np.any(z < 0):
        raise ValueError("Lambert W fallback expects non-negative inputs")
    w = np.where(z < 1.0, z / (1.0 + z), np.log1p(z))
    w = np.maximum(w, 0.0)
    for _ in range(max_iter):
        ew = np.exp(w)
        f = w * ew - z
        denom = ew * (w + 1.0) - ((w + 2.0) * f) / np.maximum(2.0 * w + 2.0, 1e-12)
        step = f / np.maximum(denom, 1e-12)
        w_next = np.maximum(w - step, 0.0)
        if np.max(np.abs(w_next - w)) < 1e-12:
            return w_next
        w = w_next
    return w


def lambert_w_transform(values: np.ndarray, delta: float = 0.1) -> np.ndarray:
    """Gaussianize heavy-tailed standardized values with the Lambert-W inverse.

    The input is expected to already be centered and scaled. ``delta=0`` keeps
    the identity transform.
    """

    x = np.asarray(values, dtype=np.float64)
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if delta == 0:
        return x.astype(np.float32, copy=False)
    z = delta * np.square(x)
    w = _lambertw_principal_nonnegative(z)
    y = np.sign(x) * np.sqrt(np.maximum(w, 0.0) / delta)
    return y.astype(np.float32, copy=False)


def inverse_lambert_w_transform(values: np.ndarray, delta: float = 0.1) -> np.ndarray:
    """Map Lambert-W Gaussianized values back to the standardized domain."""

    y = np.asarray(values, dtype=np.float64)
    if delta < 0:
        raise ValueError("delta must be non-negative")
    if delta == 0:
        return y.astype(np.float32, copy=False)
    exponent = np.clip(0.5 * delta * np.square(y), 0.0, 20.0)
    x = y * np.exp(exponent)
    return x.astype(np.float32, copy=False)


def _lambertw_principal_nonnegative_torch(
    z: torch.Tensor,
    *,
    max_iter: int = 20,
) -> torch.Tensor:
    """Differentiable principal Lambert W branch for non-negative torch inputs."""

    if torch.any(z < 0):
        raise ValueError("Lambert W fallback expects non-negative inputs")
    w = torch.where(z < 1.0, z / (1.0 + z), torch.log1p(z))
    w = torch.clamp_min(w, 0.0)
    for _ in range(max_iter):
        ew = torch.exp(w)
        f = w * ew - z
        denom = ew * (w + 1.0) - ((w + 2.0) * f) / torch.clamp(2.0 * w + 2.0, min=1e-12)
        step = f / torch.clamp(denom, min=1e-12)
        w = torch.clamp_min(w - step, 0.0)
    return w


def lambert_w_transform_torch(values: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """Torch version of :func:`lambert_w_transform` for differentiable losses."""

    if delta < 0:
        raise ValueError("delta must be non-negative")
    if delta == 0:
        return values
    z = float(delta) * values.square()
    eps = torch.finfo(values.dtype).eps
    z_safe = torch.clamp_min(z, eps)
    w = _lambertw_principal_nonnegative_torch(z_safe)
    # Use the equivalent x * sqrt(W(delta*x^2)/(delta*x^2)) form so the
    # derivative at x=0 follows the identity-limit instead of sign/sqrt NaNs.
    factor = torch.sqrt(torch.clamp_min(w, 0.0) / z_safe)
    factor = torch.where(z > eps, factor, torch.ones_like(factor))
    return values * factor


def inverse_lambert_w_transform_torch(values: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """Torch version of :func:`inverse_lambert_w_transform` for differentiable rollout."""

    if delta < 0:
        raise ValueError("delta must be non-negative")
    if delta == 0:
        return values
    exponent = torch.clamp(0.5 * float(delta) * values.square(), 0.0, 20.0)
    return values * torch.exp(exponent)
