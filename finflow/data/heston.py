"""Heston data generation for the V3 autoregressive world-model pipeline.

The variance process is simulated with Andersen's quadratic-exponential (QE)
scheme. The log-price update uses the QE-M martingale-style discretization,
which couples the return innovation to the variance transition without ever
making the variance negative.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """Default Heston parameters used in the V3 notes."""

    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    v0: float = 0.04
    s0: float = 100.0
    mu: float = 0.05
    dt: float = 1.0 / 252.0


def _validate_params(params: HestonParams) -> None:
    if params.kappa <= 0:
        raise ValueError("kappa must be positive")
    if params.theta <= 0:
        raise ValueError("theta must be positive")
    if params.xi < 0:
        raise ValueError("xi must be non-negative")
    if not -1.0 <= params.rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    if params.v0 < 0:
        raise ValueError("v0 must be non-negative")
    if params.s0 <= 0:
        raise ValueError("s0 must be positive")
    if params.dt <= 0:
        raise ValueError("dt must be positive")


def _qe_variance_step(
    v_t: np.ndarray,
    rng: np.random.Generator,
    params: HestonParams,
    psi_c: float,
) -> np.ndarray:
    """One Andersen QE step for the CIR variance process."""

    if params.xi == 0.0:
        exp_kdt = np.exp(-params.kappa * params.dt)
        return params.theta + (v_t - params.theta) * exp_kdt

    exp_kdt = np.exp(-params.kappa * params.dt)
    m = params.theta + (v_t - params.theta) * exp_kdt
    s2 = (
        v_t
        * params.xi**2
        * exp_kdt
        * (1.0 - exp_kdt)
        / params.kappa
        + params.theta
        * params.xi**2
        * (1.0 - exp_kdt) ** 2
        / (2.0 * params.kappa)
    )

    eps = np.finfo(np.float64).eps
    m = np.maximum(m, eps)
    psi = s2 / np.maximum(m * m, eps)

    z = rng.standard_normal(v_t.shape)
    u = rng.random(v_t.shape)
    v_next = np.empty_like(v_t, dtype=np.float64)

    quadratic = psi <= psi_c
    if np.any(quadratic):
        psi_q = np.maximum(psi[quadratic], eps)
        inv_psi = 2.0 / psi_q
        b2 = inv_psi - 1.0 + np.sqrt(inv_psi) * np.sqrt(np.maximum(inv_psi - 1.0, 0.0))
        a = m[quadratic] / (1.0 + b2)
        v_next[quadratic] = a * (np.sqrt(np.maximum(b2, 0.0)) + z[quadratic]) ** 2

    exponential = ~quadratic
    if np.any(exponential):
        psi_e = psi[exponential]
        p = np.clip((psi_e - 1.0) / (psi_e + 1.0), 0.0, 1.0 - eps)
        beta = (1.0 - p) / m[exponential]
        u_e = u[exponential]
        draws = np.zeros_like(u_e)
        nonzero = u_e > p
        draws[nonzero] = np.log((1.0 - p[nonzero]) / (1.0 - u_e[nonzero])) / beta[nonzero]
        v_next[exponential] = draws

    return np.maximum(v_next, 0.0)


def _qe_m_log_return(
    v_t: np.ndarray,
    v_next: np.ndarray,
    rng: np.random.Generator,
    params: HestonParams,
    gamma1: float,
    gamma2: float,
) -> np.ndarray:
    """QE-M log-return update conditional on the variance transition."""

    if params.xi == 0.0:
        vol = np.sqrt(np.maximum(v_t, 0.0) * params.dt)
        z = rng.standard_normal(v_t.shape)
        return (params.mu - 0.5 * v_t) * params.dt + vol * z

    z = rng.standard_normal(v_t.shape)
    rho = params.rho
    xi = params.xi
    kappa = params.kappa
    dt = params.dt

    k0 = -rho * kappa * params.theta * dt / xi
    k1 = gamma1 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = gamma2 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = gamma1 * dt * (1.0 - rho * rho)
    k4 = gamma2 * dt * (1.0 - rho * rho)
    conditional_var = np.maximum(k3 * v_t + k4 * v_next, 0.0)
    return params.mu * dt + k0 + k1 * v_t + k2 * v_next + np.sqrt(conditional_var) * z


def simulate_heston_qe(
    n_paths: int,
    n_steps: int = 252,
    params: HestonParams | None = None,
    seed: int | None = None,
    dtype: np.dtype[Any] = np.float32,
    psi_c: float = 1.5,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
) -> dict[str, np.ndarray]:
    """Simulate Heston paths with Andersen QE variance and QE-M returns.

    Returns:
        A dict with `s_paths` and `v_paths` of shape `[n_paths, n_steps + 1]`,
        plus `log_returns` of shape `[n_paths, n_steps]`.
    """

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if psi_c <= 0:
        raise ValueError("psi_c must be positive")
    if gamma1 < 0 or gamma2 < 0:
        raise ValueError("gamma1 and gamma2 must be non-negative")

    params = params or HestonParams()
    _validate_params(params)
    rng = np.random.default_rng(seed)

    v_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_s_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    log_returns = np.empty((n_paths, n_steps), dtype=np.float64)
    v_paths[:, 0] = params.v0
    log_s_paths[:, 0] = np.log(params.s0)

    for step in range(n_steps):
        v_t = v_paths[:, step]
        v_next = _qe_variance_step(v_t, rng, params, psi_c)
        r_t = _qe_m_log_return(v_t, v_next, rng, params, gamma1, gamma2)
        v_paths[:, step + 1] = v_next
        log_returns[:, step] = r_t
        log_s_paths[:, step + 1] = log_s_paths[:, step] + r_t

    return {
        "s_paths": np.exp(log_s_paths).astype(dtype, copy=False),
        "v_paths": v_paths.astype(dtype, copy=False),
        "log_returns": log_returns.astype(dtype, copy=False),
    }


def build_transition_arrays(
    v_paths: np.ndarray,
    log_returns: np.ndarray,
    eps: float = 1e-8,
    dtype: np.dtype[Any] = np.float32,
    include_index: bool = True,
) -> dict[str, np.ndarray]:
    """Flatten Heston paths into V3 one-step transition samples.

    Alignment:
        condition at step t: `(v_t, r_{t-1})`, where `r_{-1}=0`
        target at step t: `(v_{t+1}, r_t)`

    This gives exactly `n_paths * n_steps` transition samples and supports
    rollout from the initial state `(v0, 0)`.
    """

    if v_paths.ndim != 2:
        raise ValueError("v_paths must have shape [n_paths, n_steps + 1]")
    if log_returns.ndim != 2:
        raise ValueError("log_returns must have shape [n_paths, n_steps]")
    n_paths, n_steps = log_returns.shape
    if v_paths.shape != (n_paths, n_steps + 1):
        raise ValueError("v_paths shape must match log_returns shape")
    if eps <= 0:
        raise ValueError("eps must be positive")

    previous_returns = np.zeros_like(log_returns)
    previous_returns[:, 1:] = log_returns[:, :-1]

    transitions = {
        "v_t": v_paths[:, :-1].reshape(-1).astype(dtype, copy=False),
        "r_t": previous_returns.reshape(-1).astype(dtype, copy=False),
        "v_next": v_paths[:, 1:].reshape(-1).astype(dtype, copy=False),
        "r_next": log_returns.reshape(-1).astype(dtype, copy=False),
        "log_v_t": np.log(v_paths[:, :-1].reshape(-1) + eps).astype(dtype, copy=False),
        "log_v_next": np.log(v_paths[:, 1:].reshape(-1) + eps).astype(dtype, copy=False),
    }
    if include_index:
        transitions["path_index"] = np.repeat(np.arange(n_paths, dtype=np.int32), n_steps)
        transitions["step_index"] = np.tile(np.arange(n_steps, dtype=np.int16), n_paths)
    return transitions


def _stats_from_train(train: dict[str, np.ndarray], eps: float) -> dict[str, float]:
    log_v = np.log(train["v_paths"] + eps)
    returns = train["log_returns"]
    return {
        "log_v_mean": float(log_v.mean()),
        "log_v_std": float(log_v.std(ddof=0) + eps),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=0) + eps),
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def generate_heston_dataset(
    output_dir: str | Path,
    n_train: int = 50_000,
    n_val: int = 5_000,
    n_test: int = 10_000,
    n_steps: int = 252,
    params: HestonParams | None = None,
    seed: int = 1234,
    save_transitions: bool = True,
    dtype: np.dtype[Any] = np.float32,
) -> dict[str, Any]:
    """Generate train/val/test Heston datasets and write them to disk."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    params = params or HestonParams()

    split_sizes = {"train": n_train, "val": n_val, "test": n_test}
    split_seeds = {"train": seed, "val": seed + 1, "test": seed + 2}
    generated: dict[str, dict[str, np.ndarray]] = {}

    for split, size in split_sizes.items():
        arrays = simulate_heston_qe(
            n_paths=size,
            n_steps=n_steps,
            params=params,
            seed=split_seeds[split],
            dtype=dtype,
        )
        generated[split] = arrays
        _save_npz(output / f"{split}.npz", arrays)
        if save_transitions:
            transitions = build_transition_arrays(arrays["v_paths"], arrays["log_returns"], dtype=dtype)
            _save_npz(output / f"{split}_transitions.npz", transitions)

    stats = _stats_from_train(generated["train"], eps=1e-8)
    metadata = {
        "params": asdict(params),
        "n_steps": n_steps,
        "split_sizes": split_sizes,
        "seed": seed,
        "save_transitions": save_transitions,
        "transition_alignment": {
            "condition": "(v_t, r_{t-1}) with r_{-1}=0",
            "target": "(v_{t+1}, r_t)",
        },
        "normalization": stats,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

