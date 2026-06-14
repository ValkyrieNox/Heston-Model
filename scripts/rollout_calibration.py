"""Sampling-time return calibration shared by rollout CLIs."""

from __future__ import annotations

import numpy as np

from finflow.baselines.quant_gan import calibrate_standardized_moments


def calibrate_return_paths(
    r_paths: np.ndarray,
    *,
    initial_s: float,
    return_mean: float,
    return_std: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Pin pooled return mean/std to training data moments and rebuild prices.

    This applies the same affine standardized-return correction used by the
    QuantGAN baseline. It changes only generated returns and prices; variance
    paths remain untouched by callers.
    """

    standardized, info = calibrate_standardized_moments(r_paths.reshape(-1), eps=eps)
    r_cal = standardized.reshape(r_paths.shape).astype(np.float64) * return_std + return_mean
    cum = np.cumsum(r_cal, axis=1)
    s_tail = float(initial_s) * np.exp(cum)
    s0_col = np.full((s_tail.shape[0], 1), float(initial_s), dtype=s_tail.dtype)
    s_paths = np.concatenate([s0_col, s_tail], axis=1)
    info.update({"return_mean": float(return_mean), "return_std": float(return_std)})
    return r_cal.astype(np.float32), s_paths.astype(np.float32), info
