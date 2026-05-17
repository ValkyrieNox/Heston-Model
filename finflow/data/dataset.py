"""Dataset loaders for generated Heston transition data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HestonTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """Load flattened V3 transitions from a generated `*_transitions.npz` file."""

    def __init__(
        self,
        path: str | Path,
        normalize: bool = False,
        log_v_mean: float = 0.0,
        log_v_std: float = 1.0,
        return_mean: float = 0.0,
        return_std: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.normalize = normalize
        self.log_v_mean = float(log_v_mean)
        self.log_v_std = float(log_v_std)
        self.return_mean = float(return_mean)
        self.return_std = float(return_std)
        if self.log_v_std <= 0 or self.return_std <= 0:
            raise ValueError("normalization std values must be positive")

        npz = np.load(self.path)
        required = {"log_v_t", "r_t", "log_v_next", "r_next"}
        missing = required.difference(npz.files)
        if missing:
            raise ValueError(f"missing transition arrays: {sorted(missing)}")
        self.arrays = {name: np.asarray(npz[name], dtype=np.float32) for name in required}
        npz.close()

    def __len__(self) -> int:
        return int(self.arrays["r_next"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        log_v_t = float(self.arrays["log_v_t"][index])
        r_t = float(self.arrays["r_t"][index])
        log_v_next = float(self.arrays["log_v_next"][index])
        r_next = float(self.arrays["r_next"][index])

        if self.normalize:
            log_v_t = (log_v_t - self.log_v_mean) / self.log_v_std
            log_v_next = (log_v_next - self.log_v_mean) / self.log_v_std
            r_t = (r_t - self.return_mean) / self.return_std
            r_next = (r_next - self.return_mean) / self.return_std

        condition = torch.tensor([log_v_t, r_t], dtype=torch.float32)
        target = torch.tensor([log_v_next, r_next], dtype=torch.float32)
        return {
            "condition": condition,
            "target": target,
            "log_v_t": condition[0],
            "r_t": condition[1],
            "log_v_next": target[0],
            "r_next": target[1],
        }
