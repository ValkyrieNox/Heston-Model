"""Dataset loaders for generated Heston transition data.

Three loaders:
- ``HestonTransitionDataset``: legacy single-stage joint dataset (condition
  ``(log_v_t, r_t)``, target ``(log_v_next, r_next)``).
- ``HestonVolTransitionDataset``: V3 Stage 1a dataset (condition
  ``(log_v_t, a_t_onehot)``, target ``log_v_next``).
- ``HestonRetTransitionDataset``: V3 Stage 1b dataset (condition
  ``(log_v_next, log_v_t, r_t, a_t_onehot)``, target ``r_next``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HestonTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """Single-stage joint dataset (kept for backward compatibility)."""

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

    def as_condition_target_tensors(self) -> dict[str, object]:
        """Return vectorized tensors for fast whole-dataset GPU caching."""

        log_v_t = self.arrays["log_v_t"].astype(np.float32, copy=False)
        r_t = self.arrays["r_t"].astype(np.float32, copy=False)
        log_v_next = self.arrays["log_v_next"].astype(np.float32, copy=False)
        r_next = self.arrays["r_next"].astype(np.float32, copy=False)
        if self.normalize:
            log_v_t = (log_v_t - self.log_v_mean) / self.log_v_std
            log_v_next = (log_v_next - self.log_v_mean) / self.log_v_std
            r_t = (r_t - self.return_mean) / self.return_std
            r_next = (r_next - self.return_mean) / self.return_std

        condition = np.stack(
            [
                np.asarray(log_v_t, dtype=np.float32),
                np.asarray(r_t, dtype=np.float32),
            ],
            axis=1,
        )
        target = np.stack(
            [
                np.asarray(log_v_next, dtype=np.float32),
                np.asarray(r_next, dtype=np.float32),
            ],
            axis=1,
        )
        return {
            "condition": torch.from_numpy(np.ascontiguousarray(condition)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "action_start": None,
            "action_dropout_prob": 0.0,
        }


def _load_action_array(npz_files: list[str], npz: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    if "action" in npz_files:
        return np.asarray(npz["action"], dtype=np.int64)
    return np.zeros(n, dtype=np.int64)


def _one_hot(index: int, dim: int) -> torch.Tensor:
    out = torch.zeros(dim, dtype=torch.float32)
    if index < 0 or index >= dim:
        raise IndexError(f"action index {index} out of range [0, {dim})")
    out[index] = 1.0
    return out


def _validate_action_dropout_prob(prob: float) -> float:
    prob = float(prob)
    if not 0.0 <= prob <= 1.0:
        raise ValueError("action_dropout_prob must be in [0, 1]")
    return prob


def _maybe_drop_action(action_onehot: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0.0:
        return action_onehot
    if torch.rand(()) < prob:
        return torch.zeros_like(action_onehot)
    return action_onehot


def _one_hot_matrix(actions: np.ndarray, dim: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int64)
    if np.any((actions < 0) | (actions >= dim)):
        raise IndexError(f"action index out of range [0, {dim})")
    out = np.zeros((actions.shape[0], dim), dtype=np.float32)
    out[np.arange(actions.shape[0]), actions] = 1.0
    return out


class HestonVolTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """V3 Stage 1a: variance transition kernel.

    condition: ``[log_v_t_norm, a_t_onehot]`` of size ``1 + num_actions``
    target:    ``[log_v_next_norm]`` of size ``1``
    """

    def __init__(
        self,
        path: str | Path,
        normalize: bool = False,
        log_v_mean: float = 0.0,
        log_v_std: float = 1.0,
        num_actions: int = 1,
        action_dropout_prob: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.normalize = normalize
        self.log_v_mean = float(log_v_mean)
        self.log_v_std = float(log_v_std)
        self.num_actions = int(num_actions)
        self.action_dropout_prob = _validate_action_dropout_prob(action_dropout_prob)
        if self.log_v_std <= 0:
            raise ValueError("log_v_std must be positive")
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")

        npz = np.load(self.path)
        required = {"log_v_t", "log_v_next"}
        missing = required.difference(npz.files)
        if missing:
            raise ValueError(f"missing transition arrays: {sorted(missing)}")
        self.log_v_t = np.asarray(npz["log_v_t"], dtype=np.float32)
        self.log_v_next = np.asarray(npz["log_v_next"], dtype=np.float32)
        self.action = _load_action_array(list(npz.files), npz, self.log_v_t.shape[0])
        npz.close()

    @property
    def condition_dim(self) -> int:
        return 1 + self.num_actions

    @property
    def state_dim(self) -> int:
        return 1

    def __len__(self) -> int:
        return int(self.log_v_t.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        log_v_t = float(self.log_v_t[index])
        log_v_next = float(self.log_v_next[index])
        action = int(self.action[index])

        if self.normalize:
            log_v_t = (log_v_t - self.log_v_mean) / self.log_v_std
            log_v_next = (log_v_next - self.log_v_mean) / self.log_v_std

        a_onehot = _maybe_drop_action(_one_hot(action, self.num_actions), self.action_dropout_prob)
        condition = torch.cat([torch.tensor([log_v_t], dtype=torch.float32), a_onehot])
        target = torch.tensor([log_v_next], dtype=torch.float32)
        return {
            "condition": condition,
            "target": target,
            "log_v_t": torch.tensor(log_v_t, dtype=torch.float32),
            "log_v_next": target[0],
            "action": torch.tensor(action, dtype=torch.long),
        }

    def as_condition_target_tensors(self) -> dict[str, object]:
        """Return vectorized tensors for fast whole-dataset GPU caching."""

        log_v_t = self.log_v_t.astype(np.float32, copy=False)
        log_v_next = self.log_v_next.astype(np.float32, copy=False)
        if self.normalize:
            log_v_t = (log_v_t - self.log_v_mean) / self.log_v_std
            log_v_next = (log_v_next - self.log_v_mean) / self.log_v_std

        condition = np.concatenate(
            [
                np.asarray(log_v_t, dtype=np.float32).reshape(-1, 1),
                _one_hot_matrix(self.action, self.num_actions),
            ],
            axis=1,
        )
        target = np.asarray(log_v_next, dtype=np.float32).reshape(-1, 1)
        return {
            "condition": torch.from_numpy(np.ascontiguousarray(condition)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "action_start": 1,
            "action_dropout_prob": self.action_dropout_prob,
        }


class HestonRetTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    """V3 Stage 1b: return transition kernel conditioned on already-known v_{t+1}.

    condition: ``[log_v_next_norm, log_v_t_norm, r_t_norm, a_t_onehot]``
               of size ``3 + num_actions``
    target:    ``[r_next_norm]`` of size ``1``

    During training ``log_v_next`` is the ground-truth variance (teacher
    forcing); at inference time it comes from the Stage 1a sampler.
    """

    def __init__(
        self,
        path: str | Path,
        normalize: bool = False,
        log_v_mean: float = 0.0,
        log_v_std: float = 1.0,
        return_mean: float = 0.0,
        return_std: float = 1.0,
        num_actions: int = 1,
        action_dropout_prob: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.normalize = normalize
        self.log_v_mean = float(log_v_mean)
        self.log_v_std = float(log_v_std)
        self.return_mean = float(return_mean)
        self.return_std = float(return_std)
        self.num_actions = int(num_actions)
        self.action_dropout_prob = _validate_action_dropout_prob(action_dropout_prob)
        if self.log_v_std <= 0 or self.return_std <= 0:
            raise ValueError("normalization std values must be positive")
        if self.num_actions <= 0:
            raise ValueError("num_actions must be positive")

        npz = np.load(self.path)
        required = {"log_v_t", "log_v_next", "r_t", "r_next"}
        missing = required.difference(npz.files)
        if missing:
            raise ValueError(f"missing transition arrays: {sorted(missing)}")
        self.log_v_t = np.asarray(npz["log_v_t"], dtype=np.float32)
        self.log_v_next = np.asarray(npz["log_v_next"], dtype=np.float32)
        self.r_t = np.asarray(npz["r_t"], dtype=np.float32)
        self.r_next = np.asarray(npz["r_next"], dtype=np.float32)
        self.action = _load_action_array(list(npz.files), npz, self.log_v_t.shape[0])
        npz.close()

    @property
    def condition_dim(self) -> int:
        return 3 + self.num_actions

    @property
    def state_dim(self) -> int:
        return 1

    def __len__(self) -> int:
        return int(self.r_next.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        log_v_t = float(self.log_v_t[index])
        log_v_next = float(self.log_v_next[index])
        r_t = float(self.r_t[index])
        r_next = float(self.r_next[index])
        action = int(self.action[index])

        if self.normalize:
            log_v_t = (log_v_t - self.log_v_mean) / self.log_v_std
            log_v_next = (log_v_next - self.log_v_mean) / self.log_v_std
            r_t = (r_t - self.return_mean) / self.return_std
            r_next = (r_next - self.return_mean) / self.return_std

        a_onehot = _maybe_drop_action(_one_hot(action, self.num_actions), self.action_dropout_prob)
        condition = torch.cat(
            [torch.tensor([log_v_next, log_v_t, r_t], dtype=torch.float32), a_onehot]
        )
        target = torch.tensor([r_next], dtype=torch.float32)
        return {
            "condition": condition,
            "target": target,
            "log_v_next": torch.tensor(log_v_next, dtype=torch.float32),
            "log_v_t": torch.tensor(log_v_t, dtype=torch.float32),
            "r_t": torch.tensor(r_t, dtype=torch.float32),
            "r_next": target[0],
            "action": torch.tensor(action, dtype=torch.long),
        }

    def as_condition_target_tensors(self) -> dict[str, object]:
        """Return vectorized tensors for fast whole-dataset GPU caching."""

        log_v_t = self.log_v_t.astype(np.float32, copy=False)
        log_v_next = self.log_v_next.astype(np.float32, copy=False)
        r_t = self.r_t.astype(np.float32, copy=False)
        r_next = self.r_next.astype(np.float32, copy=False)
        if self.normalize:
            log_v_t = (log_v_t - self.log_v_mean) / self.log_v_std
            log_v_next = (log_v_next - self.log_v_mean) / self.log_v_std
            r_t = (r_t - self.return_mean) / self.return_std
            r_next = (r_next - self.return_mean) / self.return_std

        condition = np.concatenate(
            [
                np.stack(
                    [
                        np.asarray(log_v_next, dtype=np.float32),
                        np.asarray(log_v_t, dtype=np.float32),
                        np.asarray(r_t, dtype=np.float32),
                    ],
                    axis=1,
                ),
                _one_hot_matrix(self.action, self.num_actions),
            ],
            axis=1,
        )
        target = np.asarray(r_next, dtype=np.float32).reshape(-1, 1)
        return {
            "condition": torch.from_numpy(np.ascontiguousarray(condition)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "action_start": 3,
            "action_dropout_prob": self.action_dropout_prob,
        }
