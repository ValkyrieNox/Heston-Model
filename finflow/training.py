"""Training and evaluation helpers for the V3 FM stages.

Two public entry families:

1. Single-stage joint model (legacy, kept for back compat with V3 baseline):
   ``TransitionFMModelConfig`` / ``TransitionFMTrainConfig`` / ``train_transition_fm``.

2. V3 two-stage models:
   - Stage 1a (variance kernel): ``train_vol_trans_fm``
   - Stage 1b (return kernel):   ``train_ret_trans_fm``
   Both share ``TwoStageFMModelConfig`` and ``TransitionFMTrainConfig`` and
   reuse the same ``TransitionFM`` backbone with different
   ``(state_dim, condition_dim)`` shapes.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from finflow.data import (
    HestonRetTransitionDataset,
    HestonTransitionDataset,
    HestonVolTransitionDataset,
)
from finflow.models import TransitionFM, conditional_flow_matching_loss


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionFMModelConfig:
    """Single-stage joint model config (legacy)."""

    state_dim: int = 2
    condition_dim: int = 2
    hidden_dim: int = 128
    time_embedding_dim: int = 64
    num_blocks: int = 4


@dataclass(frozen=True)
class TwoStageFMModelConfig:
    """V3 two-stage model config. Used by both vol and ret stages."""

    state_dim: int
    condition_dim: int
    hidden_dim: int = 128
    time_embedding_dim: int = 64
    num_blocks: int = 4


@dataclass(frozen=True)
class TransitionFMTrainConfig:
    """Shared training hyperparameters."""

    batch_size: int = 512
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    time_eps: float = 1e-4
    num_workers: int = 0
    seed: int = 1234
    device: str = "auto"
    log_every: int = 50
    max_train_batches: int | None = None
    max_val_batches: int | None = None


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_metadata(data_dir: str | Path) -> dict[str, Any]:
    path = Path(data_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_normalization(data_dir: str | Path) -> dict[str, float]:
    metadata = load_metadata(data_dir)
    normalization = metadata.get("normalization")
    if not normalization:
        raise ValueError("metadata.json is missing normalization stats")
    return {
        "log_v_mean": float(normalization["log_v_mean"]),
        "log_v_std": float(normalization["log_v_std"]),
        "return_mean": float(normalization["return_mean"]),
        "return_std": float(normalization["return_std"]),
    }


def load_num_actions(data_dir: str | Path) -> int:
    metadata = load_metadata(data_dir)
    return int(metadata.get("num_actions", 1))


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _iterate_batches(loader: DataLoader, max_batches: int | None):
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        yield batch


def _effective_num_batches(loader: DataLoader, max_batches: int | None) -> int:
    if max_batches is None:
        return len(loader)
    return min(len(loader), max_batches)


def evaluate_model(
    model: TransitionFM,
    loader: DataLoader,
    device: torch.device,
    time_eps: float,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in _iterate_batches(loader, max_batches):
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            loss = conditional_flow_matching_loss(
                model, condition=condition, target=target, time_eps=time_eps,
            )
            batch_size = condition.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
    return total_loss / max(total_items, 1)


def train_one_epoch(
    model: TransitionFM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    time_eps: float,
    grad_clip_norm: float,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in _iterate_batches(loader, max_batches):
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = conditional_flow_matching_loss(
            model, condition=condition, target=target, time_eps=time_eps,
        )
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        batch_size = condition.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def save_checkpoint(
    path: str | Path,
    model: TransitionFM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    normalization: dict[str, float],
    stage: str = "joint",
    num_actions: int = 1,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_config": model_config,
        "train_config": train_config,
        "normalization": normalization,
        "stage": stage,
        "num_actions": int(num_actions),
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def build_run_dir(output_dir: str | Path, run_name: str | None = None, prefix: str = "transition_fm") -> Path:
    output_dir = Path(output_dir)
    if run_name is None:
        run_name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------


def _run_fm_training(
    model: TransitionFM,
    datasets: dict[str, Dataset],
    train_config: TransitionFMTrainConfig,
    run_dir: Path,
    normalization: dict[str, float],
    model_config_dict: dict[str, Any],
    stage: str,
    num_actions: int,
    config_blob_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = resolve_device(train_config.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    train_loader = build_dataloader(
        datasets["train"], batch_size=train_config.batch_size, shuffle=True,
        num_workers=train_config.num_workers, device=device,
    )
    val_loader = build_dataloader(
        datasets["val"], batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, device=device,
    )

    config_blob = {
        "run_dir": str(run_dir.resolve()),
        "stage": stage,
        "num_actions": num_actions,
        "model_config": model_config_dict,
        "train_config": asdict(train_config),
        "normalization": normalization,
    }
    if config_blob_extra:
        config_blob.update(config_blob_extra)
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    best_val_loss = float("inf")
    global_step = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, train_config.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device=device,
            time_eps=train_config.time_eps, grad_clip_norm=train_config.grad_clip_norm,
            max_batches=train_config.max_train_batches,
        )
        val_loss = evaluate_model(
            model, val_loader, device=device, time_eps=train_config.time_eps,
            max_batches=train_config.max_val_batches,
        )
        global_step += _effective_num_batches(train_loader, train_config.max_train_batches)

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "global_step": global_step}
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer,
            epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
            model_config=model_config_dict, train_config=asdict(train_config),
            normalization=normalization, stage=stage, num_actions=num_actions,
            extra={"train_loss": train_loss, "val_loss": val_loss},
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
                model_config=model_config_dict, train_config=asdict(train_config),
                normalization=normalization, stage=stage, num_actions=num_actions,
                extra={"train_loss": train_loss, "val_loss": val_loss},
            )

    summary = {
        "run_dir": str(run_dir),
        "stage": stage,
        "num_actions": num_actions,
        "checkpoints": {
            "best": str(ckpt_dir / "best.pt"),
            "last": str(ckpt_dir / "last.pt"),
        },
        "best_val_loss": best_val_loss,
        "history": history,
        "device": str(device),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Stage: single-stage joint (legacy)
# ---------------------------------------------------------------------------


def build_transition_datasets(
    data_dir: str | Path,
    normalization: dict[str, float],
) -> dict[str, HestonTransitionDataset]:
    data_dir = Path(data_dir)
    return {
        split: HestonTransitionDataset(
            data_dir / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            return_mean=normalization["return_mean"],
            return_std=normalization["return_std"],
        )
        for split in ("train", "val", "test")
    }


def build_model(config: TransitionFMModelConfig) -> TransitionFM:
    return TransitionFM(**asdict(config))


def train_transition_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    model_config: TransitionFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
) -> dict[str, Any]:
    """Legacy: single-stage joint FM. Predicts (v_{t+1}, r_{t+1}) given (v_t, r_t)."""

    model_config = model_config or TransitionFMModelConfig()
    train_config = train_config or TransitionFMTrainConfig()

    set_seed(train_config.seed)
    normalization = load_normalization(data_dir)
    datasets = build_transition_datasets(data_dir, normalization)
    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="transition_fm")
    model = build_model(model_config)
    return _run_fm_training(
        model=model,
        datasets=datasets,
        train_config=train_config,
        run_dir=run_dir,
        normalization=normalization,
        model_config_dict=asdict(model_config),
        stage="joint",
        num_actions=load_num_actions(data_dir),
        config_blob_extra={"data_dir": str(Path(data_dir).resolve())},
    )


def load_model_from_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu",
) -> tuple[TransitionFM, dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    config = checkpoint["model_config"]
    if "state_dim" not in config or "condition_dim" not in config:
        raise ValueError("checkpoint model_config missing state_dim/condition_dim")
    model = TransitionFM(
        state_dim=int(config["state_dim"]),
        condition_dim=int(config["condition_dim"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        time_embedding_dim=int(config.get("time_embedding_dim", 64)),
        num_blocks=int(config.get("num_blocks", 4)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    split: str = "val",
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a single-stage joint FM checkpoint on Heston transitions."""

    resolved_device = resolve_device(device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, map_location=resolved_device)
    normalization = checkpoint["normalization"]
    dataset = HestonTransitionDataset(
        Path(data_dir) / f"{split}_transitions.npz",
        normalize=True,
        log_v_mean=normalization["log_v_mean"],
        log_v_std=normalization["log_v_std"],
        return_mean=normalization["return_mean"],
        return_std=normalization["return_std"],
    )
    loader = build_dataloader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, device=resolved_device,
    )
    loss = evaluate_model(
        model, loader, device=resolved_device,
        time_eps=float(checkpoint["train_config"]["time_eps"]), max_batches=max_batches,
    )
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "split": split,
        "loss": loss,
        "device": str(resolved_device),
        "normalization": normalization,
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint["train_config"],
    }


# ---------------------------------------------------------------------------
# Stage 1a: variance transition kernel
# ---------------------------------------------------------------------------


def build_vol_datasets(
    data_dir: str | Path,
    normalization: dict[str, float],
    num_actions: int,
) -> dict[str, HestonVolTransitionDataset]:
    data_dir = Path(data_dir)
    return {
        split: HestonVolTransitionDataset(
            data_dir / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            num_actions=num_actions,
        )
        for split in ("train", "val", "test")
    }


def train_vol_trans_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    num_actions: int | None = None,
    model_config: TwoStageFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
) -> dict[str, Any]:
    """Train Stage 1a: ``p(v_{t+1} | v_t, a_t)``.

    If ``num_actions`` is None it is read from the data ``metadata.json``.
    ``model_config`` defaults to ``state_dim=1, condition_dim=1+num_actions``.
    """

    train_config = train_config or TransitionFMTrainConfig()
    set_seed(train_config.seed)

    if num_actions is None:
        num_actions = load_num_actions(data_dir)
    normalization = load_normalization(data_dir)
    datasets = build_vol_datasets(data_dir, normalization, num_actions)

    if model_config is None:
        model_config = TwoStageFMModelConfig(
            state_dim=1, condition_dim=1 + num_actions,
        )
    else:
        expected = 1 + num_actions
        if model_config.state_dim != 1:
            raise ValueError("vol-stage model_config.state_dim must be 1")
        if model_config.condition_dim != expected:
            raise ValueError(
                f"vol-stage condition_dim must be 1 + num_actions = {expected},"
                f" got {model_config.condition_dim}"
            )

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="vol_trans_fm")
    model = TransitionFM(**asdict(model_config))
    return _run_fm_training(
        model=model,
        datasets=datasets,
        train_config=train_config,
        run_dir=run_dir,
        normalization=normalization,
        model_config_dict=asdict(model_config),
        stage="vol",
        num_actions=num_actions,
        config_blob_extra={"data_dir": str(Path(data_dir).resolve())},
    )


# ---------------------------------------------------------------------------
# Stage 1b: return transition kernel
# ---------------------------------------------------------------------------


def build_ret_datasets(
    data_dir: str | Path,
    normalization: dict[str, float],
    num_actions: int,
) -> dict[str, HestonRetTransitionDataset]:
    data_dir = Path(data_dir)
    return {
        split: HestonRetTransitionDataset(
            data_dir / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            return_mean=normalization["return_mean"],
            return_std=normalization["return_std"],
            num_actions=num_actions,
        )
        for split in ("train", "val", "test")
    }


def train_ret_trans_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    num_actions: int | None = None,
    model_config: TwoStageFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
) -> dict[str, Any]:
    """Train Stage 1b: ``p(r_{t+1} | v_{t+1}, v_t, r_t, a_t)``.

    Training uses teacher-forced ground-truth ``v_{t+1}`` from the data. At
    inference time the Stage 1a sampler supplies ``v_{t+1}``.
    """

    train_config = train_config or TransitionFMTrainConfig()
    set_seed(train_config.seed)

    if num_actions is None:
        num_actions = load_num_actions(data_dir)
    normalization = load_normalization(data_dir)
    datasets = build_ret_datasets(data_dir, normalization, num_actions)

    if model_config is None:
        model_config = TwoStageFMModelConfig(
            state_dim=1, condition_dim=3 + num_actions,
        )
    else:
        expected = 3 + num_actions
        if model_config.state_dim != 1:
            raise ValueError("ret-stage model_config.state_dim must be 1")
        if model_config.condition_dim != expected:
            raise ValueError(
                f"ret-stage condition_dim must be 3 + num_actions = {expected},"
                f" got {model_config.condition_dim}"
            )

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="ret_trans_fm")
    model = TransitionFM(**asdict(model_config))
    return _run_fm_training(
        model=model,
        datasets=datasets,
        train_config=train_config,
        run_dir=run_dir,
        normalization=normalization,
        model_config_dict=asdict(model_config),
        stage="ret",
        num_actions=num_actions,
        config_blob_extra={"data_dir": str(Path(data_dir).resolve())},
    )


# ---------------------------------------------------------------------------
# Shared evaluation entry for two-stage checkpoints
# ---------------------------------------------------------------------------


def evaluate_two_stage_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    stage: Literal["vol", "ret"],
    split: str = "val",
    batch_size: int = 512,
    num_workers: int = 0,
    device: str = "auto",
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate a vol- or ret-stage FM checkpoint."""

    resolved_device = resolve_device(device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, map_location=resolved_device)
    normalization = checkpoint["normalization"]
    num_actions = int(checkpoint.get("num_actions", 1))
    ckpt_stage = checkpoint.get("stage", "joint")
    if ckpt_stage != stage:
        raise ValueError(f"checkpoint stage='{ckpt_stage}' does not match requested stage='{stage}'")

    if stage == "vol":
        dataset = HestonVolTransitionDataset(
            Path(data_dir) / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            num_actions=num_actions,
        )
    elif stage == "ret":
        dataset = HestonRetTransitionDataset(
            Path(data_dir) / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            return_mean=normalization["return_mean"],
            return_std=normalization["return_std"],
            num_actions=num_actions,
        )
    else:
        raise ValueError("stage must be 'vol' or 'ret'")

    loader = build_dataloader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, device=resolved_device,
    )
    loss = evaluate_model(
        model, loader, device=resolved_device,
        time_eps=float(checkpoint["train_config"]["time_eps"]), max_batches=max_batches,
    )
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "stage": stage,
        "split": split,
        "loss": loss,
        "device": str(resolved_device),
        "normalization": normalization,
        "num_actions": num_actions,
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint["train_config"],
    }
