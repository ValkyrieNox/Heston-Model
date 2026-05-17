"""Training and evaluation helpers for the V3 transition FM baseline."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from finflow.data import HestonTransitionDataset
from finflow.models import TransitionFM, conditional_flow_matching_loss


@dataclass(frozen=True)
class TransitionFMModelConfig:
    state_dim: int = 2
    condition_dim: int = 2
    hidden_dim: int = 128
    time_embedding_dim: int = 64
    num_blocks: int = 4


@dataclass(frozen=True)
class TransitionFMTrainConfig:
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


def build_dataloader(
    dataset: HestonTransitionDataset,
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
                model,
                condition=condition,
                target=target,
                time_eps=time_eps,
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
            model,
            condition=condition,
            target=target,
            time_eps=time_eps,
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
    model_config: TransitionFMModelConfig,
    train_config: TransitionFMTrainConfig,
    normalization: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "normalization": normalization,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def build_run_dir(output_dir: str | Path, run_name: str | None = None) -> Path:
    output_dir = Path(output_dir)
    if run_name is None:
        run_name = datetime.now().strftime("transition_fm_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


def train_transition_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    model_config: TransitionFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
) -> dict[str, Any]:
    model_config = model_config or TransitionFMModelConfig()
    train_config = train_config or TransitionFMTrainConfig()

    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    normalization = load_normalization(data_dir)
    datasets = build_transition_datasets(data_dir, normalization)

    run_dir = build_run_dir(output_dir, run_name=run_name)
    ckpt_dir = run_dir / "checkpoints"
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    train_loader = build_dataloader(
        datasets["train"],
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        device=device,
    )
    val_loader = build_dataloader(
        datasets["val"],
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        device=device,
    )

    config_blob = {
        "data_dir": str(Path(data_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "run_dir": str(run_dir.resolve()),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "normalization": normalization,
    }
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.jsonl"
    best_val_loss = float("inf")
    global_step = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, train_config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            time_eps=train_config.time_eps,
            grad_clip_norm=train_config.grad_clip_norm,
            max_batches=train_config.max_train_batches,
        )
        val_loss = evaluate_model(
            model,
            val_loader,
            device=device,
            time_eps=train_config.time_eps,
            max_batches=train_config.max_val_batches,
        )
        global_step += _effective_num_batches(train_loader, train_config.max_train_batches)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "global_step": global_step,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            model_config=model_config,
            train_config=train_config,
            normalization=normalization,
            extra={"train_loss": train_loss, "val_loss": val_loss},
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                model_config=model_config,
                train_config=train_config,
                normalization=normalization,
                extra={"train_loss": train_loss, "val_loss": val_loss},
            )

    summary = {
        "run_dir": str(run_dir),
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


def load_model_from_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[TransitionFM, dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    model = build_model(TransitionFMModelConfig(**checkpoint["model_config"]))
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
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=resolved_device,
    )
    loss = evaluate_model(
        model,
        loader,
        device=resolved_device,
        time_eps=float(checkpoint["train_config"]["time_eps"]),
        max_batches=max_batches,
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
