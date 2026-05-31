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
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Enable TensorFloat32 for faster matmul on Ampere+ GPUs (RTX 30/40, A100, etc.)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

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
    progress: bool = True
    progress_min_interval: float = 0.2
    action_dropout_prob: float = 0.0
    scheduled_sampling_max_prob: float = 0.0
    scheduled_sampling_start_epoch: int = 1
    scheduled_sampling_fm_steps: int = 20


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


def _make_progress(
    iterable,
    total: int,
    desc: str,
    disable: bool,
    min_interval: float = 0.2,
):
    # When stderr is not a real TTY (piped to file, captured by pytest, etc.)
    # tqdm cannot rewrite a single line, so crank up the refresh interval so
    # the log file gets one progress line every few seconds instead of one
    # per micro-batch.
    if not disable and not sys.stderr.isatty():
        min_interval = max(min_interval, 10.0)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        disable=disable,
        leave=False,
        dynamic_ncols=True,
        mininterval=min_interval,
        file=sys.stderr,
    )


def evaluate_model(
    model: TransitionFM,
    loader: DataLoader,
    device: torch.device,
    time_eps: float,
    max_batches: int | None = None,
    desc: str = "val",
    disable_progress: bool = False,
    progress_min_interval: float = 0.2,
) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    total = _effective_num_batches(loader, max_batches)
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=total, desc=desc,
        disable=disable_progress, min_interval=progress_min_interval,
    )
    with torch.no_grad():
        for batch in bar:
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            loss = conditional_flow_matching_loss(
                model, condition=condition, target=target, time_eps=time_eps,
            )
            batch_size = condition.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            if not disable_progress:
                bar.set_postfix(loss=f"{total_loss / max(total_items, 1):.4f}", refresh=False)
    bar.close()
    return total_loss / max(total_items, 1)


def train_one_epoch(
    model: TransitionFM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    time_eps: float,
    grad_clip_norm: float,
    max_batches: int | None = None,
    desc: str = "train",
    disable_progress: bool = False,
    progress_min_interval: float = 0.2,
    condition_transform: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    total = _effective_num_batches(loader, max_batches)
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=total, desc=desc,
        disable=disable_progress, min_interval=progress_min_interval,
    )
    for batch in bar:
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if condition_transform is not None:
            condition = condition_transform(batch, condition)
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
        if not disable_progress:
            bar.set_postfix(loss=f"{total_loss / max(total_items, 1):.4f}", refresh=False)
    bar.close()
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
    train_condition_transform_factory: (
        Callable[[int], tuple[Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor] | None, dict[str, Any]]]
        | None
    ) = None,
) -> dict[str, Any]:
    device = resolve_device(train_config.device)
    model = model.to(device)

    # Note: torch.compile disabled on Windows due to Triton compatibility issues
    # Uncomment below on Linux for 10-30% speedup:
    # if device.type == "cuda" and hasattr(torch, "compile"):
    #     try:
    #         model = torch.compile(model, mode="reduce-overhead")
    #     except Exception:
    #         pass

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    # Cosine annealing LR schedule with warmup
    warmup_epochs = max(1, train_config.epochs // 10)
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, train_config.epochs - warmup_epochs)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

    disable_progress = not train_config.progress
    train_batches = _effective_num_batches(train_loader, train_config.max_train_batches)
    val_batches = _effective_num_batches(val_loader, train_config.max_val_batches)
    n_params = sum(p.numel() for p in model.parameters())

    if train_config.progress:
        header = (
            f"[finflow] stage={stage} | run={run_dir.name} | device={device} | "
            f"params={n_params/1e3:.1f}k | train={len(datasets['train'])} samples "
            f"({train_batches} batch/epoch x {train_config.batch_size}) | "
            f"val={len(datasets['val'])} samples ({val_batches} batch) | epochs={train_config.epochs}"
        )
        print(header, file=sys.stderr, flush=True)

    run_start = time.monotonic()

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.monotonic()
        desc_train = f"epoch {epoch:>3}/{train_config.epochs} train"
        desc_val = f"epoch {epoch:>3}/{train_config.epochs} val  "
        condition_transform = None
        epoch_extra: dict[str, Any] = {}
        if train_condition_transform_factory is not None:
            condition_transform, epoch_extra = train_condition_transform_factory(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device=device,
            time_eps=train_config.time_eps, grad_clip_norm=train_config.grad_clip_norm,
            max_batches=train_config.max_train_batches,
            desc=desc_train, disable_progress=disable_progress,
            progress_min_interval=train_config.progress_min_interval,
            condition_transform=condition_transform,
        )
        val_loss = evaluate_model(
            model, val_loader, device=device, time_eps=train_config.time_eps,
            max_batches=train_config.max_val_batches,
            desc=desc_val, disable_progress=disable_progress,
            progress_min_interval=train_config.progress_min_interval,
        )
        epoch_time = time.monotonic() - epoch_start
        global_step += train_batches

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "global_step": global_step,
            "epoch_time_s": epoch_time,
        }
        record.update(epoch_extra)
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer,
            epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
            model_config=model_config_dict, train_config=asdict(train_config),
            normalization=normalization, stage=stage, num_actions=num_actions,
            extra={"train_loss": train_loss, "val_loss": val_loss, **epoch_extra},
        )
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_val_loss,
                model_config=model_config_dict, train_config=asdict(train_config),
                normalization=normalization, stage=stage, num_actions=num_actions,
                extra={"train_loss": train_loss, "val_loss": val_loss, **epoch_extra},
            )

        scheduler.step()

        if train_config.progress:
            elapsed = time.monotonic() - run_start
            eta_s = (elapsed / epoch) * (train_config.epochs - epoch)
            current_lr = optimizer.param_groups[0]['lr']
            line = (
                f"  epoch {epoch:>3}/{train_config.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"{_format_epoch_extra(epoch_extra)}"
                f"best={best_val_loss:.4f}{' *' if is_best else '  '} | "
                f"lr={current_lr:.2e} | "
                f"epoch={_fmt_time(epoch_time)} | elapsed={_fmt_time(elapsed)} | "
                f"eta={_fmt_time(eta_s)}"
            )
            print(line, file=sys.stderr, flush=True)

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
        "total_time_s": time.monotonic() - run_start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:5.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes:d}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m"


def _format_epoch_extra(epoch_extra: dict[str, Any]) -> str:
    if "scheduled_sampling_prob" not in epoch_extra:
        return ""
    return f"sched={float(epoch_extra['scheduled_sampling_prob']):.3f} | "


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
    train_action_dropout_prob: float = 0.0,
) -> dict[str, HestonVolTransitionDataset]:
    data_dir = Path(data_dir)
    return {
        split: HestonVolTransitionDataset(
            data_dir / f"{split}_transitions.npz",
            normalize=True,
            log_v_mean=normalization["log_v_mean"],
            log_v_std=normalization["log_v_std"],
            num_actions=num_actions,
            action_dropout_prob=train_action_dropout_prob if split == "train" else 0.0,
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
    datasets = build_vol_datasets(
        data_dir, normalization, num_actions,
        train_action_dropout_prob=train_config.action_dropout_prob,
    )

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
    train_action_dropout_prob: float = 0.0,
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
            action_dropout_prob=train_action_dropout_prob if split == "train" else 0.0,
        )
        for split in ("train", "val", "test")
    }


def _scheduled_sampling_prob_for_epoch(
    epoch: int,
    total_epochs: int,
    max_prob: float,
    start_epoch: int,
) -> float:
    if max_prob <= 0.0:
        return 0.0
    if not 0.0 <= max_prob <= 1.0:
        raise ValueError("scheduled_sampling_max_prob must be in [0, 1]")
    if start_epoch <= 0:
        raise ValueError("scheduled_sampling_start_epoch must be positive")
    if epoch < start_epoch:
        return 0.0
    active_epochs = max(total_epochs - start_epoch + 1, 1)
    progress = (epoch - start_epoch + 1) / active_epochs
    return min(max_prob, max_prob * progress)


def _make_ret_scheduled_sampling_factory(
    *,
    vol_sampler_checkpoint: str | Path,
    device: torch.device,
    num_actions: int,
    train_config: TransitionFMTrainConfig,
):
    from finflow.inference.samplers import load_sampler_from_checkpoint

    loaded = load_sampler_from_checkpoint(
        vol_sampler_checkpoint,
        device=device,
        fm_n_steps=train_config.scheduled_sampling_fm_steps,
    )
    if loaded.stage != "vol":
        raise ValueError(
            f"vol_sampler_checkpoint must load a vol-stage sampler, got stage='{loaded.stage}'"
        )
    if loaded.num_actions != num_actions:
        raise ValueError(
            f"vol sampler num_actions={loaded.num_actions} does not match ret data num_actions={num_actions}"
        )
    expected_condition_dim = 1 + num_actions
    if loaded.sampler.condition_dim != expected_condition_dim:
        raise ValueError(
            f"vol sampler condition_dim must be {expected_condition_dim}, "
            f"got {loaded.sampler.condition_dim}"
        )

    def factory(epoch: int):
        prob = _scheduled_sampling_prob_for_epoch(
            epoch,
            train_config.epochs,
            train_config.scheduled_sampling_max_prob,
            train_config.scheduled_sampling_start_epoch,
        )

        def transform(
            batch: dict[str, torch.Tensor],
            condition: torch.Tensor,
        ) -> torch.Tensor:
            if prob <= 0.0:
                return condition
            vol_condition = torch.cat([condition[:, 1:2], condition[:, 3:]], dim=1)
            with torch.no_grad():
                sampled_log_v_next = loaded.sampler.sample(vol_condition).to(
                    device=condition.device,
                    dtype=condition.dtype,
                )
            mask = (torch.rand(condition.shape[0], 1, device=condition.device) < prob)
            updated = condition.clone()
            updated[:, 0:1] = torch.where(mask, sampled_log_v_next, updated[:, 0:1])
            return updated

        return transform, {
            "scheduled_sampling_prob": prob,
            "scheduled_sampling_checkpoint": str(Path(vol_sampler_checkpoint).resolve()),
        }

    return factory


def train_ret_trans_fm(
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    num_actions: int | None = None,
    model_config: TwoStageFMModelConfig | None = None,
    train_config: TransitionFMTrainConfig | None = None,
    vol_sampler_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train Stage 1b: ``p(r_{t+1} | v_{t+1}, v_t, r_t, a_t)``.

    By default training uses teacher-forced ground-truth ``v_{t+1}`` from the
    data. If ``vol_sampler_checkpoint`` is provided, scheduled sampling replaces
    a linearly increasing fraction of each training batch's ``v_{t+1}`` with
    Stage 1a sampler output.
    """

    train_config = train_config or TransitionFMTrainConfig()
    set_seed(train_config.seed)

    if num_actions is None:
        num_actions = load_num_actions(data_dir)
    normalization = load_normalization(data_dir)
    datasets = build_ret_datasets(
        data_dir, normalization, num_actions,
        train_action_dropout_prob=train_config.action_dropout_prob,
    )

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
    resolved_device = resolve_device(train_config.device)
    train_condition_transform_factory = None
    scheduled_extra: dict[str, Any] = {"data_dir": str(Path(data_dir).resolve())}
    if vol_sampler_checkpoint is not None:
        train_condition_transform_factory = _make_ret_scheduled_sampling_factory(
            vol_sampler_checkpoint=vol_sampler_checkpoint,
            device=resolved_device,
            num_actions=num_actions,
            train_config=train_config,
        )
        scheduled_extra.update({
            "vol_sampler_checkpoint": str(Path(vol_sampler_checkpoint).resolve()),
            "scheduled_sampling_enabled": train_config.scheduled_sampling_max_prob > 0.0,
        })
    return _run_fm_training(
        model=model,
        datasets=datasets,
        train_config=train_config,
        run_dir=run_dir,
        normalization=normalization,
        model_config_dict=asdict(model_config),
        stage="ret",
        num_actions=num_actions,
        config_blob_extra=scheduled_extra,
        train_condition_transform_factory=train_condition_transform_factory,
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
