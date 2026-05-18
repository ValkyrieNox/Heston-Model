"""Consistency Distillation (Song et al. 2023) on a trained FM teacher.

For adjacent points on the same FM ODE trajectory we enforce::

    f_student(x_{t_{n+1}}, t_{n+1}, c) == sg( f_target(x_{t_n}, t_n, c) )

where ``f_target`` is the EMA of ``f_student`` and ``x_{t_n}`` is obtained by
one teacher Euler step from ``x_{t_{n+1}}`` toward ``t_n``::

    x_{t_n} = x_{t_{n+1}} - (t_{n+1} - t_n) * v_teacher(x_{t_{n+1}}, t_{n+1}, c)
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finflow.models import (
    ConsistencyStudent,
    TransitionFM,
    warm_start_consistency_from_fm,
)
from finflow.training import (
    TwoStageFMModelConfig,
    _effective_num_batches,
    _fmt_time,
    _iterate_batches,
    _make_progress,
    build_dataloader,
    build_ret_datasets,
    build_run_dir,
    build_vol_datasets,
    load_model_from_checkpoint,
    load_normalization,
    load_num_actions,
    resolve_device,
    save_checkpoint,
    set_seed,
)


@dataclass(frozen=True)
class ConsistencyDistillConfig:
    """Configuration for Consistency Distillation."""

    teacher_checkpoint: str
    batch_size: int = 512
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    time_eps: float = 1e-3
    num_workers: int = 0
    seed: int = 1234
    device: str = "auto"
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    # CD-specific
    n_discretization: int = 18
    ema_decay: float = 0.999
    warm_start: bool = True
    progress: bool = True
    progress_min_interval: float = 0.2


def _schedule(n_discretization: int, time_eps: float, device, dtype) -> torch.Tensor:
    """Uniform schedule on ``[time_eps, 1 - time_eps]`` with ``n_discretization+1`` points."""

    return torch.linspace(time_eps, 1.0 - time_eps, n_discretization + 1, device=device, dtype=dtype)


def consistency_distill_step(
    student: ConsistencyStudent,
    target_net: ConsistencyStudent,
    teacher: TransitionFM,
    condition: torch.Tensor,
    target: torch.Tensor,
    schedule: torch.Tensor,
) -> torch.Tensor:
    """Single Consistency Distillation training step (loss only)."""

    if target.ndim != 2 or condition.ndim != 2:
        raise ValueError("target and condition must be 2D")

    batch_size = target.shape[0]
    device = target.device
    dtype = target.dtype
    n_segments = schedule.shape[0] - 1  # number of intervals = n_discretization

    noise = torch.randn_like(target)

    # Pick a random index n in {0, ..., n_segments - 1} per sample.
    n_idx = torch.randint(0, n_segments, (batch_size,), device=device)
    t_next = schedule[n_idx + 1]  # shape [B]
    t_curr = schedule[n_idx]      # shape [B]

    t_next_view = t_next.reshape(batch_size, *([1] * (target.ndim - 1)))
    t_curr_view = t_curr.reshape(batch_size, *([1] * (target.ndim - 1)))

    # Construct x at t_next via straight-line CFM interpolation.
    x_next = (1.0 - t_next_view) * noise + t_next_view * target

    # Teacher Euler step toward t_curr (one step backwards in t).
    with torch.no_grad():
        v_teacher = teacher(x_tau=x_next, tau=t_next, condition=condition)
        x_curr_hat = x_next - (t_next_view - t_curr_view) * v_teacher
        # Target value from EMA student.
        target_val = target_net(x_curr_hat, t_curr, condition)

    pred = student(x_next, t_next, condition)
    return F.mse_loss(pred, target_val)


def _ema_update(target_net: torch.nn.Module, source_net: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for p_t, p_s in zip(target_net.parameters(), source_net.parameters()):
            p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)


def _train_one_epoch_consistency(
    student: ConsistencyStudent,
    target_net: ConsistencyStudent,
    teacher: TransitionFM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    schedule: torch.Tensor,
    device: torch.device,
    *,
    ema_decay: float,
    grad_clip_norm: float,
    max_batches: int | None,
    disable_progress: bool,
    progress_min_interval: float,
    desc: str,
) -> float:
    student.train()
    target_net.eval()
    teacher.eval()
    total = 0.0
    seen = 0
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=_effective_num_batches(loader, max_batches),
        desc=desc, disable=disable_progress, min_interval=progress_min_interval,
    )
    for batch in bar:
        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = consistency_distill_step(
            student, target_net, teacher, condition, target, schedule,
        )
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip_norm)
        optimizer.step()
        _ema_update(target_net, student, ema_decay)
        bs = condition.shape[0]
        total += float(loss.item()) * bs
        seen += bs
        if not disable_progress:
            bar.set_postfix(loss=f"{total / max(seen, 1):.4f}", refresh=False)
    bar.close()
    return total / max(seen, 1)


def _evaluate_consistency(
    student: ConsistencyStudent,
    target_net: ConsistencyStudent,
    teacher: TransitionFM,
    loader: DataLoader,
    schedule: torch.Tensor,
    device: torch.device,
    *,
    max_batches: int | None,
    disable_progress: bool,
    progress_min_interval: float,
    desc: str,
) -> float:
    student.eval()
    target_net.eval()
    teacher.eval()
    total = 0.0
    seen = 0
    bar = _make_progress(
        _iterate_batches(loader, max_batches),
        total=_effective_num_batches(loader, max_batches),
        desc=desc, disable=disable_progress, min_interval=progress_min_interval,
    )
    with torch.no_grad():
        for batch in bar:
            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            loss = consistency_distill_step(
                student, target_net, teacher, condition, target, schedule,
            )
            bs = condition.shape[0]
            total += float(loss.item()) * bs
            seen += bs
            if not disable_progress:
                bar.set_postfix(loss=f"{total / max(seen, 1):.4f}", refresh=False)
    bar.close()
    return total / max(seen, 1)


def train_consistency_distill(
    data_dir: str | Path,
    output_dir: str | Path,
    stage: Literal["vol", "ret"],
    distill_config: ConsistencyDistillConfig,
    student_config: TwoStageFMModelConfig | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Distill a Consistency student from a trained FM teacher."""

    if stage not in ("vol", "ret"):
        raise ValueError("stage must be 'vol' or 'ret'")
    set_seed(distill_config.seed)

    device = resolve_device(distill_config.device)
    teacher, teacher_ckpt = load_model_from_checkpoint(
        distill_config.teacher_checkpoint, map_location=device,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    teacher_stage = teacher_ckpt.get("stage", "joint")
    if teacher_stage != stage:
        raise ValueError(
            f"teacher checkpoint stage='{teacher_stage}' does not match requested stage='{stage}'"
        )
    num_actions = int(teacher_ckpt.get("num_actions", load_num_actions(data_dir)))
    normalization = teacher_ckpt.get("normalization") or load_normalization(data_dir)

    if stage == "vol":
        datasets = build_vol_datasets(data_dir, normalization, num_actions)
        expected_state, expected_cond = 1, 1 + num_actions
    else:
        datasets = build_ret_datasets(data_dir, normalization, num_actions)
        expected_state, expected_cond = 1, 3 + num_actions

    if student_config is None:
        student_config = TwoStageFMModelConfig(
            state_dim=teacher.state_dim,
            condition_dim=teacher.condition_dim,
            hidden_dim=teacher.hidden_dim,
            time_embedding_dim=teacher.time_embedding.embedding_dim,
            num_blocks=len(teacher.blocks),
        )

    if student_config.state_dim != expected_state or student_config.condition_dim != expected_cond:
        raise ValueError(
            f"{stage}-stage student must have state_dim={expected_state}, "
            f"condition_dim={expected_cond}; got "
            f"({student_config.state_dim}, {student_config.condition_dim})"
        )

    student = ConsistencyStudent(**asdict(student_config)).to(device)
    warm_copied = 0
    if distill_config.warm_start:
        warm_copied = warm_start_consistency_from_fm(student, teacher)
    target_net = copy.deepcopy(student)
    for p in target_net.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=distill_config.lr,
        weight_decay=distill_config.weight_decay,
    )

    train_loader = build_dataloader(
        datasets["train"], batch_size=distill_config.batch_size, shuffle=True,
        num_workers=distill_config.num_workers, device=device,
    )
    val_loader = build_dataloader(
        datasets["val"], batch_size=distill_config.batch_size, shuffle=False,
        num_workers=distill_config.num_workers, device=device,
    )

    schedule = _schedule(
        distill_config.n_discretization, distill_config.time_eps,
        device=device, dtype=torch.float32,
    )

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix=f"cd_{stage}_distill")
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    config_blob = {
        "run_dir": str(run_dir.resolve()),
        "stage": stage,
        "num_actions": num_actions,
        "student_config": asdict(student_config),
        "distill_config": asdict(distill_config),
        "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
        "warm_started_params": warm_copied,
        "normalization": normalization,
        "data_dir": str(Path(data_dir).resolve()),
        "schedule": schedule.cpu().tolist(),
    }
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    disable_progress = not distill_config.progress
    train_batches = _effective_num_batches(train_loader, distill_config.max_train_batches)
    val_batches = _effective_num_batches(val_loader, distill_config.max_val_batches)
    n_params = sum(p.numel() for p in student.parameters())
    if distill_config.progress:
        header = (
            f"[finflow] cd_distill stage={stage} | run={run_dir.name} | device={device} | "
            f"params={n_params/1e3:.1f}k | warm_start={warm_copied} tensors | "
            f"N={distill_config.n_discretization} ema={distill_config.ema_decay} | "
            f"train={len(datasets['train'])} ({train_batches} batch x {distill_config.batch_size}) | "
            f"val={len(datasets['val'])} ({val_batches} batch) | epochs={distill_config.epochs}"
        )
        print(header, file=sys.stderr, flush=True)

    best_val = float("inf")
    history: list[dict[str, Any]] = []
    run_start = time.monotonic()
    global_step = 0

    for epoch in range(1, distill_config.epochs + 1):
        epoch_start = time.monotonic()
        desc_train = f"epoch {epoch:>3}/{distill_config.epochs} train"
        desc_val = f"epoch {epoch:>3}/{distill_config.epochs} val  "
        train_loss = _train_one_epoch_consistency(
            student, target_net, teacher, train_loader, optimizer, schedule, device,
            ema_decay=distill_config.ema_decay,
            grad_clip_norm=distill_config.grad_clip_norm,
            max_batches=distill_config.max_train_batches,
            disable_progress=disable_progress,
            progress_min_interval=distill_config.progress_min_interval,
            desc=desc_train,
        )
        val_loss = _evaluate_consistency(
            student, target_net, teacher, val_loader, schedule, device,
            max_batches=distill_config.max_val_batches,
            disable_progress=disable_progress,
            progress_min_interval=distill_config.progress_min_interval,
            desc=desc_val,
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
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        is_best = val_loss < best_val
        save_checkpoint(
            ckpt_dir / "last.pt", target_net, optimizer,
            epoch=epoch, global_step=global_step, best_val_loss=best_val,
            model_config=asdict(student_config),
            train_config=asdict(distill_config),
            normalization=normalization, stage=f"cd_{stage}", num_actions=num_actions,
            extra={
                "train_loss": train_loss, "val_loss": val_loss,
                "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
                "warm_started_params": warm_copied,
                "kind": "consistency",
                "model_state_kind": "ema",
                "ema_decay": distill_config.ema_decay,
                "n_discretization": distill_config.n_discretization,
            },
        )
        if is_best:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt", target_net, optimizer,
                epoch=epoch, global_step=global_step, best_val_loss=best_val,
                model_config=asdict(student_config),
                train_config=asdict(distill_config),
                normalization=normalization, stage=f"cd_{stage}", num_actions=num_actions,
                extra={
                    "train_loss": train_loss, "val_loss": val_loss,
                    "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
                    "warm_started_params": warm_copied,
                    "kind": "consistency",
                    "model_state_kind": "ema",
                    "ema_decay": distill_config.ema_decay,
                    "n_discretization": distill_config.n_discretization,
                },
            )

        if distill_config.progress:
            elapsed = time.monotonic() - run_start
            eta_s = (elapsed / epoch) * (distill_config.epochs - epoch)
            print(
                f"  epoch {epoch:>3}/{distill_config.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"best={best_val:.4f}{' *' if is_best else '  '} | "
                f"epoch={_fmt_time(epoch_time)} | elapsed={_fmt_time(elapsed)} | "
                f"eta={_fmt_time(eta_s)}",
                file=sys.stderr, flush=True,
            )

    summary = {
        "run_dir": str(run_dir),
        "stage": f"cd_{stage}",
        "num_actions": num_actions,
        "checkpoints": {
            "best": str(ckpt_dir / "best.pt"),
            "last": str(ckpt_dir / "last.pt"),
        },
        "best_val_loss": best_val,
        "history": history,
        "device": str(device),
        "teacher_checkpoint": str(Path(distill_config.teacher_checkpoint).resolve()),
        "warm_started_params": warm_copied,
        "total_time_s": time.monotonic() - run_start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
