"""Path-level fine-tuning for two-stage FM teachers.

This module keeps the existing one-step FM/LWFM teachers intact and adds a
separate second-stage objective over full generated return paths. It borrows the
useful part of the Quant GAN baseline: a TCN critic trained on whole sequences,
with optional Lambert-W Gaussianization of standardized returns.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from finflow.baselines.quant_gan import QuantGANDiscriminator
from finflow.models import TransitionFM
from finflow.training import (
    build_run_dir,
    load_metadata,
    load_model_from_checkpoint,
    load_normalization,
    resolve_device,
    save_checkpoint,
    set_seed,
)
from finflow.transforms import (
    inverse_lambert_w_transform_torch,
    lambert_w_transform_torch,
)


@dataclass(frozen=True)
class PathwiseTeacherFineTuneConfig:
    """Hyperparameters for path-level teacher fine-tuning."""

    batch_size: int = 128
    epochs: int = 3
    steps_per_epoch: int = 100
    n_steps: int = 252
    fm_n_steps: int = 8
    lr_teacher: float = 1e-5
    lr_critic: float = 2e-4
    critic_steps: int = 3
    gradient_penalty_weight: float = 10.0
    transform_delta: float = 0.1
    moment_weight: float = 1.0
    terminal_weight: float = 1.0
    abs_sum_weight: float = 0.25
    kurtosis_weight: float = 0.1
    anchor_weight: float = 1e-6
    train_vol: bool = True
    train_ret: bool = True
    critic_hidden_channels: int = 32
    critic_num_blocks: int = 5
    critic_kernel_size: int = 3
    initial_v: float = 0.04
    initial_s: float = 100.0
    initial_r_prev: float = 0.0
    seed: int = 1234
    device: str = "auto"
    progress: bool = True
    compile_models: bool = False
    compile_mode: str = "reduce-overhead"


class ReturnPathBatcher:
    """Fast random batches of real return paths and their regime actions."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        n_steps: int,
        normalization: dict[str, float],
        device: torch.device,
    ) -> None:
        data = np.load(Path(data_dir) / "train.npz")
        returns = np.asarray(data["log_returns"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64) if "actions" in data.files else None
        data.close()
        if returns.ndim != 2:
            raise ValueError("train.npz log_returns must have shape [n_paths, n_steps]")
        if n_steps <= 0 or n_steps > returns.shape[1]:
            raise ValueError(f"n_steps must be in [1, {returns.shape[1]}]")
        if actions is None:
            actions = np.zeros((returns.shape[0], returns.shape[1]), dtype=np.int64)
        if actions.shape != returns.shape:
            raise ValueError("actions must have the same [n_paths, n_steps] shape as log_returns")

        self.returns = torch.from_numpy(np.ascontiguousarray(returns[:, :n_steps])).to(device)
        self.actions = torch.from_numpy(np.ascontiguousarray(actions[:, :n_steps])).to(device)
        self.return_mean = float(normalization["return_mean"])
        self.return_std = float(normalization["return_std"])
        self.n_paths = int(self.returns.shape[0])
        self.n_steps = int(n_steps)
        self.device = device

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        idx = torch.randint(self.n_paths, (batch_size,), device=self.device)
        real = self.returns.index_select(0, idx)
        actions = self.actions.index_select(0, idx)
        real_norm = (real - self.return_mean) / self.return_std
        return real_norm, actions


def _onehot(actions: torch.Tensor, num_actions: int) -> torch.Tensor:
    return F.one_hot(actions.long(), num_classes=num_actions).to(actions.device).float()


def _onehot_sequence(actions: torch.Tensor, num_actions: int, dtype: torch.dtype) -> torch.Tensor:
    return F.one_hot(actions.long(), num_classes=num_actions).to(device=actions.device, dtype=dtype)


def _fm_tau_grid(n_steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(n_steps, device=device, dtype=dtype) / float(n_steps)


def _fm_sample_with_grad(
    model: nn.Module,
    condition: torch.Tensor,
    *,
    noise: torch.Tensor,
    n_steps: int,
    num_actions: int,
    cfg_w: float = 0.0,
    tau_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    unconditional = None
    if cfg_w > 0.0:
        unconditional = condition.clone()
        unconditional[:, -num_actions:] = 0.0
    x = noise
    dt = 1.0 / n_steps
    batch = condition.shape[0]
    for step in range(n_steps):
        if tau_grid is None:
            tau = torch.full((batch,), step * dt, device=condition.device, dtype=condition.dtype)
        else:
            tau = tau_grid[step].expand(batch)
        velocity = model(x_tau=x, tau=tau, condition=condition)
        if unconditional is not None:
            velocity_uncond = model(x_tau=x, tau=tau, condition=unconditional)
            velocity = (1.0 + cfg_w) * velocity - cfg_w * velocity_uncond
        x = x + dt * velocity
    return x


def _differentiable_rollout_norm(
    vol_model: nn.Module,
    ret_model: nn.Module,
    actions: torch.Tensor,
    *,
    normalization: dict[str, float],
    num_actions: int,
    initial_v: float,
    initial_r_prev: float,
    fm_n_steps: int,
    vol_lambert_w_delta: float,
    cfg_w: float = 0.0,
    action_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll out normalized returns with gradients flowing to FM models."""

    batch, n_steps = actions.shape
    device = actions.device
    dtype = next(ret_model.parameters()).dtype
    log_v_mean = float(normalization["log_v_mean"])
    log_v_std = float(normalization["log_v_std"])
    return_mean = float(normalization["return_mean"])
    return_std = float(normalization["return_std"])
    log_v0 = (np.log(initial_v) - log_v_mean) / log_v_std
    r0 = (initial_r_prev - return_mean) / return_std
    log_v_t = torch.full((batch, 1), float(log_v0), device=device, dtype=dtype)
    r_prev_t = torch.full((batch, 1), float(r0), device=device, dtype=dtype)
    if action_features is None:
        action_features = _onehot_sequence(actions, num_actions, dtype)
    tau_grid = _fm_tau_grid(fm_n_steps, device=device, dtype=dtype)
    out: list[torch.Tensor] = []

    for step in range(n_steps):
        a_onehot = action_features[:, step, :]
        vol_cond = torch.cat([log_v_t, a_onehot], dim=-1)
        z_vol = torch.randn(batch, 1, device=device, dtype=dtype)
        log_v_next = _fm_sample_with_grad(
            vol_model, vol_cond, noise=z_vol, n_steps=fm_n_steps,
            num_actions=num_actions, cfg_w=cfg_w, tau_grid=tau_grid,
        )
        if vol_lambert_w_delta > 0.0:
            log_v_next = inverse_lambert_w_transform_torch(
                log_v_next, delta=vol_lambert_w_delta,
            )

        ret_cond = torch.cat([log_v_next, log_v_t, r_prev_t, a_onehot], dim=-1)
        z_ret = torch.randn(batch, 1, device=device, dtype=dtype)
        r_next = _fm_sample_with_grad(
            ret_model, ret_cond, noise=z_ret, n_steps=fm_n_steps,
            num_actions=num_actions, cfg_w=cfg_w, tau_grid=tau_grid,
        )
        out.append(r_next)
        log_v_t = log_v_next
        r_prev_t = r_next

    return torch.cat(out, dim=1)


def _gradient_penalty(
    critic: QuantGANDiscriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    batch = real.shape[0]
    alpha = torch.rand(batch, 1, 1, device=real.device, dtype=real.dtype)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated)
    grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(batch, -1)
    return (grad.norm(2, dim=1) - 1.0).square().mean()


def _critic_domain(returns_norm: torch.Tensor, delta: float) -> torch.Tensor:
    transformed = lambert_w_transform_torch(returns_norm, delta=delta)
    return transformed.unsqueeze(1)


def _std(x: torch.Tensor) -> torch.Tensor:
    return x.std(unbiased=False).clamp_min(1e-6)


def _kurtosis(x: torch.Tensor) -> torch.Tensor:
    centered = x - x.mean()
    std = _std(centered)
    z = centered / std
    return z.pow(4).mean()


def _path_moment_loss(fake_norm: torch.Tensor, real_norm: torch.Tensor, cfg: PathwiseTeacherFineTuneConfig) -> torch.Tensor:
    loss = fake_norm.new_tensor(0.0)
    if cfg.moment_weight > 0.0:
        loss = loss + cfg.moment_weight * (
            (fake_norm.mean() - real_norm.mean()).square()
            + (_std(fake_norm) - _std(real_norm)).square()
        )
    fake_terminal = fake_norm.sum(dim=1)
    real_terminal = real_norm.sum(dim=1)
    if cfg.terminal_weight > 0.0:
        loss = loss + cfg.terminal_weight * (
            (fake_terminal.mean() - real_terminal.mean()).square()
            + (_std(fake_terminal) - _std(real_terminal)).square()
        )
    if cfg.abs_sum_weight > 0.0:
        fake_abs = fake_norm.abs().sum(dim=1)
        real_abs = real_norm.abs().sum(dim=1)
        loss = loss + cfg.abs_sum_weight * (
            (fake_abs.mean() - real_abs.mean()).square()
            + (_std(fake_abs) - _std(real_abs)).square()
        )
    if cfg.kurtosis_weight > 0.0:
        loss = loss + cfg.kurtosis_weight * (_kurtosis(fake_norm) - _kurtosis(real_norm)).square()
    return loss


def _set_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(trainable)


def _anchor_loss(models: list[nn.Module], anchors: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for mi, model in enumerate(models):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ref = anchors[f"{mi}:{name}"]
            value = (param - ref).square().mean()
            total = value if total is None else total + value
    if total is None:
        raise ValueError("no trainable parameters for anchor loss")
    return total


def _maybe_compile_model(model: nn.Module, *, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return compile_fn(model, mode=mode)


def _save_finetuned_checkpoint(
    path: Path,
    model: TransitionFM,
    optimizer: torch.optim.Optimizer,
    *,
    source_checkpoint: dict[str, Any],
    stage: str,
    num_actions: int,
    config: PathwiseTeacherFineTuneConfig,
    epoch: int,
    global_step: int,
    score: float,
    extra: dict[str, Any],
) -> None:
    base_extra = dict(source_checkpoint.get("extra", {}))
    base_extra.update(extra)
    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=epoch,
        global_step=global_step,
        best_val_loss=score,
        model_config=source_checkpoint["model_config"],
        train_config={"pathwise_finetune": asdict(config)},
        normalization=source_checkpoint["normalization"],
        stage=stage,
        num_actions=num_actions,
        extra=base_extra,
    )


def train_pathwise_teacher_finetune(
    *,
    vol_checkpoint: str | Path,
    ret_checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    run_name: str | None = None,
    config: PathwiseTeacherFineTuneConfig | None = None,
) -> dict[str, Any]:
    """Fine-tune FM teacher checkpoints with a QGAN-style path critic.

    The input checkpoints are read-only. New vol/ret checkpoints are written
    under ``output_dir/run_name/checkpoints``.
    """

    config = config or PathwiseTeacherFineTuneConfig()
    if not config.train_vol and not config.train_ret:
        raise ValueError("at least one of train_vol/train_ret must be true")
    if config.transform_delta < 0:
        raise ValueError("transform_delta must be non-negative")
    set_seed(config.seed)
    device = resolve_device(config.device)

    vol_model, vol_ckpt = load_model_from_checkpoint(vol_checkpoint, map_location=device)
    ret_model, ret_ckpt = load_model_from_checkpoint(ret_checkpoint, map_location=device)
    if vol_ckpt.get("stage") != "vol":
        raise ValueError(f"vol checkpoint stage must be 'vol', got {vol_ckpt.get('stage')!r}")
    if ret_ckpt.get("stage") != "ret":
        raise ValueError(f"ret checkpoint stage must be 'ret', got {ret_ckpt.get('stage')!r}")
    num_actions = int(vol_ckpt.get("num_actions", 1))
    if int(ret_ckpt.get("num_actions", 1)) != num_actions:
        raise ValueError("vol and ret checkpoints must have the same num_actions")
    normalization = vol_ckpt.get("normalization") or ret_ckpt.get("normalization") or load_normalization(data_dir)
    metadata = load_metadata(data_dir)
    if config.n_steps > int(metadata.get("n_steps", config.n_steps)):
        raise ValueError("config.n_steps exceeds the dataset path length")

    _set_trainable(vol_model, config.train_vol)
    _set_trainable(ret_model, config.train_ret)
    vol_model.train(config.train_vol)
    ret_model.train(config.train_ret)
    critic = QuantGANDiscriminator(
        hidden_channels=config.critic_hidden_channels,
        num_blocks=config.critic_num_blocks,
        kernel_size=config.critic_kernel_size,
    ).to(device)
    vol_forward = _maybe_compile_model(vol_model, enabled=config.compile_models, mode=config.compile_mode)
    ret_forward = _maybe_compile_model(ret_model, enabled=config.compile_models, mode=config.compile_mode)
    # Keep the WGAN-GP critic in eager mode: the gradient penalty uses double
    # backward, which is not supported by AOTAutograd in all PyTorch builds.
    critic_forward = critic

    teacher_params = [p for p in list(vol_model.parameters()) + list(ret_model.parameters()) if p.requires_grad]
    teacher_optimizer = torch.optim.AdamW(teacher_params, lr=config.lr_teacher)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config.lr_critic, betas=(0.0, 0.9))
    batcher = ReturnPathBatcher(data_dir, n_steps=config.n_steps, normalization=normalization, device=device)
    vol_lambert_w_delta = float(vol_ckpt.get("extra", {}).get("lambert_w_delta", 0.0) or 0.0)
    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="pathwise_teacher")
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    models = [vol_model, ret_model]
    anchors = {
        f"{mi}:{name}": param.detach().clone()
        for mi, model in enumerate(models)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    cfg_blob = {
        "run_dir": str(run_dir.resolve()),
        "vol_checkpoint": str(Path(vol_checkpoint).resolve()),
        "ret_checkpoint": str(Path(ret_checkpoint).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "config": asdict(config),
        "num_actions": num_actions,
        "normalization": normalization,
        "vol_lambert_w_delta": vol_lambert_w_delta,
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_blob, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    global_step = 0
    start_time = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        epoch_metrics: dict[str, float] = {
            "critic_loss": 0.0,
            "generator_loss": 0.0,
            "adv_loss": 0.0,
            "moment_loss": 0.0,
            "anchor_loss": 0.0,
            "wasserstein_estimate": 0.0,
        }
        for _ in range(config.steps_per_epoch):
            for _critic_step in range(config.critic_steps):
                real_norm, actions = batcher.sample(config.batch_size)
                action_features = _onehot_sequence(actions, num_actions, next(ret_model.parameters()).dtype)
                with torch.no_grad():
                    fake_norm = _differentiable_rollout_norm(
                        vol_forward, ret_forward, actions,
                        normalization=normalization,
                        num_actions=num_actions,
                        initial_v=config.initial_v,
                        initial_r_prev=config.initial_r_prev,
                        fm_n_steps=config.fm_n_steps,
                        vol_lambert_w_delta=vol_lambert_w_delta,
                        action_features=action_features,
                    )
                real_c = _critic_domain(real_norm, config.transform_delta)
                fake_c = _critic_domain(fake_norm, config.transform_delta)
                d_real = critic_forward(real_c)
                d_fake = critic_forward(fake_c)
                gp = _gradient_penalty(critic_forward, real_c, fake_c)
                critic_loss = d_fake.mean() - d_real.mean() + config.gradient_penalty_weight * gp
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_optimizer.step()

            for param in critic.parameters():
                param.requires_grad_(False)
            real_norm, actions = batcher.sample(config.batch_size)
            action_features = _onehot_sequence(actions, num_actions, next(ret_model.parameters()).dtype)
            fake_norm = _differentiable_rollout_norm(
                vol_forward, ret_forward, actions,
                normalization=normalization,
                num_actions=num_actions,
                initial_v=config.initial_v,
                initial_r_prev=config.initial_r_prev,
                fm_n_steps=config.fm_n_steps,
                vol_lambert_w_delta=vol_lambert_w_delta,
                action_features=action_features,
            )
            fake_c = _critic_domain(fake_norm, config.transform_delta)
            adv_loss = -critic_forward(fake_c).mean()
            moment_loss = _path_moment_loss(fake_norm, real_norm, config)
            anchor = _anchor_loss(models, anchors) if config.anchor_weight > 0.0 else fake_norm.new_tensor(0.0)
            generator_loss = adv_loss + moment_loss + config.anchor_weight * anchor
            teacher_optimizer.zero_grad(set_to_none=True)
            generator_loss.backward()
            teacher_optimizer.step()
            for param in critic.parameters():
                param.requires_grad_(True)

            global_step += 1
            epoch_metrics["critic_loss"] += float(critic_loss.detach().cpu())
            epoch_metrics["generator_loss"] += float(generator_loss.detach().cpu())
            epoch_metrics["adv_loss"] += float(adv_loss.detach().cpu())
            epoch_metrics["moment_loss"] += float(moment_loss.detach().cpu())
            epoch_metrics["anchor_loss"] += float(anchor.detach().cpu())
            epoch_metrics["wasserstein_estimate"] += float((d_real.mean() - d_fake.mean()).detach().cpu())

        denom = max(config.steps_per_epoch, 1)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            **{key: value / denom for key, value in epoch_metrics.items()},
            "elapsed_s": time.monotonic() - start_time,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        score = float(record["generator_loss"])
        is_best = score < best_score
        if is_best:
            best_score = score
        for name, model, source, stage in (
            ("vol", vol_model, vol_ckpt, "vol"),
            ("ret", ret_model, ret_ckpt, "ret"),
        ):
            extra = {
                "pathwise_finetuned": True,
                "source_checkpoint": str(Path(vol_checkpoint if name == "vol" else ret_checkpoint).resolve()),
                "pathwise_score": score,
                "kind": "fm",
            }
            _save_finetuned_checkpoint(
                ckpt_dir / f"{name}_last.pt",
                model,
                teacher_optimizer,
                source_checkpoint=source,
                stage=stage,
                num_actions=num_actions,
                config=config,
                epoch=epoch,
                global_step=global_step,
                score=score,
                extra=extra,
            )
            if is_best:
                _save_finetuned_checkpoint(
                    ckpt_dir / f"{name}_best.pt",
                    model,
                    teacher_optimizer,
                    source_checkpoint=source,
                    stage=stage,
                    num_actions=num_actions,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    score=score,
                    extra=extra,
                )

        if config.progress:
            print(
                f"[pathwise] epoch {epoch}/{config.epochs} "
                f"G={record['generator_loss']:.4f} D={record['critic_loss']:.4f} "
                f"Mom={record['moment_loss']:.4f} W={record['wasserstein_estimate']:.4f}",
                flush=True,
            )

    summary = {
        "run_dir": str(run_dir),
        "checkpoints": {
            "vol_best": str(ckpt_dir / "vol_best.pt"),
            "vol_last": str(ckpt_dir / "vol_last.pt"),
            "ret_best": str(ckpt_dir / "ret_best.pt"),
            "ret_last": str(ckpt_dir / "ret_last.pt"),
        },
        "best_score": best_score,
        "history": history,
        "device": str(device),
        "total_time_s": time.monotonic() - start_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
