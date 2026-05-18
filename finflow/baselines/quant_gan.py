"""Minimal Quant GAN baseline (Wiese et al. 2020).

A small Temporal Convolutional Network (TCN) generator and discriminator
trained with LSGAN loss on Heston log-return sequences. Kept intentionally
simple so it serves as a comparison point against the V3 FM / MF pipeline,
not a SOTA replication.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from finflow.training import (
    _effective_num_batches,
    _fmt_time,
    _iterate_batches,
    _make_progress,
    build_run_dir,
    load_metadata,
    load_normalization,
    resolve_device,
    set_seed,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HestonLogReturnSequenceDataset(Dataset[torch.Tensor]):
    """Yield normalized log-return sequences ``[n_steps]`` from a split ``.npz``."""

    def __init__(
        self,
        npz_path: str | Path,
        *,
        return_mean: float = 0.0,
        return_std: float = 1.0,
    ) -> None:
        self.npz_path = Path(npz_path)
        if return_std <= 0:
            raise ValueError("return_std must be positive")
        npz = np.load(self.npz_path)
        if "log_returns" not in npz.files:
            raise ValueError(f"{self.npz_path} missing 'log_returns'")
        self.returns = np.asarray(npz["log_returns"], dtype=np.float32)
        npz.close()
        self.return_mean = float(return_mean)
        self.return_std = float(return_std)

    def __len__(self) -> int:
        return int(self.returns.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        r = self.returns[index]
        r_norm = (r - self.return_mean) / self.return_std
        return torch.from_numpy(np.ascontiguousarray(r_norm)).float()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class _DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
        )
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(x + self.net(x))


class QuantGANGenerator(nn.Module):
    """1D TCN generator: ``z [B, latent_dim, L] -> returns [B, 1, L]``."""

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_channels: int = 32,
        num_blocks: int = 5,
        kernel_size: int = 3,
        seq_len: int = 252,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_proj = nn.Conv1d(latent_dim, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _DilatedConvBlock(hidden_channels, kernel_size, dilation=2 ** i)
                for i in range(num_blocks)
            ]
        )
        self.output = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        return self.output(h)


class QuantGANDiscriminator(nn.Module):
    """1D TCN discriminator: ``returns [B, 1, L] -> realism score [B]``."""

    def __init__(
        self,
        hidden_channels: int = 32,
        num_blocks: int = 5,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(1, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _DilatedConvBlock(hidden_channels, kernel_size, dilation=2 ** i)
                for i in range(num_blocks)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.pool(h).squeeze(-1)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantGANConfig:
    latent_dim: int = 8
    hidden_channels: int = 32
    num_blocks: int = 5
    kernel_size: int = 3
    seq_len: int = 252


@dataclass(frozen=True)
class QuantGANTrainConfig:
    batch_size: int = 128
    epochs: int = 20
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    d_steps_per_g: int = 1
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    seed: int = 1234
    device: str = "auto"
    max_train_batches: int | None = None
    progress: bool = True
    progress_min_interval: float = 0.2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _ls_loss(value: torch.Tensor, target: float) -> torch.Tensor:
    return 0.5 * ((value - target) ** 2).mean()


def train_quant_gan(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model_config: QuantGANConfig | None = None,
    train_config: QuantGANTrainConfig | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train a Quant GAN baseline on Heston log-return sequences."""

    model_config = model_config or QuantGANConfig()
    train_config = train_config or QuantGANTrainConfig()
    set_seed(train_config.seed)

    device = resolve_device(train_config.device)
    metadata = load_metadata(data_dir)
    normalization = load_normalization(data_dir)
    n_steps_meta = int(metadata.get("n_steps", model_config.seq_len))
    if n_steps_meta != model_config.seq_len:
        raise ValueError(
            f"metadata n_steps={n_steps_meta} does not match config seq_len={model_config.seq_len}"
        )

    data_dir = Path(data_dir)
    train_dataset = HestonLogReturnSequenceDataset(
        data_dir / "train.npz",
        return_mean=normalization["return_mean"],
        return_std=normalization["return_std"],
    )
    val_dataset = HestonLogReturnSequenceDataset(
        data_dir / "val.npz",
        return_mean=normalization["return_mean"],
        return_std=normalization["return_std"],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=True,
        num_workers=train_config.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size, shuffle=False,
        num_workers=train_config.num_workers, pin_memory=device.type == "cuda",
    )

    generator = QuantGANGenerator(
        latent_dim=model_config.latent_dim,
        hidden_channels=model_config.hidden_channels,
        num_blocks=model_config.num_blocks,
        kernel_size=model_config.kernel_size,
        seq_len=model_config.seq_len,
    ).to(device)
    discriminator = QuantGANDiscriminator(
        hidden_channels=model_config.hidden_channels,
        num_blocks=model_config.num_blocks,
        kernel_size=model_config.kernel_size,
    ).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=train_config.lr_g, betas=train_config.betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=train_config.lr_d, betas=train_config.betas)

    run_dir = build_run_dir(output_dir, run_name=run_name, prefix="quant_gan")
    ckpt_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"

    config_blob = {
        "run_dir": str(run_dir.resolve()),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "normalization": normalization,
        "data_dir": str(Path(data_dir).resolve()),
    }
    (run_dir / "config.json").write_text(json.dumps(config_blob, indent=2), encoding="utf-8")

    disable_progress = not train_config.progress
    train_batches = _effective_num_batches(train_loader, train_config.max_train_batches)
    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())
    if train_config.progress:
        header = (
            f"[finflow] quant_gan | run={run_dir.name} | device={device} | "
            f"params_G={n_params_g/1e3:.1f}k params_D={n_params_d/1e3:.1f}k | "
            f"train={len(train_dataset)} samples ({train_batches} batch x {train_config.batch_size}) | "
            f"epochs={train_config.epochs}"
        )
        print(header, file=sys.stderr, flush=True)

    history: list[dict[str, Any]] = []
    run_start = time.monotonic()
    global_step = 0
    best_val = float("inf")

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.monotonic()
        bar = _make_progress(
            _iterate_batches(train_loader, train_config.max_train_batches),
            total=train_batches,
            desc=f"epoch {epoch:>3}/{train_config.epochs} train",
            disable=disable_progress,
            min_interval=train_config.progress_min_interval,
        )
        running_g, running_d, running_n = 0.0, 0.0, 0
        generator.train(); discriminator.train()
        for batch in bar:
            real = batch.to(device).unsqueeze(1)  # [B, 1, L]
            B = real.shape[0]

            # Train D
            for _ in range(train_config.d_steps_per_g):
                z = torch.randn(B, model_config.latent_dim, model_config.seq_len, device=device)
                with torch.no_grad():
                    fake = generator(z)
                d_real = discriminator(real)
                d_fake = discriminator(fake)
                loss_d = _ls_loss(d_real, 1.0) + _ls_loss(d_fake, 0.0)
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if train_config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), train_config.grad_clip_norm)
                opt_d.step()

            # Train G
            z = torch.randn(B, model_config.latent_dim, model_config.seq_len, device=device)
            fake = generator(z)
            d_fake = discriminator(fake)
            loss_g = _ls_loss(d_fake, 1.0)
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if train_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), train_config.grad_clip_norm)
            opt_g.step()

            running_g += float(loss_g.item()) * B
            running_d += float(loss_d.item()) * B
            running_n += B
            if not disable_progress:
                bar.set_postfix(
                    g=f"{running_g / max(running_n, 1):.4f}",
                    d=f"{running_d / max(running_n, 1):.4f}",
                    refresh=False,
                )
        bar.close()
        train_g = running_g / max(running_n, 1)
        train_d = running_d / max(running_n, 1)

        # Validation: discriminator loss on real validation batch + fake.
        generator.eval(); discriminator.eval()
        v_running, v_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                real = batch.to(device).unsqueeze(1)
                B = real.shape[0]
                z = torch.randn(B, model_config.latent_dim, model_config.seq_len, device=device)
                fake = generator(z)
                d_real = discriminator(real)
                d_fake = discriminator(fake)
                loss = _ls_loss(d_real, 1.0) + _ls_loss(d_fake, 0.0)
                v_running += float(loss.item()) * B
                v_n += B
        val_d = v_running / max(v_n, 1)
        epoch_time = time.monotonic() - epoch_start
        global_step += train_batches

        record = {
            "epoch": epoch,
            "train_g_loss": train_g,
            "train_d_loss": train_d,
            "val_d_loss": val_d,
            "epoch_time_s": epoch_time,
            "global_step": global_step,
        }
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        is_best = val_d < best_val
        if is_best:
            best_val = val_d
        ckpt = {
            "generator_state": {n: t.detach().cpu() for n, t in generator.state_dict().items()},
            "discriminator_state": {n: t.detach().cpu() for n, t in discriminator.state_dict().items()},
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "normalization": normalization,
            "epoch": epoch,
            "global_step": global_step,
            "best_val_d_loss": best_val,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if is_best:
            torch.save(ckpt, ckpt_dir / "best.pt")

        if train_config.progress:
            elapsed = time.monotonic() - run_start
            eta_s = (elapsed / epoch) * (train_config.epochs - epoch)
            print(
                f"  epoch {epoch:>3}/{train_config.epochs} | "
                f"G={train_g:.4f} D={train_d:.4f} | val_D={val_d:.4f}{' *' if is_best else '  '} | "
                f"epoch={_fmt_time(epoch_time)} | elapsed={_fmt_time(elapsed)} | "
                f"eta={_fmt_time(eta_s)}",
                file=sys.stderr, flush=True,
            )

    summary = {
        "run_dir": str(run_dir),
        "checkpoints": {
            "best": str(ckpt_dir / "best.pt"),
            "last": str(ckpt_dir / "last.pt"),
        },
        "best_val_d_loss": best_val,
        "history": history,
        "device": str(device),
        "total_time_s": time.monotonic() - run_start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_quant_gan_paths(
    generator: QuantGANGenerator,
    n_paths: int,
    *,
    s0: float = 100.0,
    return_mean: float = 0.0,
    return_std: float = 1.0,
    device: str | torch.device = "cpu",
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate ``n_paths`` log-return + price sequences from a trained Quant GAN."""

    device = torch.device(device) if isinstance(device, str) else device
    generator = generator.to(device).eval()
    rng = torch.Generator(device="cpu")
    if seed is not None:
        rng.manual_seed(seed)
    z = torch.randn(
        n_paths, generator.latent_dim, generator.seq_len,
        generator=rng,
    ).to(device)
    fake_norm = generator(z).squeeze(1)  # [B, L]
    fake = fake_norm.cpu().numpy() * return_std + return_mean
    s = np.empty((n_paths, generator.seq_len + 1), dtype=np.float32)
    s[:, 0] = s0
    log_s = np.log(s0) + np.cumsum(fake, axis=1)
    s[:, 1:] = np.exp(log_s).astype(np.float32)
    return {"log_returns": fake.astype(np.float32), "s_paths": s}


def load_quant_gan_generator(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[QuantGANGenerator, dict[str, Any]]:
    ckpt = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    cfg = ckpt["model_config"]
    generator = QuantGANGenerator(
        latent_dim=int(cfg["latent_dim"]),
        hidden_channels=int(cfg["hidden_channels"]),
        num_blocks=int(cfg["num_blocks"]),
        kernel_size=int(cfg["kernel_size"]),
        seq_len=int(cfg["seq_len"]),
    )
    generator.load_state_dict(ckpt["generator_state"])
    generator.to(map_location).eval()
    return generator, ckpt
