# V3 P0 实施总结

> Date: 2026-05-19  
> Scope: [05_next_steps.md](05_next_steps.md) 中的 P0 工程修复。  
> Status: P0 工程阶段已完成。medium smoke 已通过，P0 中暴露出的两个主要失效模式已经修复或缓解。

---

## 1. P0 目标

P0 的目标是在进入 P1 方法论优化前，先把 V3 四路对比管线变成技术上可用、可复现、可评测的状态：

- 用 iCT 风格训练救活 Consistency Distillation；
- 把 Quant GAN 从 collapsed / drift baseline 修成有意义的对照组；
- 为 regime-switching 数据补 MC-oracle pricing reference；
- 增加一键全量评测入口；
- 用 smoke tests 捕获 rollout 和 pricing 层面的系统性故障。

P0 初始 smoke 又暴露了两个额外问题：

- CD loss 会下降，但 rollout 会爆炸。根因是 consistency direction 与本项目 FM 时间约定不一致。
- Quant GAN 不再价格全 0，但出现了系统性正向 drift。

这两个问题都已在 P0 阶段处理。

---

## 2. 已完成改动

### 2.1 Consistency Distillation

涉及文件：

- `finflow/distillation/consistency.py`
- `scripts/distill_consistency.py`
- `tests/test_consistency_distill.py`

完成内容：

- iCT-style discretization curriculum: `n_min -> n_max`
- Karras-style EMA schedule
- Pseudo-Huber loss
- lognormal time interval sampling
- per-epoch logging of `n_discretization`, `ema_decay`, `huber_c`, and `time_sampling`
- CLI flags for all new CD controls

关键 bug 修复：

本项目的 FM 时间约定是：

```text
t = 0: noise
t = 1: data
```

原 CD 实现沿用了相反的 consistency 方向，实际训练的是：

```python
student(x_next, t_next) -> target(x_curr, t_curr)
```

其中 `t_next > t_curr`，在本项目中 `t_next` 更接近 data。因此旧实现等价于让 cleaner point 匹配 noisier EMA target，表现为 loss 能下降但 rollout 不稳定。

修正后的方向是：

```python
target_val = target_net(x_next, t_next, condition)
pred = student(x_curr_hat, t_curr, condition)
```

也就是让 noisier student call 匹配 cleaner EMA target。

### 2.2 Quant GAN

涉及文件：

- `finflow/baselines/quant_gan.py`
- `finflow/baselines/__init__.py`
- `scripts/train_quant_gan.py`
- `scripts/sample_quant_gan.py`
- `tests/test_quant_gan.py`

完成内容：

- Lambert W preprocessing and inverse transform
- WGAN-GP training instead of LSGAN
- generator output head: `LayerNorm + learnable_scale * tanh + learnable_shift`
- sampling-time moment calibration, enabled by default
- optional `--no-calibrate-moments` for raw samples
- generator moment penalty during training:

```text
loss_g = -E[D(fake)] + lambda * moment_loss(fake, real)
```

The moment loss matches global mean/std in the transformed training domain.

结果：

- 旧的“价格全 0 / 所有路径低于 `S0`”问题已经消失。
- 后续暴露出的正向 drift 通过 sampling-time calibration 纠偏，并在训练端用 moment penalty 约束。

### 2.3 MC-Oracle Pricing

涉及文件：

- `finflow/eval/pricing.py`
- `finflow/eval/reports.py`
- `finflow/eval/__init__.py`
- `scripts/evaluate_rollout.py`
- `scripts/generate_mc_oracle.py`
- `tests/test_pricing_eval.py`
- `tests/test_eval_reports.py`
- `tests/test_mc_oracle_script.py`

完成内容：

- `pricing_rmse_vs_mc_oracle(...)`
- `build_full_report(..., oracle_s_paths=...)`
- `evaluate_rollout.py --mc-oracle`
- `scripts/generate_mc_oracle.py`

这样 regime-switching 数据也有了 pricing reference，不需要强行退化到不合理的 single-Heston Carr-Madan 对比。

### 2.4 一键评测脚本

File:

- `scripts/run_full_evaluation.sh`

完成内容：

- scans a rollout directory for:
  - `rollout_fm.npz`
  - `rollout_mf.npz`
  - `rollout_cd.npz`
  - `quant_gan_paths.npz`
- runs `evaluate_rollout.py` for each available model,
- writes a compact markdown table with Wasserstein and pricing metrics.

---

## 3. 验证

### 3.1 测试套件

最终测试结果：

```text
76 passed in 4.97s
```

额外检查：

```text
git diff --check: no whitespace errors
```

新增测试覆盖：

- CD iCT curriculum and EMA schedule,
- corrected CD consistency direction,
- Lambert W round trip,
- Quant GAN moment calibration,
- Quant GAN train/sample smoke,
- MC-oracle generation,
- MC-oracle pricing and report integration.

### 3.2 Medium Smoke 配置

主 smoke 目录：

```text
/tmp/finflow_p0_medium_smoke
```

修正后 CD smoke 目录：

```text
/tmp/finflow_p0_medium_smoke_fixed_cd
```

Quant GAN 校准评测目录：

```text
/tmp/finflow_p0_medium_smoke_qgan_calibrated
```

配置：

- device: CPU
- data: 256 train / 64 val / 64 test
- steps: 128
- regime switching: enabled
- MC oracle: 5000 paths
- FM teacher: 3 epochs
- CD iCT: 5 epochs, `N = 8 -> 64`
- Quant GAN: 5 epochs, WGAN-GP
- rollout: 256 paths
- pricing maturities: `0.1, 0.25, 0.5`

---

## 4. 关键结果

### 4.1 CD 修复

修复时间方向前，CD rollout 不稳定：

| Metric | CD before fix |
|---|---:|
| `S_T` min | 13.21 |
| `S_T` median | 73.93 |
| `S_T` max | 488.82 |
| return std | 0.06165 |
| `v` median | 1.457 |
| `v` max | 1.747 |

修复后：

| Metric | CD fixed |
|---|---:|
| `S_T` min | 68.11 |
| `S_T` median | 96.04 |
| `S_T` max | 126.51 |
| return std | 0.00884 |
| `v` median | 0.02445 |
| `v` max | 0.11098 |

CD training after the fix:

| Stage | best val loss |
|---|---:|
| CD vol | 0.000283 |
| CD ret | 0.000072 |

### 4.2 Quant GAN Drift 校准

medium checkpoint 的 raw Quant GAN 有明显正向 drift：

| Metric | Raw |
|---|---:|
| `S_T` mean | 117.63 |
| ATM positive paths | 256 / 256 |
| return mean | 0.001267 |
| return std | 0.008507 |

采样时 moment calibration 后：

| Metric | Calibrated |
|---|---:|
| `S_T` mean | 100.97 |
| ATM positive paths | 151 / 256 |
| return mean | 0.000071 |
| return std | 0.013104 |

定价影响：

| Metric | Raw | Calibrated |
|---|---:|---:|
| Total-return W1 | 0.1515 | 0.1064 |
| Pricing RMSE | 7.9791 | 2.5235 |
| Pricing MAPE | 0.9464 | 0.4229 |

### 4.3 最终 Medium Smoke 评测

使用修正后的 CD 和 calibrated Quant GAN：

| Model | Marginal W1 mean | Marginal W1 max | Total-return W1 | Pricing ref | Pricing RMSE | Pricing MAPE |
|---|---:|---:|---:|---|---:|---:|
| FM teacher | 0.0049 | 0.0079 | 0.0796 | MC oracle | 2.6683 | 0.3485 |
| Consistency | 0.0034 | 0.0058 | 0.0801 | MC oracle | 3.4828 | 0.4365 |
| Quant GAN calibrated | 0.0040 | 0.0058 | 0.1064 | MC oracle | 2.5235 | 0.4229 |

解读：

- CD no longer explodes and is close to FM teacher in distribution metrics.
- Quant GAN is now a usable baseline rather than a collapsed model.
- Pricing metrics are still smoke-level only because teachers are intentionally undertrained.

---

## 5. 当前限制

P0 已完成，但以下问题还没有解决：

- FM teacher is still undertrained in smoke runs.
- Quant GAN calibration is post-hoc by default; formal reporting should disclose raw and calibrated metrics.
- Mean Flow still lacks boundary/identity loss decomposition.
- Ret-stage scheduled sampling is not implemented yet.
- Signature / path-level distance beyond current Wasserstein summaries is not implemented.

---

## 6. 建议下一步

进入 P1：

1. Implement Mean Flow boundary curriculum.
2. Split Mean Flow logging into:
   - total loss,
   - boundary loss,
   - identity loss,
   - boundary fraction.
3. Re-run medium smoke to verify identity loss decreases.
4. Then implement ret-stage scheduled sampling.

建议按这个顺序推进，因为 Mean Flow 是主方法，而当前 MF loss 没有 boundary / identity 拆分，诊断价值不足。
