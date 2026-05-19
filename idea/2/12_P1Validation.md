# V3 P1 Validation

日期：2026-05-19

验证目标：确认 P1 hook 不只是单元测试通过，而是在一组受控 smoke 中能产生日志、指标和可解释的行为。

## 1. 配置

数据：

- 输出目录：`runs/p1_validation/data`
- regime switching：开启，3 actions
- train / val / test：512 / 128 / 128 paths
- steps：32
- seed：20260519

训练：

- vol FM：4 epoch，batch 64，hidden 32，2 blocks，`action_dropout_prob=0.1`，每 epoch 限制 24 train batches / 8 val batches
- ret vanilla：3 epoch，每 epoch 限制 16 train batches / 8 val batches
- ret scheduled：同 ret vanilla，额外使用 vol checkpoint，`scheduled_sampling_max_prob=0.5`
- MF vol distill：5 epoch，`boundary_prob 0.5 -> 0.1`，开启 `identity_residual_eval`

rollout / eval：

- vol checkpoint：`runs/p1_validation/vol_fm/vol_p1_fast/checkpoints/best.pt`
- ret checkpoint：`runs/p1_validation/ret_fm/ret_sched_fast/checkpoints/best.pt`
- rollout：128 paths，32 steps，same action seed / noise seed，`fm_n_steps=2`
- CFG sweep：`cfg_w=0` vs `cfg_w=2`
- evaluation：`signature_depth=3`

## 2. 结果

### Vol Teacher

| Metric | First | Last |
|---|---:|---:|
| val_loss | 2.4494 | 1.4001 |

vol teacher 在受限 batch 的 smoke 下正常下降，可作为后续 ret scheduled sampler。

### Ret Scheduled Sampling

| Run | Best val_loss |
|---|---:|
| ret vanilla | 1.9933 |
| ret scheduled | 1.9296 |

scheduled sampling probability：

| Epoch | p |
|---:|---:|
| 1 | 0.1667 |
| 2 | 0.3333 |
| 3 | 0.5000 |

结论：ret scheduled sampling hook 生效，训练稳定，并且在这次 smoke 中 best val loss 略好于 vanilla。

### Mean Flow Boundary Curriculum

| Epoch | boundary_prob | val_identity_loss | val_identity_residual |
|---:|---:|---:|---:|
| 1 | 0.5000 | 1.1229 | 1.3402 |
| 2 | 0.4000 | 0.9133 | 1.0950 |
| 3 | 0.3000 | 0.8002 | 0.7502 |
| 4 | 0.2000 | 0.7118 | 0.6148 |
| 5 | 0.1000 | 0.6375 | 0.5362 |

结论：MF curriculum hook 生效，identity 相关指标随 epoch 明显下降。这是 P1 最强的正向信号。

### CFG Sweep

| cfg_w | Marginal W1 mean | Total-return W1 | Sig-W1 mean |
|---:|---:|---:|---:|
| 0 | 0.002219 | 0.027724 | 0.005110 |
| 2 | 0.002301 | 0.034030 | 0.006225 |

同一 action / noise seed 下，`cfg_w=2` 与 `cfg_w=0` 的生成路径差异：

| Metric | Value |
|---|---:|
| mean abs return-path diff | 0.000199 |
| mean abs terminal-price diff | 0.6415 |

按 action 的平均 return 变化：

| Action | Count | cfg_w=0 mean r | cfg_w=2 mean r | delta |
|---:|---:|---:|---:|---:|
| 0 | 3886 | 0.000371 | 0.000569 | 0.000198 |
| 1 | 172 | 0.001096 | 0.001343 | 0.000247 |
| 2 | 38 | 0.003434 | 0.003304 | -0.000130 |

结论：CFG 机制确实改变了输出，但在这次弱 teacher + 小样本 smoke 中 `cfg_w=2` 没有改善分布距离，反而略差。P1 的 CFG hook 可用，但还不能声称 CFG 本身带来质量收益。

### Sig-Wasserstein

`evaluate_rollout.py --signature-depth 3` 正常写出：

- `distances.signature_wasserstein.depth`
- `distances.signature_wasserstein.mean`
- `distances.signature_wasserstein.max`
- `distances.signature_wasserstein.per_coordinate`
- `distances.signature_wasserstein.coordinate_names`

结论：Sig-W report integration 生效。当前 smoke 中 Sig-W 与 total-return W1 对 CFG 的排序一致：`cfg_w=0` 优于 `cfg_w=2`。

## 3. 总结

P1 有效性结论：

- 有效：Mean Flow boundary curriculum + identity logging。
- 有效：ret scheduled sampling，且本次 smoke 略优于 vanilla。
- 有效：Sig-Wasserstein 接入报告，可作为 path-level metric。
- 部分有效：CFG hook 生效，但本次 smoke 不支持 `cfg_w=2` 提升质量的结论。

建议下一步：

1. 提交 P1 代码和本文档。
2. P2 前做一个更合理的 CFG sweep：`cfg_w in {0, 0.5, 1, 2}`，并使用更好的 teacher / 更多 paths。
3. 正式实验里同时报告 raw rollout 和 CFG rollout，不默认把 CFG 当成必然收益。
