# P2 Medium Results: Core Findings and Next Plan

日期：2026-05-25

远端实验位置：

- medium 主基线：`runs/experiments/p2_medium_complete`
- 新 teacher 选择：`runs/experiments/p2_teacher_select_20260522T100554Z`
- 新 teacher 蒸馏：`runs/experiments/p2_new_teacher_distill_20260524T144921Z`

本文只记录当前 medium 实验的核心结论、复现/创新边界，以及下一步计划。

## 1. 当前主结论

当前结果不是“创新方法全面超过所有复现方法”，而是更细：

- **two-stage regime-conditioned FM teacher** 是当前最强 flow-family 模型。
- 它在 path-level distribution 上超过 QGAN calibrated：Total W1 `0.0232` vs `0.0321`，Sig-W1 `0.0061` vs `0.0083`。
- 它在 option pricing 上略弱于 QGAN calibrated：Pricing RMSE `1.7640` vs `1.5141`。
- MF/CD 蒸馏提升了旧 MF/CD，但没有超过新 FM teacher，也没有超过 QGAN calibrated。
- 高 CFG 对新 MF 是负向的，不应继续作为主要优化方向。

## 2. 核心结果矩阵

| Model | 定位 | Total W1 | Sig-W1 | Pricing RMSE | Pricing MAPE |
|---|---|---:|---:|---:|---:|
| **FM new teacher last** | 本项目最强 flow 模型 | **0.0232** | **0.0061** | 1.7640 | 0.1857 |
| MF new best cfg=0 | 新 teacher 蒸馏后的最好 MF pricing 学生 | 0.0486 | 0.0266 | 2.1657 | 0.2357 |
| MF new last cfg=0.25 | 新 teacher 蒸馏后的较均衡 MF 学生 | 0.0380 | 0.0109 | 2.7975 | 0.2900 |
| CD new best | 新 teacher 蒸馏后的最好 CD 学生 | 0.0533 | 0.0113 | 3.3795 | 0.1998 |
| old FM teacher | 旧 flow teacher | 0.2718 | 0.0795 | 4.7400 | 0.3526 |
| old MF best CFG | 旧 MF baseline | 0.0792 | 0.0165 | 2.9719 | 0.2778 |
| old CD | 旧 CD baseline | 0.0632 | 0.0122 | 3.6069 | 0.2832 |
| **QGAN last calibrated** | 最强 pricing baseline | 0.0321 | 0.0083 | **1.5141** | **0.1549** |

可以写进报告的准确表述：

> Our two-stage regime-conditioned FM teacher achieves the best path-level distribution metrics among all evaluated models, while QGAN last calibrated remains the strongest option-pricing baseline. MF/CD distillation improves over old students but loses part of the teacher's pricing advantage.

## 3. 复现方法 vs 项目创新

### 复现/应用已有论文方法

| 实验项 | 性质 | 说明 |
|---|---|---|
| FM teacher | 复现/应用已有 Flow Matching | FM 不是我们提出的；我们把它用于金融路径 transition generation。 |
| MF / Mean Flow | 复现/应用已有 Mean Flow | 作为 1-NFE student distillation 方法。 |
| CD / Consistency Distillation | 复现/应用已有 Consistency/iCT | 作为另一个 1-NFE student baseline。 |
| QGAN raw | 复现金融生成 baseline | Quant GAN 是已有金融时间序列生成模型。 |
| QGAN calibrated / QGAN last | baseline 改造与消融 | calibration 和 last-checkpoint 是实验变体，不应夸成核心方法创新。 |
| CFG sweep | 应用已有 CFG 思路 | 本实验发现高 CFG 反而破坏 MF 路径分布。 |
| Heston / MC oracle / pricing metrics | 经典金融 benchmark 和评估 | 用来保证实验可控和可解释。 |

### 本项目自己的贡献

| 贡献 | 性质 | 当前证据 |
|---|---|---|
| two-stage vol/ret financial world model | 核心结构设计 | 新 FM teacher 显著强于旧 FM/MF/CD。 |
| regime-action conditional generation | 应用创新 | 把 Heston regime action 纳入生成条件。 |
| FM / MF / CD / QGAN 完整矩阵 | 实验设计贡献 | 能同时比较 path distribution 与 pricing。 |
| best vs last checkpoint ablation | 实证发现 | 新 ret teacher 的 `last.pt` 显著优于 `best.pt`。 |
| pricing-aware diagnosis | 分析贡献 | 发现 validation loss、path metric、pricing metric 排序不一致。 |
| negative result: high CFG hurts MF | 负结果贡献 | `cfg >= 0.25` 会显著放大 Total W1 / Sig-W1 / pricing error。 |

## 4. 创新方法是否比复现方法更好？

答案：**部分更好，但不是全面更好。**

如果“创新方法”指本项目的 two-stage regime-conditioned FM teacher：

- 对 old FM / old MF / old CD：明显更好。
- 对 QGAN raw / QGAN best raw / QGAN best calibrated：整体更好。
- 对 QGAN last calibrated：path-level 更好，但 pricing RMSE 稍弱。

如果“创新方法”指 MF/CD 蒸馏学生：

- 比 old MF/CD 有提升。
- 没有超过新 FM teacher。
- 没有超过 QGAN last calibrated。
- 因此 MF/CD 应写成 **speed-quality tradeoff**，而不是最终最强模型。

## 5. 关键实验发现

### Teacher selection

| Teacher | Total W1 | Sig-W1 | Pricing RMSE |
|---|---:|---:|---:|
| tf_p0_start6 best | 0.1098 | 0.0216 | 4.0884 |
| **tf_p0_start6 last** | **0.0232** | **0.0061** | **1.7640** |
| ss02_start6 last | 0.1282 | 0.0240 | 6.1571 |
| ss05_start6 last | 0.1362 | 0.0254 | 6.4182 |

结论：

- `tf_p0_start6 last` 是当前最好 teacher。
- scheduled sampling teacher 没有收益。
- checkpoint selection 不能只看 validation loss。

### Distillation failure modes

- MF best cfg=0 的 pricing 尚可，但产生极端左尾：terminal `q01=9.7152`，return kurtosis `12.8516`。
- MF last cfg=0.25 的 path metrics 更均衡，但 pricing RMSE 仍有 `2.7975`。
- CD best 的短期 pricing 很好，但长期 maturity 偏差大：1 年 RMSE `5.7585`。
- CD last 明显退化，不作为主结果。

## 6. 当前报告主线

建议最终报告采用三条主线：

1. **方法主线**：two-stage regime-conditioned FM teacher 是有效的金融路径 world model。
2. **baseline 主线**：QGAN last calibrated 是强 pricing baseline，但依赖 sampling-time calibration。
3. **蒸馏主线**：MF/CD 能把旧学生模型做强，但当前 1-NFE distillation 会损失 teacher 的 pricing surface。

不要写：

- “我们提出了 Flow Matching / Mean Flow / Consistency / QGAN。”
- “我们的创新方法全面超过所有 baseline。”

可以写：

- “We adapt and compare FM, Mean Flow, Consistency Distillation, and QGAN in a two-stage regime-conditioned Heston world-model pipeline.”
- “The proposed two-stage FM teacher is the best path-level model, while calibrated QGAN remains the strongest pricing baseline.”
- “Our checkpoint and maturity-wise diagnostics reveal that validation loss is not sufficient for financial path generation.”

## 7. 下一步计划

优先级：

1. **固化写作口径**
   - 明确复现方法和项目贡献边界。
   - 主结果使用 `FM new teacher last`、`QGAN last calibrated`、`MF new best cfg=0`、`MF new last cfg=0.25`、`CD new best`。

2. **做一个小型 pricing-aware checkpoint selection 实验**
   - 不再按 validation loss 单独选 checkpoint。
   - 每隔若干 epoch 做小 rollout。
   - 用 Total W1 / pricing proxy 选 checkpoint。
   - 这是最可能形成“自己方法改进”的下一步。

3. **MF targeted fix**
   - 只保留 `cfg=0` 和 `cfg=0.25`。
   - 试 `epochs=20`，lr 降到 `1e-4` 或 `2e-4`。
   - 目标是减少极端左尾，同时保住 pricing。

4. **CD targeted diagnosis**
   - 只看 best checkpoint。
   - 分解 vol student / ret student 对 terminal drift 的贡献。
   - 重点解释为什么 CD 的 1 年 maturity pricing 偏差大。

5. **最后做可视化和报告 polish**
   - path spaghetti / terminal distribution quantiles。
   - maturity-wise pricing RMSE。
   - full matrix table。
   - checkpoint ablation figure。

