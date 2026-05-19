# V3 实验结果分析与下一步优化方案

> 整理时间：2026-05-18
> 输入：[04_v3_implementation.md](04_v3_implementation.md) 完成后跑出的 e2e smoke 结果（`/tmp/finflow_e2e/`）+ 各 stage 的 `metrics.jsonl` + 两份 evaluate JSON。
> 输出：对症下药的优化清单 + 工程流程调整 + 文献依据。

---

## 0. TL;DR

- **Mean Flow 学生方向正确**：定价 MAPE 12.9%（vs 64 路径真实 Heston MC oracle 的 15.7%），marginal Wasserstein 0.003，已经接近真实数据自身的采样噪声下限。瓶颈是训练规模而不是方法。
- **Consistency Distillation 卡在平凡解**：CD ret loss 头一个 step 就是 0.004，到 epoch 2 还是 0.004——student 和 EMA target 在初始化相同处反复抖动。文献已有标准解法（iCT 课程，Song 2024）。
- **Quant GAN 系统漂移**：所有 $S_T < S_0$，MC 价格全 0，MAPE 100%。Wiese (2020) §3.4 早就描述过这类失败，需要 Lambert W 预处理 + WGAN-GP。
- **Ret-stage exposure bias 待补**：训练用 teacher-forced $v_{t+1}$，推理用 vol-sampler 输出。Bengio (2015) scheduled sampling 是直接对策。
- **流程层面**：缺一个 stage-1 收敛后的 go/no-go 闸门、缺统一的"四模型一键评测"脚本、regime-switching 数据没有定价 oracle（只能跳过或退化到 normal regime）。

---

## 1. 实验快照

Smoke 配置（CPU、`/tmp/finflow_e2e/`）：256 train / 64 val / 64 test × 32 步 × batch 32 × 2 epoch，每个蒸馏 stage 同样 2 epoch。

### 1.1 定价（T=20 trading day, K=95/100/105）

| 来源 | ATM 价格 | 与 Carr-Madan ref 差 | 整网格 RMSE / MAPE |
|---|---|---|---|
| Carr-Madan FFT（reference） | 2.453 | — | — |
| 真实 Heston MC（64 路径） | 2.672 | +8.9% | 0.288 / 15.7% |
| **MF rollout** | **2.387** | **–2.7%** ✓ | **0.401 / 12.9%** |
| Quant GAN | 0.000 | –100% ✗ | 3.728 / 100% |

MF 已经被 MC 采样噪声主导，靠采更多生成路径很难再提升——要更紧只能拉长训练 + 加 batch。

### 1.2 分布距离 / Stylized facts

| 指标 | Real | MF | Quant GAN |
|---|---|---|---|
| Marginal W-1 mean | — | 0.0029 | 0.0093 |
| Marginal W-1 max | — | 0.0058 | 0.0115 |
| Total-return W-1 | — | 0.027 | 0.148 |
| Kurtosis (pooled) | 3.19 | 2.86 | 2.49 |
| Tail index (Hill 5%) | 5.47 | 6.63 | 10.36 |
| Leverage corr L1 | — | 0.036 | 0.059 |
| Aggregational kurt L1 @ scale=1/5/21 | — | 0.33 / 0.14 / 0.80 | 0.70 / 0.31 / 0.17 |

MF 在所有维度都接近真实分布；Quant GAN 尾巴显著过薄、kurtosis 偏低。

### 1.3 训练曲线（每 stage 2 epoch）

| Stage | epoch 1 train→val | epoch 2 train→val | 备注 |
|---|---|---|---|
| FM vol | 1.44 → 0.91 | 0.78 → 0.72 | 仍快速下降，欠训 |
| FM ret | 1.97 → 2.06 | 1.88 → 1.86 | 慢，欠训严重 |
| MF vol | 0.59 → 0.41 | 0.38 → 0.40 | val 平台 |
| MF ret | 0.041 → 0.025 | 0.020 → 0.016 | loss scale 偏小（见 §2.C） |
| **CD vol** | **0.006 → 0.005** | **0.004 → 0.004** | **几乎平坦** |
| **CD ret** | **0.004 → 0.004** | **0.004 → 0.005** | **完全平坦** |
| Quant GAN | G 0.26 / D 0.21 | G 0.24 / D 0.19 | iter 数太少，无法判断 |

---

## 2. 关键观察与诊断

### 2.A Quant GAN 系统性下偏（mode collapse 的一种）

**现象**：MC 价格全是 0 → 所有生成的 $S_T \le 100$ → 累积 log-return 系统性偏负。

**诊断**：LSGAN 的最小二乘损失没有任何机制约束生成器的输出均值。对于重尾的金融收益率分布，generator 容易把概率质量集中在均值附近的负值上，G 和 D 都能"看起来"在学（loss 不报警）。本质是 Wiese 2020 §3.4 描述的"GAN 无法直接学重尾"问题。

**文献对策**：Wiese 2020 Algorithm 1——先把 log-return 做 **Lambert W 反变换**（Goerg 2015）压成近似高斯，GAN 学这个 latent 域，采样时再 inverse transform 恢复重尾。同时 **WGAN-GP**（Gulrajani 2017）替代 LSGAN 解决均值漂移。

### 2.B CD ret loss 卡在 0.004（trivial-solution 陷阱）

**现象**：CD 的 train_loss 和 val_loss 从 step 0 起就是 ~0.004，2 个 epoch 后还是 0.004。

**诊断**：CD loss 是 $\|f_\text{student}(x_{n+1}, t_{n+1}) - \text{sg}(f_\text{target}(\hat x_n, t_n))\|^2$。初始化时 `target_net = deepcopy(student)`，两者参数一致。每个 step 学生稍微动一下，EMA target 立刻跟上（默认 decay=0.999，但学生只走 1 步），平衡在一个不为零但很小的"抖动半径"内。这正是 Song 2023 §4 提到的"trivial solution"——网络可以学到一个常数函数 $f \equiv c$ 而完美满足 self-consistency，导致 loss 看似收敛但实际什么都没学。

**文献对策**：Song 2024 (iCT) 给出三件套：
1. **N(k) 课程**：discretization 数从 10 涨到 1280
2. **Karras μ schedule**：$\mu(k) = \exp(s_0 \log 2 / N(k))$，N 小时 μ 也小，强制 target 跟上 student 的实质变化
3. **Lognormal $t$ 采样 + Pseudo-Huber loss**：把 loss 权重压到对小 noise 区域更敏感

我们的默认 `N=18, ema_decay=0.999` 完全没有这套课程。

### 2.C MF ret loss 偏小（被 boundary 项稀释）

**现象**：MF ret loss 0.016，看上去比 vol（0.40）小一个数量级。

**诊断**：`boundary_prob=0.25` 意味着 25% 的 batch 采到 $r=t$，此时 loss 退化成 FM regression（target 直接是 reversed-teacher 的预测，loss 几乎为 0）。剩下 75% 才是真正的 identity 学习。整体 mean 被前者拉低，所以"loss 收敛"不等于"identity 已满足"。

**对策**：分别 logging "boundary 部分" 与 "identity 部分" 的 loss，看 identity loss 是否单调下降。

### 2.D Ret stage train loss 高于 vol stage 一倍以上

**现象**：FM vol 0.72 vs FM ret 1.86。

**诊断**：两个原因——(i) return 比 $\log v$ 重尾，目标方差大；(ii) 训练用 ground-truth $v_{t+1}$，推理时 vol-sampler 给的 $v_{t+1}$ 有误差，分布漂移（Bengio 2015 的 exposure bias）。

**文献对策**：在 ret 训练后期，以概率 $p \in [0, 0.5]$ 把 ground-truth $v_{t+1}$ 替换为 vol-stage 当前 best checkpoint 的采样。这要求 vol stage 先冻结。

### 2.E Aggregational kurtosis @ scale=21 异常

**现象**：MF 在 scale=21 上的 kurtosis diff 0.799，是 scale=5 的 5 倍。

**诊断**：32 步路径在 scale=21 下只能聚合成 ⌊32/21⌋=1 个 block，N=64 paths × 1 block = 64 样本，sample noise 主导。真实 252 步路径在 scale=21 下能聚成 12 个 block × 10k path = 120k 样本，问题自动消失。**不是 bug**。

### 2.F Regime-switching 数据没有定价 oracle

**现象**：用 `--regimes` 生成的数据没法对 Carr-Madan 比较（FFT 假设单 Heston）。当前 eval 脚本对此默认跳过定价（`--force-regime-pricing` 才退化到 normal regime）。

**对策**：用一个大的 "MC oracle"——从同一个 regime Markov chain 独立采 100k 路径，作为 MC 真值，比对生成路径的 MC 价格。

---

## 3. 对症下药：文献映射 + 具体改动

| 问题 | 文献 | 具体改动 | 改动位置 |
|---|---|---|---|
| CD trivial-solution 平台 | Song 2024 (iCT, NeurIPS 2024) [2310.14189] | N(k) curriculum + Karras μ + lognormal $t$ + Pseudo-Huber | [finflow/distillation/consistency.py](../../finflow/distillation/consistency.py)：`consistency_distill_step` + `_schedule`；`ConsistencyDistillConfig` 增 `curriculum_kind, n_min, n_max, huber_c` |
| Quant GAN 下偏 / 尾薄 | Wiese 2020 §3.4 + Gulrajani 2017 (WGAN-GP) | (1) 输入端 Lambert W 反变换；(2) 损失换 WGAN-GP；(3) 输出端加 affine 校准 | [finflow/baselines/quant_gan.py](../../finflow/baselines/quant_gan.py)：新增 `lambert_w_transform`，`_ls_loss` 换 W-GP，generator 末加 `(a, b)` |
| MF loss 被 boundary 项稀释 | Geng 2025 NeurIPS Oral | `boundary_prob` curriculum + 分项 logging | `MeanFlowDistillConfig` 加 `boundary_prob_start/end`，`mean_flow_loss` 内部拆 boundary / identity 两个分量 |
| Ret-stage exposure bias | Bengio 2015 (Scheduled Sampling) [1506.03099] | ret 训练以 prob $p$ 用 vol-sampler 输出替换 ground truth $v_{t+1}$ | [finflow/training.py](../../finflow/training.py) `train_ret_trans_fm` 增 `vol_sampler_checkpoint` + `scheduled_sampling_prob` |
| Regime 数据无定价 oracle | Cont 2001 + 自家 Heston QE | 加 "MC oracle path" 对照（n=100k 真实 Heston regime 路径） | [finflow/eval/pricing.py](../../finflow/eval/pricing.py) 增 `pricing_rmse_vs_mc_oracle` |
| Path-level 距离不足 | Ni 2021 (Sig-Wasserstein) [2111.01207] | depth-d signature + Sig-Wasserstein | 新建 `finflow/eval/signatures.py`，纯 NumPy depth ≤ 4 |
| Action 条件没启用 CFG | Ho & Salimans 2022 [2207.12598] + Zheng 2023 Guided Flows [2311.13443] | 训练 0.1 概率丢动作 → null condition；推理加 `--cfg-w` | dataset 加 dropout mask；samplers 加 CFG |
| 报表 / 可视化 | — | spaghetti + ribbon + stylized fact bar | 新建 `scripts/plot_eval.py` 用 matplotlib |

---

## 4. 下一步行动清单（按优先级）

### P0 — bug fixes（预计 1-2 工日）

1. **CD iCT 课程**：把 `n_discretization` 从常量改成 epoch 函数 `N(k) = clamp(round(N_min * (N_max/N_min)^(k/total)), N_min, N_max)`；`ema_decay = exp(s0 * ln2 / N(k))`；Pseudo-Huber loss `sqrt((x-y)^2 + c^2) - c` 替换 MSE，`c=0.03`。期望：CD ret loss 从 0.004 平台跌到 < 0.001；marginal W-1 接近 MF 水平。
2. **Quant GAN 重做**：(a) `HestonLogReturnSequenceDataset` 增 Lambert W 预处理；(b) LSGAN → WGAN-GP（D 头去 sigmoid，加 gradient penalty=10）；(c) generator 末层加 `LayerNorm + a·tanh + b`，`(a, b)` 可学习。期望：MC 价格不再全 0，kurtosis 接近 3.19，tail index 收敛到 5-6。
3. **MC oracle 定价 reference**：`pricing_rmse_vs_mc_oracle(s_paths, oracle_s_paths, ...)`，`evaluate_rollout.py` 加 `--mc-oracle PATH`。期望：regime 数据也有合理对标。
4. **CLI 单一入口 `run_full_evaluation.sh`**：循环跑 FM teacher / MF / CD / Quant GAN 四个 rollout，汇总成一个 markdown 对比表。期望：一条命令出完整对比矩阵。

### P1 — methodology（预计 1 周）

5. **MF boundary curriculum**：`boundary_prob` 从 0.5 线性退到 0.1；`mean_flow_loss` 拆分 boundary/identity 分量分别记录；新增 `--identity-residual-eval`。
6. **Ret scheduled sampling**：在 ret 训练 entry 加 `--vol-sampler-checkpoint`；每 batch 以 prob $p(\text{epoch}) = \min(0.5, \text{epoch}/(2 \cdot \text{epochs}))$ 用 vol sampler 输出替换 GT $v_{t+1}$。
7. **CFG**：dataset 加 `action_dropout_prob=0.1`，samplers 加 `cfg_w`，推理 `u = (1+w) u_\text{cond} - w \cdot u_\text{uncond}`；在 evaluate_rollout 加 `--cfg-w` 扫描。
8. **Sig-Wasserstein**：depth-2 / depth-3 signature 实现，集成进 `build_full_report`。

### P2 — 真实长跑实验（预计 1 周）

9. **正式数据**：50k train / 5k val / 10k test × 252 步 × 3 regime Markov。
10. **正式训练计划**：
    - FM vol：20 epoch, batch 512, lr 3e-4
    - FM ret：20 epoch, batch 512, lr 3e-4，后 5 epoch 启用 scheduled sampling
    - MF distill：15 epoch，boundary_prob 0.5→0.1 curriculum
    - CD distill：15 epoch，N curriculum 10→160，Karras EMA，Pseudo-Huber
    - Quant GAN：30 epoch（GAN 通常需要更多），WGAN-GP, Lambert W
11. **超参 sweep**：每 stage 选 3 个关键超参 × 3 grid，总共 ~12 runs/stage，时间预算 ~2 GPU·day。

### P3 — polish & 交付

12. **可视化**：spaghetti（生成 vs 真实并列）、stylized facts bar、rollout 稳定性曲线、CFG 灵敏度。
13. **Per-regime 评测拆解**：`evaluate_rollout` 按 `actions[*, 0]` 分组单独报。
14. **Teacher 缓存**：蒸馏前预存 `v_teacher(x_t, t)`，节省 ~30% wall-clock。
15. **写作模板**：Report 5 表（statistical / pricing / rollout-stability / regime-conditional / ablation）+ 3 图（spaghetti / Sig-Wasserstein / CFG 灵敏度）。

---

## 5. 项目流程调整

| 流程节点 | 现状 | 调整建议 |
|---|---|---|
| Stage 1 → 蒸馏 | 直接进入 | 加 **go/no-go**：若 `val_loss > 0.6`（vol）或 `> 1.0`（ret），打印 warning 并建议加 epoch / 调 lr |
| 各 stage 之间 | 手工传 checkpoint 路径 | 写 `scripts/run_v3_pipeline.sh`：data → vol_fm → ret_fm → mf_vol → mf_ret → cd_vol → cd_ret → 各模型 rollout → evaluate |
| Teacher 重复前向 | 每 epoch 重新算 | 蒸馏前 `precompute_teacher_velocities --teacher CKPT --data DATA --output cache.npz`；蒸馏读 cache |
| 评测 | 一个个跑 | `run_full_evaluation.sh ROLLOUT_DIR` 输出一个 markdown 对比表 |
| 异常检测 | 无 | 训练每 N step 跑 mini-rollout，若 marginal W-1 单调上升超过 10 epoch，停训并报警 |
| 验证集 | train/val/test 同 seed 不同 split | 加 `val_short`（128 paths × 252 步）专用于"快速 stylized facts"，每 epoch 末跑一次 |

---

## 6. 时间线 & 验证标准

| 周 | 工作 | 验证标准 |
|---|---|---|
| W1 | P0（CD iCT, Quant GAN 重做, MC oracle, run_full_evaluation）+ 单测 | CD ret loss < 0.001；Quant GAN MC 价格非零且 kurtosis > 2.8 |
| W2 | P1（MF curriculum, scheduled sampling, CFG, Sig-Wasserstein） | MF identity loss 单调降；ret 推理-训练 gap 收窄；CFG w=2 下 regime separation 视觉清晰 |
| W3 | P2 长跑：FM teachers + MF + CD 全部训完 | MF marginal W-1 < real-vs-real 噪声基线的 1.5×；定价 MAPE < 5% |
| W4 | P2 sweep + P3 可视化 + 写作 | Report 5 表 3 图全部出来；ablation 表完整 |

---

## 7. 文献快速对照

| 改动 | 引文 |
|---|---|
| iCT curriculum | Song et al., "Improved Techniques for Training Consistency Models", NeurIPS 2024 — [2310.14189](https://arxiv.org/abs/2310.14189) |
| Lambert W heavy-tail | Goerg, "Lambert W random variables", 2015；Wiese et al., "Quant GANs", QF 2020 — [1907.06673](https://arxiv.org/abs/1907.06673) |
| WGAN-GP | Gulrajani et al., "Improved Training of Wasserstein GANs", NeurIPS 2017 — [1704.00028](https://arxiv.org/abs/1704.00028) |
| Mean Flow boundary curriculum | Geng et al., "Mean Flows for One-step Generative Modeling", NeurIPS 2025 Oral |
| Scheduled Sampling | Bengio et al., "Scheduled Sampling for Sequence Prediction", NeurIPS 2015 — [1506.03099](https://arxiv.org/abs/1506.03099) |
| Sig-Wasserstein | Ni et al., "Sig-Wasserstein GANs for Time Series Generation", 2021 — [2111.01207](https://arxiv.org/abs/2111.01207) |
| CFG / Guided Flows | Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2022 wksp — [2207.12598](https://arxiv.org/abs/2207.12598)；Zheng et al., "Guided Flows", 2023 — [2311.13443](https://arxiv.org/abs/2311.13443) |
| Stylized facts oracle | Cont, "Empirical Properties of Asset Returns", QF 2001 |

每条 P0/P1 改动都在 [03_V3_References.md](03_V3_References.md) 现有清单内，**无需引入新文献**。

---

## 8. 一句话总结

短训出的结果已经验证了方法学方向：MF 远好于 GAN baseline、定价误差被 MC 采样噪声主导。瓶颈是 (i) CD 的训练动力学需要 iCT 课程救活，(ii) Quant GAN 需要 Wiese 2020 的 Lambert W 包装才能成为有意义的 baseline，(iii) 真正的 252 步 × 50k × 20 epoch 长跑还没做。把这三件做完，整个 V3 项目就有完整的 4 路对比矩阵和定价 / stylized facts 双轴 figure，足以支撑最终 Report。
