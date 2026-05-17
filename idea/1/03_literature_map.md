# LOB 生成模型 & 相关工作调研（2023–2025）

## 按代际分类

| 代际 | 代表工作 | 年份 | 技术要点 | 核心缺陷 |
|---|---|---|---|---|
| 参数化 | Cont et al. | 2010 | 手工统计模型 | 不真实、不可扩展 |
| GAN | Coletta；Cont/Cucuringu | 2022–23 | CGAN 生成 LOB 事件 | 训练不稳、mode collapse；大 tick 股票失效；只到 top-6 levels |
| AR / SSM | **LOBS5** (Nagy) | 2023 | 35M，逐 message 自回归 | LOB-Bench 冠军但判别器 ROC 仍 0.83；下游预测反掉点 |
| AR-Transformer | RWKV-4/6；TLOB | 2025 | 170M 级 | 快速发散 / 只做分类 |
| Diffusion（事件域）| **TRADES** | 2025-02 | Transformer-DDPM；过去 256 订单 + 10 级 LOB 条件；滑窗 AR | 6h / 仿真 h (3090)；DDIM-1 劣化 2.6× |
| Diffusion（图像域）| **Painting the Market** | 2025-09 | LOB→图像，inpainting 并行生成 | 重全局结构、**弱 local 细节**（作者自述）|
| Diffusion（volume）| **DiffVolume** | 2025-08 | 只扩散 10 档 volume（20 维）| 不含价格 / 事件 |
| 反事实 | **DiffLOB** | 2026-02 | regime-conditioned 反事实 | 非开环仿真，用途窄 |

## 关键评测基准

**[LOB-Bench](https://arxiv.org/abs/2502.09172)** (ICML 2025)
- 距离类：L1 / Wasserstein-1 on spread, order volumes, order imbalance, message inter-arrival times
- 条件分布匹配（如 spread × 时段）
- **Market impact**：cross-correlation + price response function
- **Adversarial score**：训练判别器区分真假（ROC）
- **Downstream**：用合成数据训练 mid-price 分类器看 F1
- 测试股票：GOOG + INTC (2023-01)
- **结论**：LOBS5 最强但判别器 ROC=0.83 仍可区分；合成数据**明显拉低**下游预测精度 — 当前模型尚无法真正"有用"
- 开源：[peernagy/lob_bench](https://github.com/peernagy/lob_bench)

## DIAMOND（世界模型范式模板）
- [arXiv 2405.12399](https://arxiv.org/abs/2405.12399)（NeurIPS 2024 spotlight，课程参考论文列表内）
- 全称 DIffusion As a Model Of eNvironment Dreams
- Atari 版仅 **4.4M 参数**，训练 2.9 天，Atari 100k 拿到 1.46 human-normalized score（纯 world-model SOTA）
- CS:GO 版扩到 381M（含 51M upsampler），**两阶段 low-res dynamics + upsample**
- **关键发现**：EDM 参数化对多步 rollout 稳定性至关重要
- 开源：[eloialonso/diamond](https://github.com/eloialonso/diamond)

## Flow Matching / Consistency 时间序列相关

| 工作 | 要点 |
|---|---|
| [FM-TS](https://arxiv.org/abs/2411.07506) | Rectified Flow 做时间序列生成，context FID 显著超 baseline |
| [FlowTS](https://arxiv.org/abs/2411.07506) | ODE-based，Stock/ETTh 数据 FID 0.019/0.011 |
| [TimeFlow](https://arxiv.org/abs/2511.07968) | SDE-based，component-wise 速度场 |
| Consistency Models (Song 2023) | 课程参考列表，1-step 生成 |
| Mean Flow (Geng 2025) | 课程参考列表，average velocity |

## Diffusion World Model for RL（C4 依据）
- [DAWM](https://arxiv.org/abs/2509.19538)：diffusion world model + inverse dynamics 做 offline RL
- [DWM](https://arxiv.org/abs/2402.03570)：multi-step 未来预测，offline RL
- [World4RL](https://arxiv.org/html/2509.19080)：机械臂 policy refinement

## DeepLOB（LOB→image 表示的先例）
- [arXiv 1808.03668](https://arxiv.org/pdf/1808.03668) — IEEE TSP
- 将 LOB 的 40 维价格/量向量沿时间堆成图像，用 CNN+Inception+LSTM
- 证明 LOB tensor 天然适合 CNN，是我们"把 LOB 当图像"的关键引用
- 开源：[zcakhaa/DeepLOB-...](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)