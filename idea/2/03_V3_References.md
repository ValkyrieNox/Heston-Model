# V3 Pipeline 参考文献清单

> 整理时间：2026-05-13
> 用途：V3（自回归世界模型 Pipeline）的全部理论基础和已有工作参考
> 原则：每条文献都要标明与本项目的具体关系，不堆砌

V3 的核心内容：用 Flow Matching 学习随机过程的转移核 $p(s_{t+1}|s_t, a_t)$，用 Mean Flow 蒸馏为单步生成器，在 Heston 合成数据上自回归滚动验证。

---

## 一、核心生成模型方法（我们直接使用的方法）

### 1.1 Flow Matching（核心训练方法）

| 文献 | arXiv | 关系 |
|---|---|---|
| Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023 | [2210.02747](https://arxiv.org/abs/2210.02747) | **课程参考论文**。V3 中 FM_vol_trans 和 FM_ret_trans 的训练方法直接来自此 |
| Albergo & Vanden-Eijnden, "Stochastic Interpolants: A Unifying Framework", ICLR 2023 | [2303.08797](https://arxiv.org/abs/2303.08797) | FM 的统一理论框架，对理解 FM 和 SDE 的关系很关键 |
| Liu et al., "Flow Straight and Fast: Rectified Flow", ICLR 2023 | [2209.03003](https://arxiv.org/abs/2209.03003) | Rectified Flow，FM 的等价视角，对路径直化有用 |
| Tong et al., "Improving and Generalizing Flow-Based Generative Models", ICML 2024 | [2302.00482](https://arxiv.org/abs/2302.00482) | OT-CFM，最优传输视角的 FM 改进 |

### 1.2 Mean Flow（核心蒸馏方法）

| 文献 | arXiv | 关系 |
|---|---|---|
| Geng et al., "Mean Flows for One-step Generative Modeling", NeurIPS 2025 **Oral** | （检索时确认）| **课程参考论文**。V3 Stage 2 的蒸馏方法。原论文只在 ImageNet 上验证，时序数据上无人做过——我们的创新点 |
| **GitHub 开源实现**：noamelata/MeanFlow | github.com/noamelata/MeanFlow | 我们 JVP 代码 port 的来源 |
| **GitHub 开源实现**：haidog-yaqub/MeanFlow | github.com/haidog-yaqub/MeanFlow | 备选 port 来源 |

### 1.3 Consistency Models（V3 的对比基线）

| 文献 | arXiv | 关系 |
|---|---|---|
| Song et al., "Consistency Models", ICML 2023 | [2303.01469](https://arxiv.org/abs/2303.01469) | **课程参考论文**。CD 蒸馏方法，作为 V3 的对比基线（CD vs MF 哪个蒸馏更好）|
| Song et al., "Improved Techniques for Training Consistency Models", NeurIPS 2024 | [2310.14189](https://arxiv.org/abs/2310.14189) | iCT，CT 超参改进版，训练更稳 |
| Song et al., "Simplifying, Stabilizing, and Scaling Continuous-time Consistency Models", 2025 | [2410.11081](https://arxiv.org/abs/2410.11081) | sCM，最新的连续时间 CM |

### 1.4 Score-Based / DDPM 基础（理论背景）

| 文献 | arXiv | 关系 |
|---|---|---|
| Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021 | [2011.13456](https://arxiv.org/abs/2011.13456) | **课程参考论文**。FM 与 Score SDE 的连接理论，Report 中要引用 |
| Ho et al., "DDPM", NeurIPS 2020 | [2006.11239](https://arxiv.org/abs/2006.11239) | 扩散模型奠基性工作，理论参考 |
| Lu et al., "DPM-Solver++", NeurIPS 2022 | [2211.01095](https://arxiv.org/abs/2211.01095) | 多步采样加速（如果做 Score SDE 基线对比时用）|
| Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", NeurIPS 2022 | [2206.00364](https://arxiv.org/abs/2206.00364) | EDM，统一扩散模型设计空间，超参参考 |

---

## 二、世界模型（V3 的核心 framing）

### 2.1 直接对标的世界模型

| 文献 | arXiv | 关系 |
|---|---|---|
| Alonso et al., "Diffusion for World Modeling: Visual Details Matter in Atari", NeurIPS 2024 (DIAMOND) | [2405.12399](https://arxiv.org/abs/2405.12399) | **课程参考论文**。V3 的直接结构类比——DIAMOND 用扩散学 $p(o_{t+1}\|o_t,a_t)$，我们用 FM 学 $p(s_{t+1}\|s_t,a_t)$ |
| Valevski et al., "Diffusion Models Are Real-Time Game Engines", 2024 (GameNGen) | [2408.14837](https://arxiv.org/abs/2408.14837) | Google 用扩散做 DOOM 实时世界模型，与 DIAMOND 并列 |
| Bruce et al., "Genie: Generative Interactive Environments", ICML 2024 | [2402.15391](https://arxiv.org/abs/2402.15391) | DeepMind 大规模 latent 动作世界模型 |

### 2.2 世界模型经典与谱系

| 文献 | arXiv | 关系 |
|---|---|---|
| Ha & Schmidhuber, "World Models", NeurIPS 2018 | [1803.10122](https://arxiv.org/abs/1803.10122) | 世界模型概念的奠基工作。Report 介绍 WM 时必引 |
| Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination", ICLR 2020 (Dreamer) | [1912.01603](https://arxiv.org/abs/1912.01603) | Dreamer-V1，RSSM 架构 |
| Hafner et al., "Mastering Atari with Discrete World Models", ICLR 2021 (DreamerV2) | [2010.02193](https://arxiv.org/abs/2010.02193) | DreamerV2，离散 latent |
| Hafner et al., "Mastering Diverse Domains through World Models", 2023 (DreamerV3) | [2301.04104](https://arxiv.org/abs/2301.04104) | DreamerV3，跨领域统一架构 |
| Micheli et al., "Transformers are Sample-Efficient World Models", ICLR 2023 (IRIS) | [2209.00588](https://arxiv.org/abs/2209.00588) | Transformer 世界模型，对比 DIAMOND 的扩散路线 |
| Robine et al., "Transformer-based World Models Are Happy with 100k Interactions", ICLR 2023 (TWM) | [2303.07109](https://arxiv.org/abs/2303.07109) | 另一篇 Transformer 世界模型 |

### 2.3 物理 / 科学 / 连续过程世界模型（与本项目最相关）

| 文献 | arXiv | 关系 |
|---|---|---|
| Lu et al., "Towards Generalist Biomedical AI", 2023 | [2307.14334](https://arxiv.org/abs/2307.14334) | 生物医学连续状态预测（参考意义有限）|
| Wu et al., "Daydreamer: World Models for Physical Robot Learning", CoRL 2022 | [2206.14176](https://arxiv.org/abs/2206.14176) | 真实物理系统的世界模型，连续动作 |
| Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis", ICML 2022 | [2205.09991](https://arxiv.org/abs/2205.09991) | Diffuser，扩散模型做规划，可作为我们生成路径用于决策的理论参考 |

---

## 三、时序生成的扩散/流匹配方法（V3 的复现基础）

### 3.1 直接复现来源

| 文献 | arXiv | 关系 |
|---|---|---|
| Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", ICML 2021 (TimeGrad) | [2101.12072](https://arxiv.org/abs/2101.12072) | **关键先例**。用扩散做时序自回归预测——V3 的自回归滚动思路直接借鉴 |
| Cheng et al., "TimeFlow: Time Series is Not All You Need", 2025 | [2511.07968](https://arxiv.org/abs/2511.07968) | FM 在通用时序上的应用，V3 单阶段 FM 基线的直接参考 |
| Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", NeurIPS 2021 | [2107.03502](https://arxiv.org/abs/2107.03502) | 条件 Score-based 时序生成，V3 的条件 FM 注入方法参考 |
| Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", NeurIPS 2023 (TSDiff) | [2307.11494](https://arxiv.org/abs/2307.11494) | 时序扩散预测最新工作 |

### 3.2 时序生成相关工作

| 文献 | arXiv | 关系 |
|---|---|---|
| Yoon et al., "Time-series Generative Adversarial Networks", NeurIPS 2019 (TimeGAN) | NeurIPS 2019 | GAN 时序生成，老 baseline |
| Alcaraz & Strodthoff, "Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models", ICLR 2023 (SSSD) | [2208.09399](https://arxiv.org/abs/2208.09399) | 结构化 state-space + 扩散，与 V3 的"结构先验"思路同源 |
| Yuan et al., "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", ICLR 2024 | [2403.01742](https://arxiv.org/abs/2403.01742) | 时序扩散 + 趋势/季节分解，分层思路类似 |
| Shen et al., "Non-autoregressive Conditional Diffusion Models for Time Series Prediction", ICML 2023 (TimeDiff) | [2306.05043](https://arxiv.org/abs/2306.05043) | 时序扩散预测 |
| Lim et al., "TSDiff: Score-based Diffusion Models for Probabilistic Time Series Forecasting", 2023 | — | 综述类参考 |

---

## 四、金融建模与数据基础

### 4.1 Heston 模型与数值方法

| 文献 | 发表 | 关系 |
|---|---|---|
| Heston, "A Closed-form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options", Review of Financial Studies 1993 | RFS 1993 | **数据生成的理论来源**。Heston 原始论文，含特征函数推导（eq. 17-18）|
| Andersen, "Efficient Simulation of the Heston Model", J. Computational Finance 2007 | JCF 2007 | **数值格式的直接来源**。QE scheme，工业标准，避免 Euler-Maruyama 负方差 |
| Carr & Madan, "Option Valuation Using the Fast Fourier Transform", J. Computational Finance 1999 | JCF 1999 | **期权定价 ground truth 的算法**。Carr-Madan FFT |
| Cox, Ingersoll, Ross, "A Theory of the Term Structure of Interest Rates", Econometrica 1985 | Econometrica 1985 | CIR 过程的奠基性论文（Heston 中的 $v_t$ 即 CIR）|
| Gatheral et al., "Volatility is Rough", Quantitative Finance 2018 | [1410.3394](https://arxiv.org/abs/1410.3394) | rough volatility 实证，V3 的扩展方向 |

### 4.2 神经网络做 Heston 路径上的工作

| 文献 | arXiv | 关系 |
|---|---|---|
| Horvath et al., "Deep Learning Volatility", Quantitative Finance 2021 | [1901.09647](https://arxiv.org/abs/1901.09647) | NN 加速 Heston/rough 校准，与 V3 不同（他们做 calibration，我们做 generation）但同领域参考 |
| Bayer et al., "Deep Calibration of Rough Stochastic Volatility Models", JCF 2021 | [1810.03399](https://arxiv.org/abs/1810.03399) | 深度校准范式，方法论先例 |
| Ruf & Wang, "Neural Networks for Option Pricing and Hedging: A Literature Review", JCF 2020 | [1911.05620](https://arxiv.org/abs/1911.05620) | 综述：NN 做期权定价的各种方法 |
| Buehler et al., "Deep Hedging", Quantitative Finance 2019 | [1802.03042](https://arxiv.org/abs/1802.03042) | RL + NN 做期权对冲 |

### 4.3 金融路径生成（GAN/Diffusion）

| 文献 | arXiv/年份 | 关系 |
|---|---|---|
| Wiese et al., "Quant GANs: Deep Generation of Financial Time Series", Quantitative Finance 2020 | [1907.06673](https://arxiv.org/abs/1907.06673) | **GAN baseline**。Heston 路径上的 GAN 生成，我们的对比对象 |
| Ni et al., "Sig-Wasserstein GANs for Time Series Generation", 2021 | [2111.01207](https://arxiv.org/abs/2111.01207) | Signature + GAN，路径评测参考 |
| Kidger et al., "Neural SDEs as Infinite-Dimensional GANs", ICML 2021 | [2102.03657](https://arxiv.org/abs/2102.03657) | Neural SDE + GAN，金融路径生成的先例 |
| Li et al., "Scalable Gradients for Stochastic Differential Equations", AISTATS 2020 (Latent SDE) | [2001.01328](https://arxiv.org/abs/2001.01328) | **Latent SDE，隐变量 SDE 生成的直接先例**。V3 的两阶段 v_t→r_t 思路与之同源 |
| Kim et al., "Diffusion Models for Financial Time Series" | 2024 | 金融 diffusion 的最新工作（具体题目检索时确认）|

### 4.4 金融时序的统计性质（评测依据）

| 文献 | 年份 | 关系 |
|---|---|---|
| Cont, "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues", Quantitative Finance 2001 | QF 2001 | **5 个 stylized facts 的权威来源**。V3 的统计评测指标直接来自此 |
| Mandelbrot, "The Variation of Certain Speculative Prices", J. Business 1963 | JB 1963 | 厚尾、波动率聚集的早期实证 |
| Engle, "Autoregressive Conditional Heteroscedasticity", Econometrica 1982 | Econometrica 1982 | ARCH 模型，volatility clustering 的统计起点 |

---

## 五、自回归生成与误差累积（V3 的关键技术难点）

V3 自回归滚动 252 步，单步误差会累积。这一节是关于"如何控制累积"的参考。

| 文献 | arXiv | 关系 |
|---|---|---|
| Bengio et al., "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks", NeurIPS 2015 | [1506.03099](https://arxiv.org/abs/1506.03099) | **Exposure bias 的奠基性分析**。V3 误差累积的理论起点 |
| Lamb et al., "Professor Forcing: A New Algorithm for Training Recurrent Networks", NeurIPS 2016 | [1610.09038](https://arxiv.org/abs/1610.09038) | 用对抗训练对齐 teacher/student 分布，缓解 exposure bias |
| Ranzato et al., "Sequence Level Training with Recurrent Neural Networks", ICLR 2016 | [1511.06732](https://arxiv.org/abs/1511.06732) | MIXER，序列级训练缓解 exposure bias |
| Brandfonbrener et al., "When Does Return-Conditioned Supervised Learning Work for Offline RL?", NeurIPS 2022 | [2206.01079](https://arxiv.org/abs/2206.01079) | 自回归 rollout 的稳定性分析（offline RL 视角）|

**待补充**：V3 实施时需要找扩散模型自回归 rollout 的稳定性专门工作（如有）。已知 DIAMOND 和 TimeGrad 都做了 rollout 但未深入分析误差累积。

---

## 六、条件生成方法（V3 中条件注入的实现参考）

| 文献 | arXiv | 关系 |
|---|---|---|
| Ho & Salimans, "Classifier-Free Diffusion Guidance", NeurIPS 2022 workshop | [2207.12598](https://arxiv.org/abs/2207.12598) | **CFG**，条件 diffusion 的标准做法。V3 中"动作 $a_t$"的注入可以用 CFG |
| Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018 | [1709.07871](https://arxiv.org/abs/1709.07871) | **FiLM**，特征级条件调制，V3 中状态 $(v_t, r_t)$ 的注入方式之一 |
| Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021 | [2105.05233](https://arxiv.org/abs/2105.05233) | Classifier guidance，CFG 的前身 |
| Zheng et al., "Guided Flows for Generative Modeling and Decision Making", 2023 | [2311.13443](https://arxiv.org/abs/2311.13443) | FM 上的 CFG 等价物，V3 可直接用 |

---

## 七、理论基础（Report 中的 intellectual contribution）

### 7.1 Girsanov 定理 与生成模型测度变换

| 文献 | 关系 |
|---|---|
| Karatzas & Shreve, "Brownian Motion and Stochastic Calculus", Springer 1991 | Girsanov 定理的标准教科书参考 |
| Øksendal, "Stochastic Differential Equations", Springer 6th ed. 2003 | SDE 教科书，第8章测度变换 |
| Shreve, "Stochastic Calculus for Finance II: Continuous-Time Models", Springer 2004 | 金融数学中的 Girsanov（P→Q 测度）|
| Anderson, "Reverse-time Diffusion Equation Models", Stochastic Processes and their Applications 1982 | Anderson 反向 SDE 公式，Score SDE 的理论起点 |

### 7.2 概率流 ODE 与等价性

| 文献 | arXiv | 关系 |
|---|---|---|
| Song et al., "Score-Based Generative Modeling through SDEs", ICLR 2021 | [2011.13456](https://arxiv.org/abs/2011.13456) | Probability Flow ODE（eq. 13），FM 与 Score SDE 等价性的基础 |
| Maoutsa et al., "Interacting Particle Solutions of Fokker-Planck Equations through Gradient-Log-Density Estimation", Entropy 2020 | [2006.00702](https://arxiv.org/abs/2006.00702) | 概率流 ODE 的物理学视角 |

---

## 八、网络架构（V3 骨干）

| 文献 | arXiv | 关系 |
|---|---|---|
| Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015 | [1505.04597](https://arxiv.org/abs/1505.04597) | U-Net 原始论文 |
| Ho et al., "DDPM", NeurIPS 2020 | [2006.11239](https://arxiv.org/abs/2006.11239) | U-Net 在扩散模型中的标准化实现（time embedding 等）|
| Kong et al., "DiffWave: A Versatile Diffusion Model for Audio Synthesis", ICLR 2021 | [2009.09893](https://arxiv.org/abs/2009.09893) | **1D U-Net 用于音频生成**。V3 骨干网络最直接的非金融参考 |
| Vaswani et al., "Attention is All You Need", NeurIPS 2017 | [1706.03762](https://arxiv.org/abs/1706.03762) | Transformer，bottleneck self-attention 用 |
| Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023 (DiT) | [2212.09748](https://arxiv.org/abs/2212.09748) | DiT，可选骨干替代方案 |

---

## 九、JVP 与高阶导数实现（Mean Flow 训练核心难点）

| 资源 | 关系 |
|---|---|
| PyTorch 官方文档：[`torch.func.jvp`](https://pytorch.org/docs/stable/generated/torch.func.jvp.html) | Mean Flow 训练用的 JVP API |
| PyTorch 官方文档：[`torch.func` overview](https://pytorch.org/docs/stable/func.html) | functorch（已合并到 torch.func）的 functional API |
| Bradbury et al., "JAX: Composable Transformations of Python+NumPy Programs", 2018 | JAX 的 jvp 实现，理论参考 |

---

## 十、需要补充检索的方向（V3 实施时再深入）

以下方向 V3 实施时需要专门检索，目前未列入：

1. **扩散模型/FM 的自回归 rollout 稳定性**：除了 TimeGrad 和 DIAMOND，是否有专门分析长 horizon rollout 误差累积的工作
2. **金融数据上的世界模型**：截止 2026 年是否有人做过 financial world model（直觉是没有，但需确认；可能有金融 RL 中用到 world model 的工作）
3. **隐变量 SDE 的神经生成模型最新进展**：Latent SDE 之后是否有跟进工作
4. **Heston 类模型的神经网络模拟器**：是否有用神经网络做 Heston 路径模拟器的工作（不是 calibration）
5. **Time-conditional Mean Flow / Consistency models**：Mean Flow / CM 用于条件生成的最新工作

---

## 十一、对话中明确出现过的文献汇总（去重后清单）

按对话历史出现顺序整理，便于回溯：

1. Heston (1993), RFS — Heston 模型
2. Andersen (2007), JCF — QE scheme
3. Carr & Madan (1999), JCF — FFT 期权定价
4. Cont (2001), QF — Stylized facts
5. Gatheral et al. (2018), QF — Rough volatility
6. Song et al. (2021), ICLR — Score SDE
7. Ho et al. (2020), NeurIPS — DDPM
8. Lu et al. (2022), NeurIPS — DPM-Solver++
9. Vincent (2011) — DSM 理论
10. Lipman et al. (2023), ICLR — Flow Matching
11. Albergo & Vanden-Eijnden (2023), ICLR — Stochastic Interpolants
12. Liu et al. (2022), ICLR — Rectified Flow
13. Song et al. (2023), ICML — Consistency Models
14. Song et al. (2024), NeurIPS — Improved Consistency Training
15. Geng et al. (2025), NeurIPS Oral — Mean Flow
16. Ho & Salimans (2022), NeurIPS wksp — Classifier-Free Guidance
17. Perez et al. (2018), AAAI — FiLM
18. Ronneberger et al. (2015), MICCAI — U-Net
19. Kong et al. (2021), ICLR — DiffWave
20. Rasul et al. (2021), ICML — TimeGrad
21. Tashiro et al. (2021), NeurIPS — CSDI
22. Cheng et al. (2025) — TimeFlow
23. Yuan et al. (2024), ICLR — Diffusion-TS
24. Shen et al. (2023) — TimeDiff
25. Alcaraz & Strodthoff (2023), ICLR — SSSD
26. Wiese et al. (2020), QF — Quant GANs
27. Kidger et al. (2021), ICML — Neural SDEs as Infinite-Dim GANs
28. Li et al. (2020), AISTATS — Latent SDE
29. Ni et al. (2021) — Sig-Wasserstein GAN
30. Alonso et al. (2024), NeurIPS — DIAMOND
31. Ho et al. (2022), NeurIPS — Cascaded Diffusion Models
32. Rombach et al. (2022), CVPR — Latent Diffusion
33. Ramesh et al. (2022) — DALL-E 2
34. Horvath et al. (2021), QF — Deep Learning Volatility
35. Bayer et al. (2021), JCF — Deep Rough Volatility Calibration
36. Ruf & Wang (2020), JCF — NN Option Pricing Review
37. Buehler et al. (2019), QF — Deep Hedging

**对话中提到但需补充的（额外检索）**：

- Ha & Schmidhuber (2018) — World Models 奠基
- Hafner et al. (Dreamer 系列) — RL 世界模型谱系
- Micheli et al. (2023), IRIS — Transformer 世界模型
- Robine et al. (2023), TWM — Transformer 世界模型
- Valevski et al. (2024), GameNGen — 实时扩散世界模型
- Bruce et al. (2024), Genie — 大规模交互式生成环境
- Janner et al. (2022), ICML — Diffuser
- Bengio et al. (2015), NeurIPS — Scheduled Sampling（误差累积理论）
- Lamb et al. (2016), NeurIPS — Professor Forcing
- Karras et al. (2022), NeurIPS — EDM
- Tong et al. (2024), ICML — OT-CFM
- Zheng et al. (2023) — Guided Flows
- Peebles & Xie (2023), ICCV — DiT
- Janner et al. (2022) — Planning with Diffusion
- Kollovieh et al. (2023), NeurIPS — TSDiff

---

## 十二、引用结构建议（Report 中如何组织 Related Work）

按 V3 项目的逻辑结构组织 4 页 Report 的 Related Work 节：

```
Related Work（建议结构）

§ 2.1 World Models
  - Ha & Schmidhuber 2018（开山）
  - Dreamer V1/V2/V3（latent dynamics）
  - DIAMOND, GameNGen（扩散模型作为世界模型，与本文直接对标）
  - IRIS, TWM（Transformer 路线，对比）

§ 2.2 Generative Models for Time Series
  - TimeGAN, Quant GANs（GAN 路线）
  - TimeGrad, CSDI, SSSD, Diffusion-TS（扩散路线）
  - TimeFlow（FM 路线，最直接对标）
  - Latent SDE, Neural SDEs as GANs（SDE 路线）

§ 2.3 One-step Generative Models（蒸馏）
  - Consistency Models, iCT
  - Mean Flow（核心创新依据）
  - Score Distillation

§ 2.4 Stochastic Volatility Modeling
  - Heston 1993, Andersen QE
  - Deep Learning Volatility（NN 做校准）
  - Rough volatility（扩展方向）
```

---

## 注意事项

1. 标记为"**课程参考论文**"的（DIAMOND、Mean Flow、Consistency Models、Flow Matching、Score SDE）是课程必须覆盖的，Report 中要显式引用并说明对应关系
2. **TimeFlow 的 arXiv 编号需要在实施时确认**（对话中标的 2511.07968 在 2025-11 之前可能尚未发布，需检索确认）
3. **Geng et al. (Mean Flow) 的 arXiv 编号需要确认**（NeurIPS 2025 Oral，正式 arXiv 号在实施时检索）
4. 金融部分文献偏多是因为数据基础需要支撑，但 Report 重心应在生成模型方法学上——金融文献仅作数据合理性说明
5. 实施时新发现的关键文献追加到本文档，作为整个项目的文献基线
