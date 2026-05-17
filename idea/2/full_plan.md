# FinFlow Bench — 完整项目实施方案

> 生成日期：2026-05-13。基于 discussion.md 的评估与修订。

---

## 零、对 discussion.md 的评估

### 合理的部分（保留）
1. **方向判断正确**：FlowLOB 确实是 red ocean（TRADES/LOBDIF/Painting-the-Market 已饱和），换到"方法学对比 + Heston testbed"的 framing 是正确决策
2. **核心技术选型正确**：Score SDE / FM / Consistency / Mean Flow 四个模型，共用 1D U-Net backbone，公平对比
3. **Mean Flow 作为创新点正确**：Geng 2025 NeurIPS Oral，在金融 1D 时序上确实无人做过
4. **Andersen QE 用于 Heston 路径生成**：正确，避免简单 Euler-Maruyama 导致方差负值问题
5. **Carr-Madan FFT 做期权定价 ground truth**：正确，利用 Heston 特征函数半解析性质
6. **可视化设计（5个piece）**：思路好，实际上是竞争优势
7. **算力估算**：单卡 RTX 3090，~30-40 小时，合理

### 需要修订的部分
1. **课程合规性评估偏乐观**：discussion 说 85%，实际偏低。需要 TA 书面确认，且 framing 要用"方法学对比，Heston 作可控测试床"而非"金融应用"
2. **Mean Flow 的 JVP 实现被低估**：`torch.func.jvp` 在实践中有梯度计算图泄漏问题，需要 `torch.func.grad` + `functorch` 的正确用法，不是"100行可 port"那么简单
3. **Consistency Training 调参被低估**：EMA decay、噪声 schedule、discretization steps 都是关键超参，训练不稳定的概率比 discussion 承认的高
4. **文献引用体系不完整**：related work 节的文献链条需要补充，尤其是非金融领域的方法论 reference
5. **三周时间线过紧**：现在是 5/13，离 6/7 还有约 3.5 周，按 discussion 的节奏勉强可行。但 Mean Flow JVP 问题很可能让 W2 拖延，建议在 W3 前设 go/no-go 检查点

---

## 一、项目定义

### 1.1 标题（两种 framing）

**向 TA 报备的标题（低风险）**：
> Benchmarking Modern Generative Paradigms on Controlled Stochastic Time Series: Score SDE, Flow Matching, Consistency Models, and Mean Flow

**对外（简历/答辩）的标题**：
> FinFlow Bench: Systematic Evaluation of One-step and Multi-step Generative Models for Financial Path Synthesis and Derivative Pricing

### 1.2 一句话定义

在合成 Heston SDE 路径（闭合解存在，可做精确评测）上，系统比较 4 种课程核心生成范式，创新点是把 Mean Flow（NeurIPS 2025 Oral）首次应用于 1D 随机过程数据，并用毫秒级期权定价作为下游 task。

### 1.3 课程符合性论证

| 课程方向 | 本项目对应 |
|---|---|
| AIGC 相关方法 | Score SDE / Flow Matching / Consistency / Mean Flow 全部来自课程参考论文 |
| 复现经典工作 | Score SDE (Song 2021) + Flow Matching (Lipman 2023) 在时间序列上的完整复现 |
| 创新改进 | Mean Flow 首次应用于 1D 随机过程；JVP 稳定性研究；精确 closed-form 评测框架 |
| 评测有效性 | Heston 特征函数提供精确 ground truth，比图像 FID 更严格 |

---

## 二、Step 1：数据模块——Heston 路径生成

### 2.1 理论基础

**Heston 模型**（Heston 1993）：

$$dS_t = \mu S_t \, dt + \sqrt{v_t} \, S_t \, dW^S_t$$
$$dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW^v_t$$
$$d\langle W^S, W^v \rangle_t = \rho \, dt$$

其中：
- $\kappa$：均值回归速度
- $\theta$：长期方差均值
- $\xi$：vol of vol
- $\rho$：价格与波动率的相关系数（一般为负，"leverage effect"）
- Feller 条件：$2\kappa\theta > \xi^2$ 保证 $v_t > 0$ a.s.

**训练用参数**（Heston 1993 基准值）：
```
κ = 2.0, θ = 0.04, ξ = 0.3, ρ = -0.7
v₀ = 0.04, S₀ = 100, μ = 0.05
T = 1 年, Δt = 1/252（日频），步数 d = 252
```

### 2.2 数值格式：Andersen QE 格式（不是 Euler）

**为什么不用 Euler-Maruyama**：简单 EM 格式会让 $v_t$ 跑到负值（当 $\xi$ 较大时），需要截断 `max(v_t, 0)`，引入系统性偏差。

**QE（Quadratic Exponential）格式**（Andersen 2007）——是 Heston 路径模拟的工业标准：

对方差过程 $v_t$：
1. 计算条件均值 $m = \theta + (v_t - \theta)e^{-\kappa\Delta t}$ 和方差 $s^2 = \frac{v_t \xi^2 e^{-\kappa\Delta t}}{\kappa}(1 - e^{-\kappa\Delta t}) + \frac{\theta\xi^2}{2\kappa}(1 - e^{-\kappa\Delta t})^2$
2. 计算 $\psi = s^2/m^2$
3. 若 $\psi \leq 2$（"quadratic"区域）：用近似高斯；若 $\psi > 2$（"exponential"区域）：用指数分布近似
4. 保证 $v_{t+\Delta t} \geq 0$ 精确成立

对价格过程 $S_t$（给定 $v_t$）：
$$\log S_{t+\Delta t} = \log S_t + (\mu - v_t/2)\Delta t + \sqrt{v_t \Delta t} Z^S$$
其中 $Z^S = \rho Z^v + \sqrt{1-\rho^2} Z^\perp$，$Z^v, Z^\perp \sim \mathcal{N}(0,1)$ 独立

**模型输入**：log-return 序列（平稳化）：
$$r_t = \log(S_{t+1}/S_t) \in \mathbb{R}, \quad t = 0, \ldots, 251$$
最终 shape：$x_0 \in \mathbb{R}^{252}$

### 2.3 期权定价 Ground Truth：Carr-Madan FFT

Heston 模型的特征函数（Heston 1993，特征函数半解析形式）：

$$\phi(u) = \mathbb{E}^{\mathbb{Q}}[e^{iu \log S_T}] = \exp(C(u,T) + D(u,T) v_0 + iu \log S_0)$$

其中 $C$、$D$ 有闭合表达式（Heston 原论文 eq. 17-18）。

**Carr-Madan (1999) FFT 定价**：

欧式看涨期权价格：
$$C(k) = \frac{e^{-\alpha k}}{\pi} \int_0^\infty e^{-iuk} \psi(u) \, du$$

其中 $k = \log(K/S_0)$，$\psi(u) = \frac{e^{-rT} \phi(u - i(\alpha+1))}{\alpha^2 + \alpha - u^2 + i(2\alpha+1)u}$

用 FFT 一次性计算 $N$ 个 strike 的价格，效率 $O(N \log N)$。

**测试 grid**：

| Strike K | Maturity T |
|---|---|
| 80, 90, 100, 110, 120 | 0.25, 0.5, 1.0 |

共 15 个 (K, T) 点，每个点有精确的 Heston 期权价格作为 ground truth。

### 2.4 数据规模

```
训练集：N_train = 50,000 条路径（每条 252 步）
验证集：N_val = 5,000 条
测试集：N_test = 10,000 条
Option GT：15 个 (K,T) 点 × Carr-Madan 精确值
存储：~500 MB（.npy 格式）
```

### 2.5 需要做的事

```
Person A（数据）：
  1. 实现 Andersen QE 格式，~150 行 Python
  2. 生成 55k 条路径，存成 .npy
  3. 实现 Carr-Madan FFT 定价，~80 行 Python
  4. 计算 15 个 (K,T) 点的精确期权价格，存成表格
  5. 实现数据 normalization（z-score on log-returns）

关键库：numpy, scipy.stats（已知标准实现）
```

### 2.6 算力需求 & 难度

| 子任务 | 算力需求 | 难度 | 预估时间 |
|---|---|---|---|
| Heston QE 实现 | CPU 即可 | ★★★☆☆ | 1天 |
| 路径生成（55k条）| CPU，~20分钟 | ★☆☆☆☆ | 0.5天 |
| Carr-Madan FFT | CPU，<1分钟 | ★★★☆☆ | 0.5天 |
| 数据管道搭建 | CPU | ★★☆☆☆ | 0.5天 |

**关键文献**：
- Andersen (2007), "Efficient Simulation of the Heston Model", J. Computational Finance
- Carr & Madan (1999), "Option Valuation Using the Fast Fourier Transform", J. Computational Finance
- Heston (1993), "A Closed-form Solution for Options with Stochastic Volatility", Review of Financial Studies

---

## 三、Step 2：骨干网络——1D U-Net

### 3.1 架构设计

**为什么用 1D U-Net 而不是 Transformer**：
- 输入序列长度 $d=252$，不算长，位置编码对性能影响有限
- 1D Conv 可以捕捉局部时序结构（volatility clustering 是局部现象）
- 训练速度比 attention 快 3-5x，适合多次迭代实验

**架构规格**（所有 4 个模型共享）：

```python
Input:  [B, 1, 252]   (batch, channels, sequence_length)
        ↓
Downsample blocks (4级):
  Conv1D(1 → 64, k=3) + GroupNorm + SiLU
  Conv1D(64 → 128, k=3) + GroupNorm + SiLU  [stride=2 → 126]
  Conv1D(128 → 256, k=3) + GroupNorm + SiLU [stride=2 → 63]
  Conv1D(256 → 512, k=3) + GroupNorm + SiLU [stride=2 → 32]

Bottleneck:
  Conv1D(512 → 512) + Self-Attention (4 heads) + GroupNorm

Upsample blocks (4级，skip connections from downsample):
  ConvTranspose1D + GroupNorm + SiLU（对应还原）

Time embedding:
  t → sinusoidal(256) → MLP(256→512→512) → added to each block

Output: [B, 1, 252]   通过 1x1 conv 映射到目标维度

参数量：~2M（轻量），~8M（完整版，推荐）
```

**条件输入（可选扩展）**：
若需要条件生成（给定 vol level 生成路径），把条件向量 concat 到 time embedding 上即可，不改变主结构。

### 3.2 需要做的事

```
Person B（模型）：
  1. 实现 1D U-Net，包含 time embedding
  2. 单元测试：随机输入 → 正确 shape 输出
  3. 验证 forward pass 梯度正确流动
  4. 确认训练内存：batch_size=256 在 24GB 单卡的显存占用

关键库：torch, torch.nn
```

### 3.3 算力需求 & 难度

| 子任务 | 算力需求 | 难度 |
|---|---|---|
| U-Net 实现 | 无需 GPU | ★★☆☆☆ |
| 调试 forward pass | CPU/GPU | ★★☆☆☆ |
| 显存 profiling | 单卡 3090 | ★☆☆☆☆ |

**关键文献（非金融 backbone 参考）**：
- Ronneberger et al. (2015), "U-Net", MICCAI — 原始 U-Net（2D 图像）
- Ho et al. (2020), "DDPM", NeurIPS — U-Net 用于扩散模型的标准化实现
- Kong et al. (2021), "DiffWave", ICLR — 1D U-Net 用于音频生成（最直接的非金融参考）

---

## 四、Step 3：模型 1——Score SDE（VP-SDE）

### 4.1 理论

**前向过程（加噪 SDE）**：

$$dx_t = -\frac{1}{2}\beta(t) x_t \, dt + \sqrt{\beta(t)} \, dW_t, \quad t \in [0,1]$$

线性 schedule：$\beta(t) = \beta_{\min} + (\beta_{\max} - \beta_{\min})t$，取 $\beta_{\min} = 0.1, \beta_{\max} = 20$。

这个 SDE 的转移核是高斯的：
$$p_{t|0}(x_t | x_0) = \mathcal{N}(x_t; \alpha(t) x_0, \sigma(t)^2 I)$$

其中 $\alpha(t) = e^{-\frac{1}{2}\int_0^t \beta(s)ds}$，$\sigma(t)^2 = 1 - \alpha(t)^2$。

**Score Matching 训练目标（Denoising Score Matching）**：

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \lambda(t) \left\| s_\theta(x_t, t) + \frac{\epsilon}{\sigma(t)} \right\|^2 \right]$$

其中 $x_t = \alpha(t) x_0 + \sigma(t) \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$，$\lambda(t) = \sigma(t)^2$（Song 2021 推荐的权重）。

等价地，网络预测 $\epsilon$ 而非 score（DDPM 参数化）：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon_\theta(x_t, t) - \epsilon \right\|^2 \right]$$

**反向采样（生成过程）**：

反向 SDE（Anderson 1982）：
$$dx_t = \left[-\frac{1}{2}\beta(t) x_t - \beta(t) \nabla_{x_t} \log p_t(x_t)\right] dt + \sqrt{\beta(t)} \, d\bar{W}_t$$

实践中用 **DPM-Solver++**（50步）或概率流 ODE（确定性采样）代替 EM，速度更快。

### 4.2 实现要点

```python
# 训练一步
t = torch.rand(B) * (1 - 1e-5) + 1e-5  # 均匀采样 t
alpha_t = compute_alpha(t)  # [B]
sigma_t = compute_sigma(t)  # [B]
eps = torch.randn_like(x0)
x_t = alpha_t[:,None] * x0 + sigma_t[:,None] * eps
eps_pred = model(x_t, t)  # U-Net 预测噪声
loss = F.mse_loss(eps_pred, eps)

# 采样：DPM-Solver++（从 diffusers 库直接用）
from diffusers import DPMSolverMultistepScheduler
scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(50)
x = torch.randn(N, d)
for t in scheduler.timesteps:
    with torch.no_grad():
        eps_pred = model(x, t)
    x = scheduler.step(eps_pred, t, x).prev_sample
```

### 4.3 超参数

```
Optimizer: AdamW, lr=2e-4, weight_decay=1e-4
Scheduler: cosine annealing
Batch size: 512
Epochs: 500 (约 50k steps)
EMA decay: 0.9999
NFE (inference): 50 (DPM-Solver++)
```

### 4.4 需要做的事

```
Person B：
  1. 实现 noise schedule（α_t, σ_t 的计算）
  2. 实现训练 loop（包含 EMA）
  3. 配置 DPM-Solver++ 采样器（diffusers 库）
  4. 验证：生成路径的均值/方差接近 Heston 真实值
```

### 4.5 算力需求 & 难度

| 项目 | 规格 |
|---|---|
| GPU | 单卡 RTX 3090（24GB），显存约 4GB |
| 训练时间 | ~1小时（500 epoch，batch=512）|
| 难度 | ★★☆☆☆（有 diffusers 库支持）|

**关键文献**：
- Song et al. (2021), "Score-Based Generative Modeling through SDEs", ICLR — `arXiv:2011.13456`（课程参考论文）
- Ho et al. (2020), "DDPM", NeurIPS — `arXiv:2006.11239`（Score SDE 的 DDPM 特例）
- Lu et al. (2022), "DPM-Solver++", NeurIPS — `arXiv:2211.01095`（采样加速）
- Vincent (2011), "Connection Between Score Matching and Denoising Autoencoders" — DSM 的理论基础

**非金融同类工作**：
- Rasul et al. (2021), "TimeGrad", ICML — `arXiv:2101.12072`（Score SDE 做时序预测的首篇重要工作）
- Tashiro et al. (2021), "CSDI", NeurIPS — `arXiv:2107.03502`（条件 score-based 时序生成）

---

## 五、Step 4：模型 2——Flow Matching

### 5.1 理论

**核心思路（Lipman et al. 2023）**：直接学习一个把噪声分布 $p_0 = \mathcal{N}(0, I)$ 映射到数据分布 $p_1$ 的 ODE 的速度场 $v_\theta$：

$$\frac{dx}{dt} = v_\theta(x_t, t)$$

**Conditional Flow Matching (CFM)**：

条件概率路径（直线插值）：
$$x_t = (1 - (1-\sigma_{\min})t) \cdot \epsilon + t \cdot x_1, \quad \epsilon \sim \mathcal{N}(0,I)$$

条件速度场（直线路径的切向量）：
$$u_t(x_t | x_1) = \frac{x_1 - (1-\sigma_{\min}) x_t - x_1 \cdot 0}{1 - (1-\sigma_{\min})t}$$

简化后：
$$u_t(x | x_1) = \frac{x_1 - (1-\sigma_{\min}) x}{1 - (1-\sigma_{\min})t}$$

**CFM Loss**（无需知道边际速度场，仅用条件速度场）：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_1, x_t} \left[ \| v_\theta(x_t, t) - u_t(x_t | x_1) \|^2 \right]$$

**训练**：采样 $t \sim U[0,1]$，$x_1 \sim p_{\text{data}}$，$\epsilon \sim \mathcal{N}(0,I)$，计算 $x_t$，计算目标 $u_t$，计算 MSE loss。

**采样**：Euler 或 Runge-Kutta 4 ODE solver，50步。

### 5.2 与 Score SDE 的关系

Flow Matching ODE 是 Score SDE 的概率流 ODE（Probability Flow ODE，Song 2021 eq. 13）在直线路径下的特例。两者学的是同一个目标分布，但 FM 的路径是直线（更短），所以 NFE 可以更少。

**数学连接**：
$$v_\theta(x, t) = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$
其中 $f, g$ 是 SDE 的 drift/diffusion。FM 等价于用直线路径参数化这个关系。

### 5.3 实现要点

```python
sigma_min = 1e-4

# 训练一步
t = torch.rand(B)
x1 = x0_batch  # 真实数据
eps = torch.randn_like(x1)
x_t = (1 - (1-sigma_min)*t[:,None]) * eps + t[:,None] * x1
target = x1 - (1-sigma_min) * eps  # 分子
denom = 1 - (1-sigma_min)*t[:,None]  # 分母
u_t = target / denom
v_pred = model(x_t, t)
loss = F.mse_loss(v_pred, u_t)

# 采样：固定步数 Euler ODE
dt = 1.0 / N_steps
x = torch.randn(N, d)
for i in range(N_steps):
    t = torch.full((N,), i / N_steps)
    with torch.no_grad():
        v = model(x, t)
    x = x + v * dt
```

### 5.4 算力需求 & 难度

| 项目 | 规格 |
|---|---|
| GPU | 单卡 RTX 3090，显存 ~3GB |
| 训练时间 | ~45分钟（比 Score SDE 快，收敛更稳）|
| 难度 | ★★☆☆☆（原理清晰，loss 简单）|

**关键文献**：
- Lipman et al. (2023), "Flow Matching for Generative Modeling", ICLR — `arXiv:2210.02747`（课程参考论文）
- Albergo & Vanden-Eijnden (2023), "Stochastic Interpolants: A Unifying Framework", ICLR — `arXiv:2303.08797`
- Liu et al. (2022), "Flow Straight and Fast: Rectified Flow", ICLR — `arXiv:2209.03003`（FM 的等价 rectified flow 视角）

**非金融同类工作**：
- Cheng et al. (2025), "TimeFlow: Time Series is Not All You Need", ICLR 2026 submission — `arXiv:2511.07968`（FM 在通用时序上，我们的直接竞品/参考）

---

## 六、Step 5：模型 3——Consistency Training

### 6.1 理论

**动机**：Score SDE 和 FM 推理需要 50+ 步。Consistency Models 训练一个"一步到位"的函数。

**Consistency Function**（Song et al. 2023）：
$$f_\theta: \mathbb{R}^d \times [\epsilon, T] \to \mathbb{R}^d$$

满足**自洽性**（consistency property）：对同一条 ODE 轨迹上的任意两点 $(x_t, t)$ 和 $(x_s, s)$：
$$f_\theta(x_t, t) = f_\theta(x_s, s) \quad \forall t, s \in [\epsilon, T]$$

即：无论从哪个噪声水平出发，映射到 $x_0$ 的结果相同。

**Boundary condition**：$f_\theta(x, \epsilon) = x$（$\epsilon$ 很小时不做任何处理）。

**实现方式（参数化）**：
$$f_\theta(x, t) = c_{\text{skip}}(t) x + c_{\text{out}}(t) F_\theta(x, t)$$
其中 $c_{\text{skip}}(t) \to 1$，$c_{\text{out}}(t) \to 0$ 当 $t \to \epsilon$，满足边界条件。

**Consistency Training（CT）Loss**（不需要预训练的 teacher）：

$$\mathcal{L}_{\text{CT}} = \mathbb{E}_{n, x, z} \left[ d\left(f_\theta(x + \sigma_{n+1} z, \sigma_{n+1}),\; f_{\theta^-}(x + \sigma_n z, \sigma_n)\right) \right]$$

其中：
- $n$ 从离散时间步中采样
- $z \sim \mathcal{N}(0, I)$
- $d(\cdot, \cdot)$ 是距离函数（LPIPS 用于图像，MSE 用于我们的 1D 数据）
- $\theta^-$ 是 EMA 参数（stopped gradient）

**采样**：NFE = 1：直接 $f_\theta(x_T, T) \to x_0$；NFE = 2：两步精修。

### 6.2 关键超参（容易训练失败的地方）

```
EMA decay μ: 从 0 开始逐步增大（CT 论文的 adaptive schedule）
  μ_n = exp(s_0 * log(μ_0) / n)，n 是总训练步数
离散化步数 N: 从 2 逐渐增加到 150（curriculum）
学习率: 1e-4（比 Score SDE/FM 小）
Batch size: 512
```

**已知坑**：EMA decay 太大 → 训练不稳定（梯度爆炸）；太小 → 收敛慢。Song 2023 论文的 appendix 有详细 schedule，必须照抄。

### 6.3 算力需求 & 难度

| 项目 | 规格 |
|---|---|
| GPU | 单卡 RTX 3090，显存 ~5GB（EMA 需要额外副本）|
| 训练时间 | ~1.5小时（EMA 收敛慢）|
| 难度 | ★★★☆☆（超参敏感，CT schedule 必须精确）|

**关键文献**：
- Song et al. (2023), "Consistency Models", ICML — `arXiv:2303.01469`（课程参考论文）
- Song et al. (2024), "Improved Consistency Training", NeurIPS — `arXiv:2310.14189`（CT 的改进版，超参更稳）

**非金融同类工作**：
- 无直接对标（CM 在音频/视频上的应用极少），这是我们的机会

---

## 七、Step 6：模型 4——Mean Flow（核心创新）

### 7.1 理论

**动机（Geng et al., NeurIPS 2025 Oral）**：Flow Matching 学的是**瞬时速度** $v(x_t, t)$，从噪声到数据需要积分。如果能直接学**平均速度**，一步就能到达目的地。

**定义**：

$$u(x_t, r, t) = \frac{1}{t - r} \int_r^t v(x_s, s) \, ds$$

这是从时刻 $r$ 到时刻 $t$ 的平均速度（$x_s$ 沿 FM 的 ODE 轨迹）。

**核心恒等式（Mean Flow Identity）**：

$$u(x_t, r, t) = v(x_t, t) - (t - r) \frac{\partial}{\partial t} u(x_t, r, t)$$

这把平均速度和瞬时速度联系起来，使得训练成为可能（不需要解 ODE）。

**推导**（用莱布尼茨法则）：
$$\frac{\partial}{\partial t} u(x_t, r, t) = \frac{1}{t-r} v(x_t, t) - \frac{1}{(t-r)^2} \int_r^t v(x_s,s) ds = \frac{v(x_t,t) - u(x_t,r,t)}{t-r}$$

代入恒等式即得。

**训练 Loss**：

$$\mathcal{L}_{\text{MF}} = \mathbb{E}_{t,r,x_0,\epsilon} \left[ \| u_\theta(x_t, r, t) - (x_1 - x_0) / 1 \|^2 + \lambda \| \text{JVP term} \|^2 \right]$$

**实现时的关键**：用 `torch.func.jvp` 计算 $\frac{\partial}{\partial t} u_\theta(x_t, r, t)$，不能用标准 autograd（会引入不必要的计算图）。

```python
# Mean Flow 训练的核心代码
import torch
from torch.func import jvp, grad

def mean_flow_loss(model, x0, t, r):
    eps = torch.randn_like(x0)
    x_t = (1 - t[:,None]) * eps + t[:,None] * x0  # x_t on FM path
    
    # 目标：平均速度应该等于 x0 - eps（= x_1 - x_0 归一化后）
    target = x0 - eps
    
    # 计算 u_θ(x_t, r, t)
    u = model(x_t, r, t)  # 模型输出平均速度
    
    # 计算 d/dt u_θ(x_t, r, t)（关键：用 JVP）
    def u_of_t(t_scalar):
        x_t_new = (1 - t_scalar) * eps + t_scalar * x0
        return model(x_t_new, r, t_scalar.expand(len(x0)))
    
    tangent = torch.ones_like(t)
    _, du_dt = jvp(u_of_t, (t,), (tangent,))
    
    # Mean Flow Identity 约束
    identity_residual = u - (model_v(x_t, t) - (t-r)[:,None] * du_dt)
    
    # 主 loss（对齐目标速度）+ identity loss
    loss = F.mse_loss(u, target) + 0.1 * F.mse_loss(identity_residual, torch.zeros_like(identity_residual))
    return loss
```

**1-NFE 采样**：

$$x_0 = x_1 - 1 \cdot u_\theta(x_1, 0, 1)$$

从纯噪声 $x_1 \sim \mathcal{N}(0, I)$ 直接一步到数据 $x_0$。

### 7.2 JVP 数值稳定性问题（重要）

`torch.func.jvp` 在实践中有以下坑：
1. **内存双倍**：JVP 需要同时保存原始计算图和切向量计算图
2. **梯度二阶问题**：如果 $t$ 同时作为 loss 的部分，容易出现梯度 NaN
3. **解决方案**：在 JVP 调用内用 `torch.no_grad()` 隔离不需要梯度的部分；使用 `detach()` 截断不必要的连接

**建议**：先从 `noamelata/MeanFlow` 或 `haidog-yaqub/MeanFlow` 两个 repo port loss 代码，它们已经处理了这些问题。

### 7.3 算力需求 & 难度

| 项目 | 规格 |
|---|---|
| GPU | 单卡 RTX 3090，显存 ~8GB（JVP 需要更多内存）|
| 训练时间 | ~2-3小时（JVP 计算开销比 FM 大 2-3x）|
| 难度 | ★★★★☆（JVP 实现是本项目最难的部分）|

**关键文献**：
- Geng et al. (2025), "Mean Flows for One-step Generative Modeling", NeurIPS 2025 Oral — **课程参考论文**（我们的主要创新参考）

**非金融同类工作**：
- 原论文仅在 ImageNet 256×256 上验证
- 无时序数据上的应用文献（这是我们的创新点）

### 7.4 Fallback 方案

如果 Mean Flow JVP 训练 W2 末仍不收敛：
- **Fallback A**：只用 Consistency Models 做 1-step 方法，创新点改为"CM 在金融路径上的首次应用 + 超参分析"
- **Fallback B**：用 Mean Flow 的近似版本（不用 JVP，用有限差分近似时间导数）

---

## 八、Step 7：评测框架

### 8.1 维度 A：Stylized Facts（生成质量）

金融时间序列的 5 个公认统计性质（Cont 2001, "Empirical properties of asset returns"):

**1. 厚尾（Heavy Tails）**：
$$\text{Kurtosis}(r) = \frac{\mathbb{E}[r^4]}{(\mathbb{E}[r^2])^2} > 3$$
用 Hill estimator 拟合尾指数 $\alpha$（$\alpha < 4$ 意味着四阶矩不存在，即"真厚尾"）

**2. 波动率聚集（Volatility Clustering）**：
$$\text{ACF}_{|r|}(k) = \text{Corr}(|r_t|, |r_{t+k}|) \approx c \cdot k^{-\gamma}$$
幂律衰减，$\gamma \in (0, 1)$

**3. 收益率无序列相关（No Autocorrelation in Returns）**：
$$\text{ACF}_r(k) = \text{Corr}(r_t, r_{t+k}) \approx 0, \quad k > 1$$

**4. 杠杆效应（Leverage Effect）**：
$$\text{Corr}(r_t, |r_{t+k}|^2) < 0 \text{ for } k > 0, \approx 0 \text{ for } k < 0$$

**5. 聚合正态性（Aggregational Gaussianity）**：
$$r_t^{(m)} = \sum_{i=0}^{m-1} r_{t+i}$$
Jarque-Bera 统计量随 $m$ 增大而趋于 0（收益率在较长尺度上更接近正态分布）

**量化指标**：对每个 stylized fact，计算生成分布 vs 真实（Heston）分布的 **Wasserstein-1 距离**。5 个指标归一化后画雷达图。

### 8.2 维度 B：采样速度

```python
# 标准化测试
N_samples = 10000
times = {}
for model_name in ['Score SDE', 'FM', 'Consistency', 'Mean Flow']:
    start = time.time()
    samples = generate(model, N_samples)
    times[model_name] = time.time() - start
```

预期结果（单卡 3090）：
| 模型 | NFE | 预期时间（10k 路径）|
|---|---|---|
| Score SDE | 1000/50 | 60s/5s |
| Flow Matching | 50 | 3s |
| Consistency | 1/2 | 0.5s |
| Mean Flow | 1 | 0.3s |

### 8.3 维度 C：衍生品定价精度（核心创新实验）

**实验设计**：

1. 用每个模型生成 $N_{\text{MC}} = 100,000$ 条路径 $\{S^{(i)}_T\}_{i=1}^{N_{\text{MC}}}$
2. 估计期权价格（Monte Carlo）：
$$\hat{C}_{\text{model}}(K, T) = e^{-rT} \frac{1}{N_{\text{MC}}} \sum_{i=1}^{N_{\text{MC}}} \max(S^{(i)}_T - K, 0)$$
3. 对比 Carr-Madan 精确值 $C^*_{\text{Heston}}(K, T)$：
$$\text{RMSE} = \sqrt{\frac{1}{15} \sum_{(K,T)} \left(\hat{C}_{\text{model}}(K,T) - C^*_{\text{Heston}}(K,T)\right)^2}$$
$$\text{MaxRE} = \max_{(K,T)} \frac{|\hat{C}_{\text{model}}(K,T) - C^*_{\text{Heston}}(K,T)|}{C^*_{\text{Heston}}(K,T)}$$

4. **隐含波动率微笑重建**（可视化）：把估计的期权价格反推 BS 隐含波动率，与 Heston 真实 smile 对比

**Mean Flow 的亮点**：单步采样 → 生成 100k 路径只需 ~3秒 → 期权定价延迟从 Heston QE MC 的分钟级降到**毫秒级**

### 8.4 消融实验

```
Ablation A：NFE 对质量的影响
  Score SDE：NFE ∈ {1, 5, 10, 50, 1000}
  FM：NFE ∈ {1, 5, 10, 50}
  Consistency：NFE ∈ {1, 2}
  Mean Flow：NFE ∈ {1} (by design)

Ablation B：数据量的影响
  N_train ∈ {5k, 10k, 25k, 50k}

Ablation C：U-Net 深度的影响
  Layers ∈ {2, 4} down-blocks
```

### 8.5 算力需求 & 难度

| 子任务 | 算力需求 | 难度 |
|---|---|---|
| Stylized facts 评测代码 | CPU | ★★☆☆☆ |
| MC 期权定价（100k 路径）| GPU，<5min | ★★★☆☆ |
| Carr-Madan 对比 | CPU | ★★☆☆☆ |
| IV smile 重建（BS 反推）| CPU | ★★★☆☆（需要数值求解 BS 公式）|

---

## 九、Step 8：可视化（5 个核心 piece）

### 9.1 采样过程动画（开场）

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
titles = ['Score SDE', 'Flow Matching', 'Consistency', 'Mean Flow']
nfe_list = [50, 50, 2, 1]

def animate(frame):
    for i, (ax, title, nfe) in enumerate(zip(axes, titles, nfe_list)):
        t = 1 - frame / max_frames  # 从噪声到数据
        x = generate_at_t(model_list[i], z_fixed, t)
        ax.clear()
        ax.plot(x[0].cpu(), alpha=0.8)
        ax.set_title(f'{title}\nNFE={nfe}')
        ax.set_ylim(-0.05, 0.05)

ani = FuncAnimation(fig, animate, frames=100, interval=50)
ani.save('denoising.gif', writer='pillow')
```

### 9.2 Stylized Facts 雷达图

5 轴雷达图，每轴 = 某个 stylized fact 的还原程度（越外圈越好，最外圈=Heston ground truth）。

### 9.3 期权定价误差 Heatmap

```python
# 5 strikes × 3 maturities 的相对误差 heatmap
import seaborn as sns
errors = (C_model - C_gt) / C_gt  # 相对误差 [5, 3]
sns.heatmap(errors, annot=True, fmt='.2%', cmap='RdYlGn_r',
            xticklabels=['T=0.25', 'T=0.5', 'T=1.0'],
            yticklabels=['K=80','K=90','K=100','K=110','K=120'])
```

### 9.4 隐含波动率微笑重建

```python
# 用生成路径反推 BS 隐含波动率
from scipy.optimize import brentq
from scipy.stats import norm

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol(C_market, S, K, T, r):
    return brentq(lambda s: bs_call(S, K, T, r, s) - C_market, 1e-6, 10)
```

### 9.5 交互式 Demo（可选，Gradio）

```python
import gradio as gr

def generate_and_price(kappa, theta, xi, rho, model_name):
    paths = generate_heston_paths(kappa, theta, xi, rho)
    prices = mc_option_price(paths, K_grid, T_grid)
    return plot_paths(paths), plot_smile(prices)

demo = gr.Interface(
    fn=generate_and_price,
    inputs=[
        gr.Slider(0.5, 5, value=2, label="κ"),
        gr.Slider(0.01, 0.1, value=0.04, label="θ"),
        gr.Slider(0.1, 1, value=0.3, label="ξ"),
        gr.Slider(-0.9, -0.1, value=-0.7, label="ρ"),
        gr.Radio(['Score SDE', 'FM', 'Consistency', 'Mean Flow'])
    ],
    outputs=[gr.Plot(), gr.Plot()]
)
```

---

## 十、完整 7 周排期（修订版）

起点：2026-04-21，展示：2026-06-07

| 周 | 日期 | Person A（数据/评测）| Person B（模型/训练）|
|---|---|---|---|
| W1 | 4/21–4/27 | Heston QE 实现 + 数据生成 + Carr-Madan FFT | 1D U-Net backbone + 单元测试 + Lightning 框架 |
| W2 | 4/28–5/04 | Stylized facts 评测代码（5 个指标）| Score SDE 训练 + DPM-Solver++ 采样 |
| W3 | 5/05–5/11 | 期权 MC 定价 pipeline + 与 Carr-Madan 对比 | Flow Matching 训练 + 收敛对比 |
| W4 | 5/12–5/18 | Proposal 1页（5/14 截止）+ 评测自动化脚本 | Consistency Training（EMA tuning）|
| W5 | 5/19–5/25 | 完整 stylized facts 表格（Score SDE vs FM vs CT）| Mean Flow JVP 实现 + 调试 |
| W6 | 5/26–6/01 | 期权定价完整实验（4 模型 × 15点）+ Vol smile | Mean Flow 训练 + 消融实验 |
| W7 | 6/02–6/07 | 写 report §4 实验 + 5 个可视化 + Gradio demo | 写 report §3 方法 + PPT + README |

### Go/No-Go 检查点

| 时间 | 检查点 | 通过条件 | 失败应对 |
|---|---|---|---|
| W2 末 | Score SDE 生成质量 | Kurtosis W1 < 0.01 | 检查数据 normalization，调整 β schedule |
| W3 末 | FM vs Score SDE | FM ≥ Score SDE on 3/5 stylized facts | 检查 CFM loss 实现，σ_min 设置 |
| W5 末 | Mean Flow 收敛 | JVP loss 下降，生成路径合理 | Fallback to CT only，CM 作为 1-step 方法 |
| W6 末 | 期权定价误差 | Mean Flow RMSE < Heston MC 误差的 2x | 增加 MC 路径数到 200k |

---

## 十一、完整算力汇总

| 实验 | GPU | 显存 | 时间 | 次数估计 | 总 GPU-小时 |
|---|---|---|---|---|---|
| 数据生成（55k Heston）| CPU | — | 20分钟 | 1次 | — |
| Score SDE 训练 | RTX 3090 | 4GB | 1小时 | 3次（调参）| 3h |
| Flow Matching 训练 | RTX 3090 | 3GB | 45分钟 | 3次 | 2.25h |
| Consistency Training | RTX 3090 | 5GB | 1.5小时 | 4次 | 6h |
| Mean Flow 训练 | RTX 3090 | 8GB | 2.5小时 | 5次（JVP 调试）| 12.5h |
| 采样（10k 路径 × 4模型）| RTX 3090 | 3GB | 10分钟 | 5次 | 0.8h |
| MC 期权定价（100k 路径）| RTX 3090 | 4GB | 5分钟 | 5次 | 0.4h |
| Ablation 实验 | RTX 3090 | 5GB | 1小时/组 | 10组 | 10h |
| **总计** | | | | | **~35 GPU-小时** |

**结论**：单卡 RTX 3090，7 周内完全够用。即使用 Colab Pro（A100 约 2-3x 快）也可以。

---

## 十二、完整文献汇总（按主题）

### A. 核心方法（必须引用）

| 论文 | 作者 | 年份 | arXiv | 对应模型 |
|---|---|---|---|---|
| Score-Based Generative Modeling through SDEs | Song et al. | ICLR 2021 | 2011.13456 | Score SDE |
| Flow Matching for Generative Modeling | Lipman et al. | ICLR 2023 | 2210.02747 | FM |
| Consistency Models | Song et al. | ICML 2023 | 2303.01469 | CM |
| Improved Consistency Training | Song et al. | NeurIPS 2024 | 2310.14189 | CM（超参改进）|
| Mean Flows for One-step Generative Modeling | Geng et al. | NeurIPS 2025 | — | Mean Flow（**课程参考**）|
| DDPM | Ho, Jain, Abbeel | NeurIPS 2020 | 2006.11239 | Score SDE baseline |
| DPM-Solver++ | Lu et al. | NeurIPS 2022 | 2211.01095 | 采样加速 |
| Neural ODE | Chen et al. | NeurIPS 2018 | 1806.07366 | FM 前身 |

### B. 时间序列生成（非金融，方法论 reference）

| 论文 | 作者 | 年份 | arXiv | 说明 |
|---|---|---|---|---|
| TimeGrad | Rasul et al. | ICML 2021 | 2101.12072 | Score SDE 做时序预测，首篇 |
| CSDI | Tashiro et al. | NeurIPS 2021 | 2107.03502 | 条件 score-based 时序生成 |
| DiffWave | Kong et al. | ICLR 2021 | 2009.09761 | 1D U-Net + 扩散做音频 |
| TimeFlow | Cheng et al. | ICLR 2026 sub. | 2511.07968 | FM 做通用时序（直接竞品/参考）|

### C. 金融时间序列生成（直接 related work）

| 论文 | 作者 | 年份 | 说明 |
|---|---|---|---|
| Quant GANs | Wiese et al. | QF 2020 | 最早金融时序 GAN，重要 baseline |
| Sig-Wasserstein GAN | Ni, Szpruch et al. | 2021 | Signature + SDE 生成路径 |
| Neural SDEs as Infinite-Dim GANs | Kidger et al. | ICML 2021 | 金融 Neural SDE baseline |
| Latent SDE | Li, Chen et al. | AISTATS 2020 | Latent SDE，另一类 baseline |
| FinDiff | Sensoy et al. | 2023 | DDPM 做金融表格数据 |
| Kim et al. | Kim et al. | 2025 | Diffusion 做金融时序，最近相关工作 |

### D. Heston 模型 & 数值方法

| 论文 | 作者 | 年份 | 说明 |
|---|---|---|---|
| A Closed-form Solution for Options with Stochastic Volatility | Heston | RFS 1993 | Heston 模型原论文 |
| Efficient Simulation of the Heston Stochastic Volatility Model | Andersen | JCF 2007 | QE 格式，必读 |
| Option Valuation Using the Fast Fourier Transform | Carr & Madan | JCF 1999 | FFT 期权定价 |

### E. 金融时序统计性质（评测依据）

| 论文 | 作者 | 年份 | 说明 |
|---|---|---|---|
| Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues | Cont | QF 2001 | 5 个 stylized facts 的权威综述 |

---

## 十三、人员分工建议

| 模块 | Person A | Person B |
|---|---|---|
| 数据生成 | ✅ 主责 | 辅助 |
| U-Net backbone | 辅助 | ✅ 主责 |
| Score SDE | 辅助 | ✅ 主责 |
| Flow Matching | 辅助 | ✅ 主责 |
| Consistency | 辅助 | ✅ 主责 |
| Mean Flow | ✅ JVP 数学推导 | ✅ 代码实现 |
| Stylized facts 评测 | ✅ 主责 | 辅助 |
| 期权 MC 定价 | ✅ 主责 | 辅助 |
| Carr-Madan | ✅ 主责 | 辅助 |
| Vol smile 重建 | ✅ 主责 | 辅助 |
| 可视化（5 piece）| ✅ 主责 | 辅助 |
| Gradio demo | 辅助 | ✅ 主责 |
| Report §3 方法 | 辅助 | ✅ 主责 |
| Report §4 实验 | ✅ 主责 | 辅助 |
| PPT | ✅ 结构 | ✅ 视觉 |

---

## 十四、最终可行性评估

### 总体评分

| 维度 | 评分 | 说明 |
|---|---|---|
| 课程合规性 | 85/100 | 用方法学 framing 后，主要风险是 TA 对"金融数据"的态度，建议先确认 |
| 技术可行性 | 80/100 | Score SDE / FM / CT 无压力；Mean Flow JVP 是卡点，有 fallback |
| 创新性 | 78/100 | Mean Flow 应用到 1D 时序是真空地带；精确 ground truth 评测是方法论贡献 |
| 文献支撑 | 75/100 | 方法论文献充足；直接对标 reference 偏少，related work 需要用"借鉴式"写法 |
| 可视化潜力 | 88/100 | 5个 piece 设计得当，差异化明显 |
| 3.5 周可达性 | 75/100 | MVP（Score SDE + FM + CT + 评测）可达；Mean Flow 需要 fallback 预案 |

### 最保守的可交付成果（MVP，必然可完成）

1. Score SDE + Flow Matching 在 Heston 路径上的系统对比
2. 5 个 stylized facts 雷达图
3. 期权定价误差 heatmap
4. Report 4页 + PPT 20min

### 完整版（理想情况，有 Mean Flow）

1. + Consistency Models（1-step 方法）
2. + Mean Flow（1-step，毫秒级定价引擎）
3. + Vol smile 重建
4. + Gradio demo

---

## 十五、最紧急事项（5/13 当天）

**明天（5/14）24:00 是 Proposal 截止**。现在最重要的是：

1. **今天发给 TA 确认**（用这个话术）：
   > "老师好，我们想做'现代生成模型（Score SDE / Flow Matching / Consistency Models / Mean Flow）在 1D 随机时间序列上的系统对比研究'，使用 Heston SDE 合成数据作为可控测试床，评估生成质量、采样效率和下游任务精度（期权定价）。核心方法均来自课程参考论文（Song 2021/2023, Lipman 2023, Geng 2025）。请问选题是否合适？"

2. **今晚写 Proposal（1页）**，包含：
   - 题目 + 一句话定义
   - 为什么用 Heston（closed-form ground truth = 精确评测）
   - 4 个方法各自对应课程哪篇论文
   - 评测指标（stylized facts + 期权定价误差）
   - 分工

3. **建 GitHub repo 骨架**：
   ```
   finflow-bench/
   ├── data/          # Heston 路径生成
   ├── models/        # 4 个生成模型
   ├── eval/          # Stylized facts + 期权定价
   ├── viz/           # 可视化代码
   └── experiments/   # 训练配置（hydra）
   ```
