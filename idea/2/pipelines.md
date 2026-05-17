# 方案存档：三版 Pipeline 技术路径

> 记录时间：2026-05-13。全部基于对话中的讨论，按版本顺序存档。
> 数据基础（三版共用）：Heston QE 模拟路径，Carr-Madan FFT 作期权定价 ground truth。

---

## 共用基础：数据模块

### Heston 随机波动率模型

$$dS_t = \mu S_t \, dt + \sqrt{v_t} \, S_t \, dW^S_t$$
$$dv_t = \kappa(\theta - v_t) \, dt + \xi \sqrt{v_t} \, dW^v_t, \quad d\langle W^S, W^v \rangle_t = \rho \, dt$$

**训练参数**：κ=2.0, θ=0.04, ξ=0.3, ρ=-0.7, v₀=0.04, S₀=100, μ=0.05, T=1年, Δt=1/252

**数值格式**：Andersen QE scheme（工业标准，保证 v_t ≥ 0 精确成立，避免 Euler-Maruyama 负方差问题）

**模型输入**：日对数收益率序列 r_t = log(S_{t+1}/S_t)，shape [B, 252]

**期权定价 Ground Truth**：Carr-Madan FFT 利用 Heston 特征函数半解析性，在 15 个 (K,T) 网格点上给出精确期权价格

**数据规模**：训练 50k 条路径，验证 5k，测试 10k，每条 252 步

---

## V1：四步递进蒸馏 Pipeline

**研究问题**：对于金融路径生成和期权定价，FM 的 1 步蒸馏（CD vs Mean Flow）哪条路更好？能否把两条路合并成一个模型？

**架构核心**：同一个 FM 教师模型，分两条路蒸馏，最后尝试合并

### Step 1：训练 Flow Matching 基础模型（可复现基础）

**目标**：训练 50 步多步生成器 v_θ，作为蒸馏的教师模型

**方法**：Conditional Flow Matching（Lipman 2023），直线概率路径

**CFM Loss**：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\, x_1,\, \epsilon} \left[ \| v_\theta(x_t, t) - (x_1 - \epsilon) \|^2 \right], \quad x_t = (1-t)\epsilon + t x_1$$

**网络**：1D U-Net，输入 [B, 1, 252]，~8M 参数，时间嵌入 sinusoidal→MLP

**推理**：50 步 Euler ODE，NFE=50

**复现来源**：TimeFlow（arXiv:2511.07968），将其单阶段 FM 应用于 Heston 路径

---

### Step 2：CD 蒸馏（已有方法，应用到金融是新的）

**目标**：将 Step 1 的 v_θ（50 步）蒸馏为 f_θ（1 步）

**方法**：Consistency Distillation（Song 2023，课程参考论文）——利用教师 ODE 轨迹构造相邻点对

**CD Loss**：

$$\mathcal{L}_{\text{CD}} = \mathbb{E}_{n,\, x,\, z} \left[ d\!\left( f_\theta(x + \sigma_{n+1} z,\; \sigma_{n+1}),\; f_{\theta^-}(\hat{x}^{\psi}_{t_n},\; \sigma_n) \right) \right]$$

其中 $\hat{x}^{\psi}_{t_n}$ 是教师 v_θ 从 $(x+\sigma_{n+1}z, \sigma_{n+1})$ 走一步 ODE 的估计，$\theta^-$ 是 EMA 参数（停止梯度），$d$ 取 MSE（1D 时序）

**边界条件**：$f_\theta(x, \epsilon) = x$，通过参数化 $f_\theta = c_{\text{skip}}(t)x + c_{\text{out}}(t)F_\theta(x,t)$ 自动满足

**推理**：NFE=1，直接 $f_\theta(x_T, T) \to x_0$；NFE=2 可加一步精修

**关键超参风险**：EMA decay schedule 必须按 Song 2023 Appendix 精确实现，否则训练不稳定

---

### Step 3：Mean Flow 蒸馏（应用到金融是新的）

**目标**：将同一个 v_θ 蒸馏为 u_θ（1 步），走 Mean Flow 路线

**方法**：Mean Flow（Geng 2025，NeurIPS Oral，课程参考论文）

**核心定义**：平均速度

$$u(x_t, r, t) = \frac{1}{t - r} \int_r^t v(x_s, s) \, ds$$

**Mean Flow Identity**（核心恒等式，由莱布尼茨法则导出）：

$$u(x_t, r, t) = v(x_t, t) - (t - r) \frac{\partial}{\partial t} u(x_t, r, t)$$

**Training Loss**：

$$\mathcal{L}_{\text{MF}} = \mathbb{E}_{t,\, r,\, x_0,\, \epsilon} \left[ \| u_\theta(x_t, r, t) - (x_0 - \epsilon) \|^2 \right] + \lambda \cdot \text{Identity Residual}$$

**JVP 实现**（关键，不能用标准 autograd）：

```python
from torch.func import jvp

def u_of_t(t_scalar):
    x_t = (1 - t_scalar) * eps + t_scalar * x0
    return model(x_t, r, t_scalar.expand(B))

_, du_dt = jvp(u_of_t, (t,), (torch.ones_like(t),))
```

**1-NFE 采样**：

$$x_0 = x_1 - 1 \cdot u_\theta(x_1, 0, 1), \quad x_1 \sim \mathcal{N}(0, I)$$

**主要技术风险**：JVP 内存双倍；梯度图泄漏导致 NaN；需从现有开源实现 port（noamelata/MeanFlow 或 haidog-yaqub/MeanFlow）

---

### Step 4（核心创新）：合并 CD + Mean Flow，提出 Combined Loss

**研究假设**：CD 的自洽性约束和 Mean Flow 的平均速度约束，能否同时施加在一个模型上？

**提出的 Combined Model**：$h_\theta(x_t, r, t)$ 同时满足：
1. 自洽性（CD 约束）：$h_\theta(x_t, t) = h_\theta(x_{t'}, t')$ 对同一轨迹上的 $(t, t')$
2. Mean Flow Identity（MF 约束）：$h = v - (t-r)\partial_t h$

**Combined Loss**：

$$\mathcal{L}_{\text{combined}} = \mathcal{L}_{\text{FM}} + \alpha \cdot \mathcal{L}_{\text{CD}} + \beta \cdot \mathcal{L}_{\text{MF}}$$

**核心问题**（尚未有文献回答）：
- CD 和 MF 的约束是否相容？（理论上需要证明）
- 联合训练是否比单独蒸馏更稳定？（实验验证）
- 最优 $\alpha, \beta$ 是多少？

**评测**：
- 统计评测：5 个 stylized facts（Wasserstein-1）——厚尾、波动率聚集、杠杆效应、聚合高斯性、无自相关
- 金融评测：15 个 (K,T) 网格点的期权定价 RMSE vs Carr-Madan ground truth

**局限性说明**：这个 pipeline 是纯方法学对比，没有条件生成、没有世界模型结构。生成的是"从噪声到完整路径"，不是"从当前状态到未来状态"。

---

## V2：两阶段分层生成 Pipeline

**核心改动**：基于 Heston 模型的隐变量结构，把单阶段生成改为两阶段——先学隐变量（波动率路径），再学观测变量（收益率路径）条件于隐变量

**架构核心**：
```
噪声 z_v → FM_vol → v̂_{1:252}（波动率路径）
噪声 z_r + 条件 v̂  → FM_ret → r̂_{1:252}（收益率路径）

两个 FM 分别做 Mean Flow 蒸馏：
  FM_vol → MF_vol（1步生成 v̂）
  FM_ret → MF_ret（1步生成 r̂，条件于 v̂）
```

### Stage 1：FM 学波动率路径 v_t

**为什么先学 v_t**：Heston 中 v_t 是 CIR 过程（均值回归、始终为正、平滑），比收益率序列更"规则"，FM 在其上收敛快且质量高

**训练目标**（标准 CFM）：

$$\mathcal{L}_{\text{vol}} = \mathbb{E}_{t,\, v_0,\, \epsilon} \left[ \| v_\theta(x_t, t) - (v_0 - \epsilon) \|^2 \right], \quad x_t = (1-t)\epsilon + t \cdot v_0$$

**网络**：1D U-Net，输入 [B, 1, 252]，独立参数集

---

### Stage 2：条件 FM 学收益率路径 r_t | v_t

**为什么 r_t | v_t 容易学**：给定 v_t，Heston 的收益率几乎是高斯的

$$r_t \approx (\mu - v_t/2)\Delta t + \sqrt{v_t \Delta t} \cdot (\rho Z_t^v + \sqrt{1-\rho^2} Z_t^\perp)$$

条件依赖结构显式捕捉了杠杆效应（leverage effect，ρ < 0 时，高波动对应负收益）

**条件注入**：Channel concat（推荐，实现最简单）——把 v_t 路径直接拼在输入通道上，网络输入变为 [B, 2, 252]

**条件 CFM Loss**：

$$\mathcal{L}_{\text{ret}} = \mathbb{E}_{t,\, r_0,\, \epsilon} \left[ \| v_\theta(x_t, t, \mathbf{c}_v) - (r_0 - \epsilon) \|^2 \right]$$

其中 $\mathbf{c}_v$ 是条件 v_t 序列，通过 channel concat 注入

**消融对比**：

| 条件注入方式 | 实现 | 难度 |
|---|---|---|
| Channel concat | [B, 2, 252] 输入 | ★ |
| FiLM | v_t 统计量调制每层特征 | ★★ |
| Cross-attention | r_t query，v_t key/value | ★★★ |

---

### Stage 3：Mean Flow 蒸馏两个 FM 模型

对 FM_vol 和 FM_ret 分别做 Mean Flow 蒸馏（方法细节同 V1 Step 3）

**蒸馏后的完整推理**（总共 2 次前向传播）：

```python
z_v = torch.randn(N, 252)
v_hat = MF_vol(z_v, r=0, t=1)          # 1步，生成波动率路径

z_r = torch.randn(N, 252)
r_hat = MF_ret(z_r, v_hat, r=0, t=1)  # 1步，条件于 v_hat
```

---

### 理论亮点（报告中的 intellectual contribution，不需要完整实验）

**Girsanov 视角**：

Heston 模型在风险中性测度 Q（定价测度）和真实测度 P（历史测度）之间通过 Girsanov 定理转换。

在 P 测度下训练的 FM 模型学到 score function $\nabla \log p^P(x_t)$；
在 Q 测度下则学 $\nabla \log p^Q(x_t)$。

两者的差：

$$\nabla \log p^Q(x_t) - \nabla \log p^P(x_t) = \nabla \log \frac{dQ}{dP}\bigg|_{x_t}$$

这个差正是 Girsanov 核（market price of risk，$\lambda_t = (\mu - r)/\sqrt{v_t}$）。

**结论**：在 P 测度数据上训练 FM，然后用 Girsanov 调整 score，等价于直接在 Q 测度上训练。这给出了一个在真实历史数据（P 测度）上训练但用于定价（Q 测度）的理论框架——连接生成建模和金融数学的桥梁。

---

### V2 vs V1 的差异

| 维度 | V1 | V2 |
|---|---|---|
| 生成结构 | 单阶段，学联合分布 (v_t, r_t) | 两阶段，先 v_t 再 r_t\|v_t |
| 杠杆效应建模 | 隐式（模型自己学相关性） | 显式（通过条件依赖结构）|
| 研究问题 | CD vs MF 蒸馏哪个好？能否合并？ | 分层结构是否比单阶段更好？|
| 世界模型元素 | 无 | 无（是分层生成模型，不是世界模型）|
| 类比课程论文 | Consistency Models + Mean Flow | DIAMOND 两阶段结构（形式类比，非功能等价）|
| 创新点清晰度 | 高（CD+MF 合并是明确问题） | 高（两阶段结构是明确贡献）|

**注意**：V2 中引用 DIAMOND 类比是形式上的类比（都有两阶段），不是功能等价——DIAMOND 是世界模型（状态+动作→下一状态），V2 是分层生成（噪声→完整路径）。

---

## V3：自回归世界模型 Pipeline

**核心改动**：从"噪声→路径"转变为"当前状态→下一状态分布"，引入真正的世界模型结构

**定义**：
- **状态**：$s_t = (v_t, r_t) \in \mathbb{R}^2$（当前波动率和收益率）
- **动作（可选）**：$a_t \in \{$正常体制, 高波动, 崩盘$\}$（宏观场景，one-hot 向量）
- **学习目标**：转移核 $p(s_{t+1} | s_t, a_t)$

**架构**：

```
输入：当前状态 (v_t, r_t) + 动作 a_t + 噪声 ε
         ↓
    FM 学转移核 p(s_{t+1} | s_t, a_t)
         ↓
    自回归滚动：t=0 → t=1 → ... → t=T
```

对比 DIAMOND（直接的功能等价）：

| | DIAMOND | V3 |
|---|---|---|
| 状态 | 游戏画面帧 $o_t$ | 市场状态 $(v_t, r_t)$ |
| 动作 | 手柄操作 $a_t$ | 宏观场景 $a_t$ |
| 转移 | $p(o_{t+1}\|o_t, a_t)$ | $p(s_{t+1}\|s_t, a_t)$ |
| 生成方式 | 扩散模型 | Flow Matching |

### Stage 1：FM 学转移核（两阶段，显式分离 v 和 r）

**Step 1a**：条件 FM 学波动率转移

$$p(v_{t+1} | v_t, a_t) \quad \text{（CIR 转移，条件于当前波动率和场景）}$$

$$\mathcal{L}_{\text{vol-trans}} = \mathbb{E} \left[ \| v_\theta(x_\tau, \tau, v_t, a_t) - (v_{t+1} - \epsilon) \|^2 \right]$$

其中 $\tau \in [0,1]$ 是 FM 内部时间，$(v_t, a_t)$ 通过 channel concat 或 cross-attention 注入

**Step 1b**：条件 FM 学收益率转移（条件于 v_{t+1} 和 v_t）

$$p(r_{t+1} | v_{t+1}, v_t, r_t, a_t)$$

$$\mathcal{L}_{\text{ret-trans}} = \mathbb{E} \left[ \| v_\theta(x_\tau, \tau, v_{t+1}, v_t, r_t, a_t) - (r_{t+1} - \epsilon) \|^2 \right]$$

**训练数据构造**：从 Heston QE 路径抽取相邻步对 $(s_t, s_{t+1}, a_t)$，共 50k × 252 ≈ 12.6M 个转移对（相比 V1/V2 数据量大幅增加）

---

### Stage 2：Mean Flow 蒸馏转移核

对 FM_vol_trans 和 FM_ret_trans 分别做 Mean Flow 蒸馏（方法同 V1 Step 3），得到 1-NFE 的单步转移

**蒸馏后的自回归滚动**：

```python
s0 = (v0, r0)  # 初始市场状态
trajectory = [s0]

for t in range(T):
    v_t, r_t = trajectory[-1]
    
    # 1步生成下一状态（各 1 次前向传播，共 2 次）
    z_v = torch.randn(N, 1)
    v_next = MF_vol_trans(z_v, v_t, a_t, r=0, t=1)
    
    z_r = torch.randn(N, 1)
    r_next = MF_ret_trans(z_r, v_next, v_t, r_t, a_t, r=0, t=1)
    
    trajectory.append((v_next, r_next))
```

---

### V3 特有的评测维度

**路径条件生成（世界模型特有）**：
- 给定 $s_0 = (v_0, r_0)$，生成多条条件轨迹，验证分布收敛到 Heston 条件分布
- 测试不同动作 $a_t$ 对路径分布的影响（高波动体制下尾部变重）

**自回归稳定性**：
- 252 步滚动后是否出现发散（误差累积问题，V3 的主要技术风险）
- 与真实 Heston 条件分布的 Wasserstein 距离 vs 滚动步数

---

### V3 的技术挑战和风险

| 挑战 | 具体问题 | 难度 |
|---|---|---|
| 训练数据量 | 转移对数量是路径数 × T，显存/磁盘压力大 | ★★★ |
| 误差累积 | 252 步自回归，单步误差会指数放大 | ★★★★ |
| 动作定义 | "宏观场景"在 Heston 里怎么映射到参数变化？需要设计 | ★★★ |
| 评测复杂度 | 条件分布评测比无条件评测复杂，需要新评测框架 | ★★★ |
| 时间成本 | 在 V2 基础上额外增加约 1 周工程量 | — |

---

### V3 vs V2 的差异

| 维度 | V2（分层生成）| V3（世界模型）|
|---|---|---|
| 本质 | 噪声 → 完整路径 | 当前状态 → 下一状态分布 |
| 是否是世界模型 | **否** | **是** |
| 类比 DIAMOND | 形式类比（两阶段结构）| 功能等价（状态+动作→转移）|
| 生成粒度 | 整条 252 步路径 | 逐步单步转移 |
| 工程复杂度 | ★★★★ | ★★★★★ |
| 误差累积风险 | 无（一次性生成）| 有（252 步滚动）|
| 截止 5/14 前可描述？ | 是 | 是 |
| 3 周内可完成？ | 是（紧张但可行）| 否（需要 4-5 周）|

---

## 三个版本的对比总结

| 维度 | V1（4步蒸馏对比）| V2（两阶段分层）| V3（自回归世界模型）|
|---|---|---|---|
| 研究问题 | CD vs MF，能否合并？ | 分层结构 vs 单阶段 | 条件路径生成 + 自回归滚动 |
| 是否有世界模型 | 否 | 否 | **是** |
| 课程参考论文覆盖 | CM + MF + FM | DIAMOND（形式）+ MF + FM | DIAMOND（功能）+ MF + FM |
| 实现难度 | ★★★★ | ★★★★ | ★★★★★ |
| 3周可完成 | 是 | 是 | **否** |
| 实验密度 | 高（4组模型对比）| 中（2组对比 + 消融）| 低（时间不够）|
| 创新点清晰度 | 高（合并 Loss 是新的）| 高（两阶段结构是新的）| 中（思路有先例）|
| 学术诚实度 | 高（明确说不是WM）| 高（明确说不是WM）| 高（真正是WM）|

---

## 当前推荐方案

基于讨论，推荐 **V1 + V2 结合**：

```
基准线（复现）：单阶段 FM 在 Heston 路径上（TimeFlow 思路）
                          ↓
创新 1（V2 思路）：两阶段分层 FM（v_t 先学，r_t|v_t 再学）
                          ↓
创新 2（V1 思路）：对两阶段模型做 Mean Flow 蒸馏（→1步）
                          ↓
消融（V1 思路）：  单阶段 FM 做 CD 蒸馏 vs Mean Flow 蒸馏对比
```

这样既有 V2 的结构创新（两阶段分层），又有 V1 的蒸馏比较（实验密度），同时诚实承认这是分层生成模型而不是世界模型，引用 DIAMOND 只作结构灵感来源。

V3（世界模型）保留为未来方向，在报告的 Future Work 节中提及即可。
