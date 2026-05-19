# V3 实施流程文档：基于 Flow Matching 的自回归金融世界模型

> 生成日期：2026-05-17
> 范围：依据 [02_pipelines.md](02_pipelines.md) 中的 V3 方案与 [03_V3_References.md](03_V3_References.md) 中的文献清单，给出从数据 → 训练 → 评测的端到端实施流程，并对照当前 `finflow/` 代码库标记完成度。
> 一句话定义：用 Flow Matching 学习 Heston 随机过程的一步转移核 $p(s_{t+1}\mid s_t, a_t)$，用 Mean Flow 蒸馏为 1-NFE 单步生成器，自回归滚动 252 步后用 Heston 闭式解评测分布与定价误差。
> 项目根：[../../](../../)；现有源码：[finflow/](../../finflow/)、[scripts/](../../scripts/)、[tests/](../../tests/)。

---

## 0. 项目的"有据可依"总览

| 维度 | 我们的做法 | 直接对应的文献（03_V3_References.md 段号）|
|---|---|---|
| 合成数据 SDE | Heston 随机波动率模型 | Heston 1993, RFS（§4.1） |
| 数值格式 | Andersen QE 方差更新 + QE-M 收益率 | Andersen 2007, JCF（§4.1） |
| 定价 ground truth | Carr-Madan FFT + Heston 特征函数 | Carr & Madan 1999, JCF（§4.1） |
| 训练数据组织 | 路径 → 相邻 `(s_t,s_{t+1},a_t)` 转移对 | TimeGrad（Rasul 2021）、DIAMOND（Alonso 2024）（§2.1、§3.1）|
| 主训练方法 | Conditional Flow Matching | Lipman 2023, ICLR（§1.1）|
| 分层结构（v 先，r\|v 后） | 隐变量 SDE 分解 | Latent SDE（Li 2020）、Heston 1993 的条件高斯结构（§4.3、§4.1）|
| 网络骨干 | 1D U-Net / FiLM-MLP + sinusoidal time embedding | DDPM（Ho 2020）、DiffWave（Kong 2021）、FiLM（Perez 2018）（§六、§八）|
| 蒸馏方法（主） | Mean Flow + JVP | Geng 2025 NeurIPS Oral、`noamelata/MeanFlow` port（§1.2、§九）|
| 蒸馏方法（对比） | Consistency Distillation | Song 2023, ICML、iCT（Song 2024）（§1.3）|
| 条件注入（动作 $a_t$）| Channel concat / FiLM；可选 CFG | FiLM（§六）、CFG（Ho & Salimans 2022）、Guided Flows（Zheng 2023）|
| 世界模型框架 | $s_t = (v_t, r_t)$、$a_t$ 体制 one-hot，学习转移核 | DIAMOND、GameNGen、Genie、DreamerV3（§2.1–2.2）|
| 自回归滚动 | 252 步、误差累积分析 | TimeGrad、DIAMOND；理论侧 Scheduled Sampling（Bengio 2015）、Professor Forcing（Lamb 2016）（§五）|
| 统计评测（5 stylized facts） | 厚尾、波动率聚集、杠杆效应、聚合高斯性、无自相关 | Cont 2001, QF（§4.4）|
| 路径距离评测 | Wasserstein-1、Sig-Wasserstein | Sig-Wasserstein GAN（Ni 2021）、Quant GANs（Wiese 2020）（§4.3）|
| 定价评测 | 15 个 (K,T) 网格点的 RMSE vs Carr-Madan | Carr-Madan 1999 + Cont 2001 评测范式（§4.1、§4.4）|
| 理论拼图 | Probability Flow ODE、Girsanov 测度变换 | Song 2021 Score SDE、Karatzas-Shreve、Shreve（§七） |

凡是 V3 用到的"合成数据 / 训练模型 / 评测手段"在 03_V3_References.md 里都有显式对应条目；本文档每一节都会再指回这些文献条目，确保 Report / 答辩可以直接引用。

---

## 1. 数据构建（Stage 0）

### 1.1 Heston 随机波动率模型

P 测度下：

$$dS_t = \mu S_t\, dt + \sqrt{v_t}\, S_t\, dW^S_t$$
$$dv_t = \kappa(\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW^v_t,\quad d\langle W^S, W^v\rangle_t = \rho\, dt$$

训练用参数（Heston 1993 基准 + 课程沿用值）：

| 参数 | 值 | 含义 |
|---|---|---|
| $\kappa$ | 2.0 | 方差均值回归速度 |
| $\theta$ | 0.04 | 长期方差均值 |
| $\xi$ | 0.3 | vol of vol |
| $\rho$ | -0.7 | 杠杆效应（负） |
| $v_0$ | 0.04 | 初始方差 |
| $S_0$ | 100 | 初始价格 |
| $\mu$ | 0.05 | 漂移率 |
| $\Delta t$ | 1/252 | 日频 |
| $T$ | 1 年（252 步） | 路径长度 |

文献来源：Heston (1993)、Andersen (2007) §4.1。

### 1.2 数值格式：Andersen QE + QE-M

- **方差**：Andersen QE 格式。计算条件均值 $m$ 与方差 $s^2$，比较 $\psi=s^2/m^2$ 与 $\psi_c=1.5$：
  - $\psi \le \psi_c$：二次型（$v_{t+1}=a(\sqrt{b^2}+z)^2$）
  - $\psi > \psi_c$：指数型（mass-at-zero + tail）
  保证 $v_{t+1}\ge 0$ 精确成立，避免 Euler-Maruyama 的负方差问题。
- **收益率**：QE-M 形式，用 $(v_t, v_{t+1})$ 共同信息：
  $$r_{t+1} = \mu\Delta t + k_0 + k_1 v_t + k_2 v_{t+1} + \sqrt{k_3 v_t + k_4 v_{t+1}}\, Z$$
  其中 $k_0,\dots,k_4$ 是 Andersen 2007 的 closed-form 系数，$Z\sim\mathcal N(0,1)$ 与方差更新独立。

文献来源：Andersen (2007) §4.1。当前代码：[finflow/data/heston.py](../../finflow/data/heston.py)。

### 1.3 路径 → 转移对（V3 特有的数据组织）

V3 不是"噪声 → 整条路径"，而是学转移核，所以训练样本是一步对：

- 条件：$s_t = (v_t, r_{t-1})$，约定 $r_{-1}=0$ 以支持从 $(v_0, 0)$ 滚动
- 目标：$s_{t+1} = (v_{t+1}, r_t)$
- 总样本数：`n_paths × 252` ≈ 12.6M（训练集 50k 路径时）

为了让 FM 训练稳定，$v_t$ 用 $\log(v_t+\epsilon)$ 重参数化（CIR 始终为正，对数空间近似高斯）。归一化统计量从 train split 计算后写入 `metadata.json`。

文献来源：TimeGrad（Rasul 2021）的相邻步训练样本组织 §3.1；DIAMOND（Alonso 2024）的 $(o_t,o_{t+1},a_t)$ 转移训练 §2.1。

### 1.4 动作（regime）设计

`a_t ∈ {normal, high-vol, crash}` 的 one-hot 向量（默认维度 3）。Heston 参数到 regime 的映射（设计选项，需在实施中固定）：

| Regime | $\kappa$ | $\theta$ | $\xi$ | 触发条件 |
|---|---|---|---|---|
| normal | 2.0 | 0.04 | 0.3 | 基线参数 |
| high-vol | 3.0 | 0.09 | 0.5 | 长期方差翻倍，vol-of-vol 提升 |
| crash | 4.0 | 0.16 | 0.8 | 短暂 5–20 步，回归更快 |

每条训练路径混合切换 regime（按 hazard 概率），逐步采样 $a_t$ 并写入 transition 数据。这部分**目前代码尚未实现**，需要在 [finflow/data/heston.py](../../finflow/data/heston.py) 增加 `simulate_regime_switching_heston()`。

参考：DIAMOND 的 action 注入 §2.1；CFG（Ho & Salimans 2022）作为 a_t 注入的标准做法 §六。

### 1.5 期权定价 ground truth：Carr-Madan FFT

利用 Heston 特征函数 $\phi(u) = \mathbb{E}^{\mathbb Q}[e^{iu\log S_T}]$ 的半解析性，对修正后的看涨期权函数做 FFT。

- 15 个 $(K,T)$ 网格点：moneyness $K/S_0 \in \{0.85, 0.90, 0.95, 1.00, 1.05\}$，maturity $T \in \{0.25, 0.5, 1.0\}$
- 输出：每个网格点的 BS 价格 $C^{\text{Heston}}(K,T)$

文献来源：Carr & Madan (1999) §4.1。**当前代码尚未实现**，建议落到 `finflow/data/option_pricing.py`，单独单元测试与 Heston 文献中给定参数集对照。

### 1.6 数据规模与产物

| split | 路径数 | 转移对数 | 用途 |
|---|---|---|---|
| train | 50,000 | 12.6M | FM 训练 + Mean Flow 蒸馏 |
| val | 5,000 | 1.26M | 早停、超参选择 |
| test | 10,000 | 2.52M | 最终评测、定价对照 |

CLI 入口：[scripts/generate_heston_data.py](../../scripts/generate_heston_data.py)。

---

## 2. 模型架构

### 2.1 总体结构（两阶段转移分解）

```
  s_t = (v_t, r_t)                a_t
       │                           │
       └──────────┬────────────────┘
                  │
        ┌─────────▼─────────┐
        │ FM_vol_trans      │   p(v_{t+1} | v_t, a_t)
        │  (1D MLP/U-Net)   │
        └─────────┬─────────┘
                  │  v_{t+1}
                  ▼
        ┌───────────────────┐
        │ FM_ret_trans      │   p(r_{t+1} | v_{t+1}, v_t, r_t, a_t)
        │  (1D MLP/U-Net)   │
        └─────────┬─────────┘
                  │  r_{t+1}
                  ▼
            s_{t+1} = (v_{t+1}, r_{t+1})
```

两阶段分解的依据：Heston 中 $r_t \mid v_t, v_{t+1}$ 显式为高斯（QE-M 给出闭式系数）；先学结构最强的 $v$，再让 $r$ 条件于 $v$，等价于在网络里 hard-code 了 leverage effect。文献：Latent SDE（Li 2020）§4.3、Cont 2001 stylized facts §4.4 的杠杆效应。

> **当前代码状态**：现有 `TransitionFM`（[finflow/models/transition_fm.py](../../finflow/models/transition_fm.py)）是**单阶段** joint 模型——`state_dim=2, condition_dim=2`，直接学 $p(v_{t+1}, r_{t+1}\mid v_t, r_t)$。下一步要拆为 `FM_vol_trans`（state_dim=1，条件 $(v_t, a_t)$）+ `FM_ret_trans`（state_dim=1，条件 $(v_{t+1}, v_t, r_t, a_t)$）。

### 2.2 骨干网络

- **MVP**：FiLM-MLP（已实现）。`SinusoidalTimeEmbedding → context MLP → FiLM 残差块 ×4 → 输出投影`。对 $s_{t+1}$ 这种 2 维状态足够，参数约 50–100k。
- **加强版**（如果 1D U-Net 更稳）：把状态视为 `[B, C, L]`（$L=1$ 退化为 token），用 DiffWave 风格的 dilated 1D U-Net；当条件信息变长（如把整段 $r_{t-k:t}$ 给进来）时收益更明显。
- **时间嵌入**：sinusoidal，DDPM 标配。

文献：DDPM（Ho 2020）、DiffWave（Kong 2021）、FiLM（Perez 2018）§六、§八。

### 2.3 条件注入

| 条件 | 注入位置 | 方法 | 备注 |
|---|---|---|---|
| 当前状态 $(v_t, r_t)$ 或 $(v_{t+1}, v_t, r_t)$ | context MLP | concat + Linear | FiLM 自动 broadcast |
| 动作 $a_t$（one-hot） | context MLP | concat | 训练时 10% 概率置零 → 支持 CFG |
| FM 内部时间 $\tau\in[0,1]$ | context MLP | sinusoidal + concat | 与上述一起进 FiLM 调制 |

CFG 实现：推理时 $\tilde v = (1+w) v(\cdot, a) - w\, v(\cdot, \varnothing)$，$w$ 可调，用于"压制/放大"动作影响（Guided Flows，Zheng 2023）。

---

## 3. 训练阶段

### 3.1 Stage 1a：FM_vol_trans（波动率转移核）

**损失**（标准 Conditional Flow Matching，直线插值）：

$$x_\tau = (1-\tau)\epsilon + \tau \cdot \log v_{t+1},\quad \epsilon\sim\mathcal N(0,1)$$

$$\mathcal L_{\text{vol}} = \mathbb{E}_{\tau,\, (v_t, v_{t+1}, a_t),\, \epsilon}\Big[\|v_\theta(x_\tau, \tau; v_t, a_t) - (\log v_{t+1} - \epsilon)\|^2\Big]$$

文献：Lipman 2023（§1.1）。

**超参（初始建议）**：

| 项 | 值 |
|---|---|
| 优化器 | AdamW, lr 3e-4, wd 1e-4 |
| batch size | 512 |
| epochs | 20（约 5 万步） |
| time-eps | 1e-4（避开端点奇异） |
| grad clip | 1.0 |
| EMA decay（采样时） | 0.999 |

代码位置：[finflow/training.py](../../finflow/training.py) 已经有完整 trainer，可直接复用；只需把 `state_dim=1, condition_dim=1+|A|`。

### 3.2 Stage 1b：FM_ret_trans（收益率转移核）

**损失**（条件 FM，条件包含已生成的 $v_{t+1}$）：

$$\mathcal L_{\text{ret}} = \mathbb{E}\Big[\|v_\theta(x_\tau, \tau; v_{t+1}, v_t, r_t, a_t) - (r_{t+1} - \epsilon)\|^2\Big]$$

训练时 $v_{t+1}$ 用 ground truth（teacher forcing）；推理时来自 Stage 1a 的采样。

**Exposure bias 缓解**（防止训练-推理分布不匹配）：以 50% 概率把训练时 $v_{t+1}$ 替换为 Stage 1a 的 noisy 估计（scheduled sampling，Bengio 2015 §五）。这一步在 Stage 1 训练稳定后再加，先跑 vanilla 版本。

### 3.3 Stage 2a：Mean Flow 蒸馏

**目标**：把多步 FM（NFE=20–50）压缩为 1-NFE 生成器。

**核心定义**（Geng 2025 §1.2）：

$$u(x_t, r, t) = \frac{1}{t-r}\int_r^t v(x_s, s)\, ds$$

**Mean Flow Identity**（莱布尼茨法则导出）：

$$u(x_t, r, t) = v(x_t, t) - (t-r)\frac{\partial u}{\partial t}(x_t, r, t)$$

**训练损失**：

$$\mathcal L_{\text{MF}} = \mathbb E_{t, r, x_0, \epsilon}\Big[\|u_\theta(x_t, r, t) - (x_0 - \epsilon)\|^2\Big] + \lambda\cdot \text{IdentityResidual}$$

**JVP 实现**（关键技术难点，Geng 2025 + `noamelata/MeanFlow` port §九）：

```python
from torch.func import jvp

def u_of_t(t_scalar):
    x_t = (1 - t_scalar) * x0 + t_scalar * eps
    return model(x_t, r, t_scalar.expand(B), cond)

_, du_dt = jvp(u_of_t, (t,), (torch.ones_like(t),))
target = (eps - x0) - (t - r) * du_dt.detach()  # stop-grad to break graph
loss = ((model(x_t, r, t, cond) - target)**2).mean()
```

**采样**（1-NFE）：

$$x_0 = x_1 - 1\cdot u_\theta(x_1, 0, 1),\quad x_1 \sim \mathcal N(0,1)$$

**风险与缓解**：
- 显存翻倍：用更小 batch + grad accumulation；首版先把 MLP 缩到 hidden=64 验证。
- 梯度图泄漏 → NaN：必须 `du_dt.detach()`，并显式 `torch.func.functional_call`。
- 不稳定：初始化 $\lambda=0.1$，线性 warmup 到 1.0。

**对每个 FM 独立蒸馏**：得到 `MF_vol_trans`、`MF_ret_trans`。

### 3.4 Stage 2b（对照基线）：Consistency Distillation

按 Song 2023（§1.3）做相邻噪声水平的自洽蒸馏：

$$\mathcal L_{\text{CD}} = \mathbb E\Big[ d\big(f_\theta(x+\sigma_{n+1}z, \sigma_{n+1}),\; f_{\theta^-}(\hat x^\psi_{t_n}, \sigma_n)\big)\Big]$$

EMA decay schedule 按 iCT（Song 2024）建议；$d$ 取 MSE（1D 状态空间，LPIPS 不适用）。

**作用**：CD vs MF 的方法学对比是 V3 实验密度的来源之一（即使我们最终推 MF 作为推荐方法，也要有 CD 作为 baseline 报告，否则审稿/答辩会问）。

### 3.5 训练资源估算

| 组件 | epoch 数 | wall-clock（RTX 3090，FP32） |
|---|---|---|
| FM_vol_trans | 20 | ~2.5h |
| FM_ret_trans | 20 | ~3h |
| MF_vol_trans 蒸馏 | 15 | ~3h |
| MF_ret_trans 蒸馏 | 15 | ~3.5h |
| CD 对照（两边） | 15 + 15 | ~6h |
| **合计** | | **~20h** |

蒸馏因为 JVP 加倍 + 教师前向，比 FM 训练慢约 1.5×。

---

## 4. 推理：自回归滚动

```python
s0 = (log_v0, 0.0)
trajectory = [s0]
a = action_schedule(T)  # 预先给定 regime 时间序列

for t in range(T):
    log_v_t, r_t = trajectory[-1]
    cond_v = pack(log_v_t, a[t])
    log_v_next = MF_vol_trans.sample(cond_v)            # 1 NFE
    cond_r = pack(log_v_next, log_v_t, r_t, a[t])
    r_next = MF_ret_trans.sample(cond_r)                # 1 NFE
    trajectory.append((log_v_next, r_next))
```

- 单步推理 = 2 次前向；252 步全路径 = 504 次前向；批量 N=1000 时 < 1s（单 GPU）。
- 反归一化后得到 $(v_t, r_t)$，由 $r_t$ 累加得 $\log S_t$，由 $\log S_t$ 求 $S_t$。

**误差累积监控**：rollout 时同时记录每个步上 $(\hat v_t, \hat r_t)$ 与从同一 $(v_0, r_0)$ 出发的 Heston 真实条件分布的 KS 距离/Wasserstein 距离；预期会随 $t$ 缓慢增长，若发散需回去做 scheduled sampling 或在训练中加 multi-step roll-out loss（§ 5.3）。

---

## 5. 评测方案

V3 的评测分三层：**统计性质 → 路径距离 → 下游定价**。每层都对应 References 里的条目。

### 5.1 统计评测：5 Stylized Facts（Cont 2001 §4.4）

在生成的 10k 测试路径上同时计算并对比真实 Heston 路径：

| Stylized fact | 度量 | 通过阈值（建议）|
|---|---|---|
| Heavy tails | log-return kurtosis、tail index Hill estimator | kurtosis 误差 < 10% |
| Volatility clustering | $\|r_t\|$ 的 ACF（lag 1–50） | ACF 形状 Wasserstein < 0.05 |
| Leverage effect | corr($r_t$, $r^2_{t+k}$) for $k=1..10$ | 符号正确、量级 ±20% |
| Aggregational Gaussianity | weekly/monthly aggregate 的 normality test | 与真实路径同向衰减 |
| Absence of return ACF | $r_t$ 的 ACF（lag 1–20） | 全部 < 0.05 |

实现位置：拟建 `finflow/eval/stylized_facts.py`（未实现）。

### 5.2 路径距离：Wasserstein 家族

- **边际 Wasserstein-1**：每步 marginal 的 1-Wasserstein 距离
- **Sig-Wasserstein**：路径 signature（depth 4）距离，对路径整体形态敏感（Ni 2021 §4.3）

对照基线：从 Heston QE 直接采样 10k 路径作为"oracle"距离参考。

### 5.3 下游评测：期权定价 RMSE

- 用生成路径估计 $\mathbb E^{\mathbb P}[(S_T - K)^+]$（Monte Carlo）
- 对比 Carr-Madan FFT 的 ground truth（§1.5）
- 15 个 $(K,T)$ 网格点的 RMSE / MAPE

**注意 P vs Q 测度**：训练在 P 测度上（含 $\mu$ 漂移），定价需要 Q 测度。两种处理：
1. 报告 P-measure 下的 MC 价格 vs P-measure 下的 Carr-Madan 价格（用 $\mu$ 替换为 $r=0$ 重新做特征函数 FFT 即可在 Q 测度上）。
2. 用 Girsanov 调整 score（详见 [02_pipelines.md](02_pipelines.md) V2 节理论亮点）；这是 Report 中的 intellectual contribution，不必跑完整实验。

Regime-switching 数据是 Markov mixture，不再默认拿 normal-regime 的单一 Heston Carr-Madan 价格当 ground truth；评测脚本会跳过该 pricing 项，除非显式要求与 normal regime 参考做诊断性对比。

### 5.4 世界模型特有评测

- **条件路径生成**：固定 $(v_0, r_0)$ 与 $a$ schedule，生成 1k 条；验证均值/方差/分位数随 $t$ 演化匹配 Heston 条件分布。
- **动作响应**：把 $a_t$ 从 normal 切到 crash，验证 $v_t$ 的瞬时增量、$r_t$ 的尾部加厚是否符合参数变化的方向（DIAMOND 的 action conditioning 评测范式 §2.1）。
- **自回归稳定性曲线**：横轴 rollout step，纵轴 Wasserstein-1 距离；做 NFE=1（MF）、NFE=4（CD 多步）、NFE=20（教师 FM）三条曲线。

### 5.5 对比实验矩阵

| 模型 | NFE | 训练方法 | 评测项 |
|---|---|---|---|
| Heston QE oracle | — | — | 上界参考（与自己采样的两批做 Wasserstein） |
| 单步 FM 教师 | 20 | CFM | 全部 |
| MF（V3 主推） | 1 | Mean Flow 蒸馏 | 全部 |
| CD（对照） | 1 | Consistency Distillation | 统计 + 定价 |
| Quant GAN 复现 baseline | — | GAN | 统计 + 定价（不做 rollout） |

GAN baseline 是 Wiese 2020 §4.3 的直接复现，用 `pip install quant_gan_replication` 或自己写 50 行 conv-GAN 即可，作为"非扩散类"对照。

---

## 6. 实验交付物（4 页 Report + 答辩）

1. **表 1**：5 stylized facts 对照（Heston oracle vs FM teacher vs MF vs CD vs Quant GAN）
2. **表 2**：15 个 $(K,T)$ 网格点的期权定价 RMSE
3. **图 1**：自回归稳定性曲线（rollout step → Wasserstein）
4. **图 2**：动作切换响应（regime change 前后的 $v_t, r_t$ 路径 + tail 改变）
5. **图 3**：定性路径样本（Heston vs MF 各 6 条 spaghetti plot）
6. **消融**：Mean Flow $\lambda$、JVP detach 的开关、scheduled sampling 比例

---

## 7. 当前代码完成度

> 全部基于 [finflow/](../../finflow/)、[scripts/](../../scripts/)、[tests/](../../tests/) 的实际文件状态判定。

### 7.1 已完成（全部 V3 组件就位）

| 模块 | 文件 | 对应 V3 阶段 |
|---|---|---|
| Heston QE 模拟（QE 方差 + QE-M 收益率，单 / 多 regime Markov 切换） | [finflow/data/heston.py](../../finflow/data/heston.py) | §1.1–1.4 |
| 路径 → `(s_t, s_{t+1}, a_t)` 转移对扁平化 | `build_transition_arrays` | §1.3 |
| Carr-Madan FFT 定价器 + BS 参考 | [finflow/data/option_pricing.py](../../finflow/data/option_pricing.py) | §1.5、§5.3 |
| 数据生成 CLI（含 `--regimes`） | [scripts/generate_heston_data.py](../../scripts/generate_heston_data.py) | §1.6 |
| Joint / Vol / Ret 三个 transition dataset | [finflow/data/dataset.py](../../finflow/data/dataset.py) | §1.6 |
| FiLM-MLP 骨干 + sinusoidal time embedding | [finflow/models/transition_fm.py](../../finflow/models/transition_fm.py) | §2.2 |
| Joint FM `TransitionFM` + Vol/Ret 两阶段复用 | 同上 | §2.1 |
| `MeanFlowStudent` 1-NFE 模型 + 教师→学生 warm start | [finflow/models/mean_flow.py](../../finflow/models/mean_flow.py) | §3.3 |
| `ConsistencyStudent`（c_skip/c_out 边界参数化）+ warm start | [finflow/models/consistency.py](../../finflow/models/consistency.py) | §3.4 |
| 训练循环（AdamW、grad clip、tqdm 进度条 + TTY 自适应、metrics.jsonl、best/last ckpt） | [finflow/training.py](../../finflow/training.py) | §3.1–3.2 |
| Mean Flow 蒸馏（`torch.func.jvp` 计算 ∂u/∂t，boundary anchor） | [finflow/distillation/mean_flow.py](../../finflow/distillation/mean_flow.py) | §3.3 |
| Consistency Distillation（noise schedule + EMA target + teacher Euler step） | [finflow/distillation/consistency.py](../../finflow/distillation/consistency.py) | §3.4 |
| 统一 Sampler 接口（FM / MF / CD）+ checkpoint 自动分派 | [finflow/inference/samplers.py](../../finflow/inference/samplers.py) | §4 |
| 自回归 rollout + Markov-chain action 调度 | [finflow/inference/rollout.py](../../finflow/inference/rollout.py) | §4 |
| 5 stylized facts（kurtosis、ACF、leverage、aggregational、Hill tail） | [finflow/eval/stylized_facts.py](../../finflow/eval/stylized_facts.py) | §5.1 |
| Wasserstein-1（marginal + path） | [finflow/eval/distances.py](../../finflow/eval/distances.py) | §5.2 |
| MC pricing vs Carr-Madan RMSE / MAPE | [finflow/eval/pricing.py](../../finflow/eval/pricing.py) | §5.3 |
| 全套 report 组合器 | [finflow/eval/reports.py](../../finflow/eval/reports.py) | §5 |
| Quant GAN baseline（TCN + LSGAN）+ 采样器 | [finflow/baselines/quant_gan.py](../../finflow/baselines/quant_gan.py) | §5.5 |
| 全部 CLI：data / vol+ret FM / MF / CD / rollout / evaluate / Quant GAN | [scripts/](../../scripts/) | — |
| 单元测试（67 个，全部通过） | [tests/](../../tests/) | — |

### 7.2 仍待扩展（非阻塞，后续 polish）

| 优先级 | 项目 | 说明 |
|---|---|---|
| P2 | Sig-Wasserstein 路径距离 | Ni 2021；目前 [finflow/eval/distances.py](../../finflow/eval/distances.py) 仅提供 marginal/path Wasserstein-1 |
| P2 | CFG 推理 + 动作消融 | 训练时 action one-hot 已支持，推理侧加 `--cfg-w` 约 30 行 |
| P2 | Scheduled sampling for ret stage | 训练时按概率把 ground-truth $v_{t+1}$ 替换为 vol-sampler 输出 |
| — | 长跑实验报表 + 可视化 | 把 evaluate_rollout 的 JSON 拼成 4 页 Report 的 5 张表 / 3 张图 |

### 7.3 一句话总结

V3 所需的端到端组件全部就位：数据 → Stage 1（FM vol+ret）→ Stage 2（MF / CD 蒸馏）→ 自回归 rollout → 三层评测（统计 / 距离 / 定价）→ Quant GAN baseline。67 个单元测试已通过，剩下的是真实长跑、报表生成、以及 CFG / Sig-Wasserstein 等 polish。

---

## 8. 文献到模块的反查表

```
合成数据
  Heston (1993) ────────► finflow/data/heston.py（参数与结构）
  Andersen (2007) ──────► finflow/data/heston.py（_qe_variance_step、_qe_m_log_return）
  Carr-Madan (1999) ────► finflow/data/option_pricing.py
  Cont (2001) ──────────► finflow/eval/stylized_facts.py

训练模型
  Lipman (2023) ────────► finflow/models/transition_fm.py（CFM loss）
  Ho (2020) DDPM ───────► SinusoidalTimeEmbedding、骨干设计
  Perez (2018) FiLM ────► FiLMResidualBlock
  Kong (2021) DiffWave ► 1D U-Net 加强版（如启用）
  Song (2023) CM ───────► finflow/distillation/consistency.py + finflow/models/consistency.py
  Geng (2025) MeanFlow ► finflow/distillation/mean_flow.py + finflow/models/mean_flow.py
  Ho & Salimans (2022) ─► CFG 推理（sampler 已留 hook，仍未启用）
  Li (2020) Latent SDE ► 两阶段分解的理论依据

世界模型 framing
  Ha & Schmidhuber (2018) ► Report 引言
  Alonso (2024) DIAMOND ──► Report 直接对标
  Bruce (2024) Genie ─────► Report 相关工作
  Rasul (2021) TimeGrad ──► 自回归 rollout 范式

误差累积控制
  Bengio (2015) Scheduled Sampling ► scheduled sampling 设计
  Lamb (2016) Professor Forcing ───► 进阶选项

评测
  Cont (2001) ──────────► 5 stylized facts
  Wiese (2020) Quant GAN► baseline 对照
  Ni (2021) Sig-Wasser ─► Sig-Wasserstein 距离
  Carr-Madan (1999) ────► 定价 ground truth

理论拼图（Report 引用，不必实验）
  Song (2021) Score SDE ──► Probability Flow ODE 等价性
  Karatzas-Shreve / Shreve ► Girsanov P → Q 测度变换
```

---

## 9. 与 02_pipelines.md 中 V1/V2 的关系

- V1 的 CD vs MF 对比 → 作为 V3 的 §3.4 对照实验保留。
- V2 的"先 v 再 r\|v"两阶段分解 → V3 在转移核层面直接采用（§2.1）。
- V3 真正新增的是把生成对象从"整条路径"变成"一步转移"，加入 action $a_t$，把自回归 rollout 作为推理范式。

[02_pipelines.md](02_pipelines.md) 末尾原本建议"V1+V2 结合，V3 留作 Future Work"——本项目选择直接走 V3，是因为：(a) `TransitionFM` 已经把转移核框架搭好，(b) Mean Flow + 世界模型这两个 framing 在课程评分上比单纯 V1/V2 更紧扣"World Model + One-step"双主题。代价是误差累积评测（§5.4）必须做，否则审稿/答辩会直接问 252 步会不会发散。
