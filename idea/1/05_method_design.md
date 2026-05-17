# FlowLOB 方法设计

## 一句话定义
> 把 LOB 视作图像的 Action-Conditional Flow-Matching World Model，通过 Consistency 蒸馏实现 1–4 步快速仿真，用下游 RL agent 的真实回测表现作为端到端评测。

## 数据
| 用途 | 数据集 | 说明 |
|---|---|---|
| 端到端复现 sanity | [TRADES-LOB](https://github.com/LeonardoBerti00/DeepMarket) | TSLA + INTC 2015-01-29/30，官方发布 |
| 主训练/评测 | [LOBSTER](https://lobsterdata.com) 免费样本 | AAPL / AMZN / GOOG / INTC / MSFT |
| 分类 baseline | [FI-2010](https://etsin.fairdata.fi) | DeepLOB 的公开基准 |
| LOB-Bench 官方测试集 | GOOG + INTC 2023-01 | 与论文直接可比 |

## 输入表示
- **Image 视图**：形状 `[C=2, L=20, T=100]`；`L` = 10 bid + 10 ask，`C` = {price, volume}
- **Event 视图**：`[N, 6]` = (price, qty, side, level, time_offset, type)，同 TRADES
- **Action 通道**：把 trader 自己下的订单事件在未来窗口 `T` 上投影为 spike image，与 LOB 图像 concat 成条件输入

## 模型（MVP）
- **主干**：UNet-2D，约 10–30M 参数（与 DIAMOND Atari 4.4M 同量级，可放大）
- **条件注入**：
  - 历史 LOB image（前 T₀=64 ticks）— cross-attn 或 channel concat
  - Action tensor — channel concat
- **训练目标**：Conditional Flow Matching loss (Lipman et al. 2023)
- **采样**：默认 NFE=4；Consistency 蒸馏后 NFE=1

## 两阶段架构 (C3)
```
[历史 LOB image + action tensor]
           │
           ▼
 Stage-1 (coarse, image domain, FM)
 → 未来 T=100 ticks 的 LOB 图像序列（并行生成）
           │
           ▼
 Stage-2 (refine, event domain, small transformer)
 → 逐 event 填充价/量/时间戳细节
           │
           ▼
 可馈入 LOB-Bench / trading sim 的标准 message 流
```

## 评测
### 主要（对齐已有工作）
- **LOB-Bench 全套**：spread、imbalance、inter-arrival L1/W1；adversarial discriminator ROC；market impact response；mid-price F1 迁移

### 消融
- DDPM vs Flow Matching vs Consistency Distillation（NFE=1/4/16/50）
- 有/无 action conditioning
- 一阶段 vs 两阶段 refine
- EDM 参数化 vs 原始 DDPM 参数化

### 下游（亮点实验）
- 在 FlowLOB 仿真器中训练 DQN/PPO 做市 agent
- 迁移到真实 LOBSTER 数据 backtest
- 指标：Sharpe、max drawdown、inventory turnover、fill ratio

## 参考 repo
- [DeepMarket / TRADES](https://github.com/LeonardoBerti00/DeepMarket) — 扩散 baseline
- [lob_bench](https://github.com/peernagy/lob_bench) — 评测
- [diamond](https://github.com/eloialonso/diamond) — EDM + 世界模型脚本
- [DeepLOB](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books) — 分类 baseline + LOB→image 预处理
