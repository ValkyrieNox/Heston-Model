# 7 周排期与分工

起点：2026-04-21。展示选 6/07 档位。

## 人员定位
- **Person A — Data & Eval**：偏数据工程 / 金融 / 实验评测
- **Person B — Model & Training**：偏深度学习 / PyTorch / 训练调参

## 排期

| 周 | 日期 | 里程碑 | Person A | Person B |
|---|---|---|---|---|
| W1 | 4/21–4/27 | 选题、仓库、环境 | LOBSTER 样本下载；LOB-image encoder；`lob_bench` 跑通 demo | 搭仓库、Hydra/Lightning；跑通 DIAMOND 官方 demo；FM 最小代码 |
| W2 | 4/28–5/04 | **基线复现** | TRADES 官方 checkpoint 跑起来 → LOB-Bench 得分入表；FI-2010 baseline | DDPM-on-LOB-image 最小版训练一次（单股票、单日）|
| W3 | 5/05–5/11 | **FM 主干 (C1)** | 整理 eval 脚本；写 stylized-fact 可视化 | Conditional FM 替换 DDPM；收敛曲线对比 |
| W4 | 5/12–5/18 | **📝 5/14 proposal** + **action conditioning (C4.1)** | 构造 action tensor；与 TRADES 对比训练速度 | 加 action-conditioning；跑第一次 LOB-Bench 评测 |
| W5 | 5/19–5/25 | **Consistency 蒸馏 (C2) + EDM (C3)** | LOB-Bench 完整 5 股票评测；adversarial ROC | Consistency distillation；EDM 参数化；NFE=1/4 对比 |
| W6 | 5/26–6/01 | **两阶段 refine + RL agent (C4.2)** | 做市 RL 环境包装；DQN/PPO baseline | Stage-2 event refiner；联合采样 |
| W7 | 6/02–6/07 | **回测 + 报告 + 展示** | 真实数据回测 Sharpe；写 report §4 实验 | 消融表收口；写 report §3 方法；**6/07 展示** |

## Go / no-go 检查点
- **W2 末**：TRADES baseline 在 LOB-Bench 上接近官方数字 → 否则改用 FI-2010 只做分类基准
- **W4 末**：FM vs DDPM 在至少 2 个 LOB-Bench 指标上持平 → 否则砍 C2/C4，退守 "FM-for-LOB" 单一故事
- **W5 末**：Consistency 蒸馏 NFE=1 掉点 <10% → 否则只报 NFE=4
- **W6 末**：RL agent 在仿真里学到非随机策略 → 否则只报仿真 fidelity，跳过真实回测

## 交付物清单
- [ ] GitHub repo（清晰 README + 注释 + 训练/评测脚本 + 复现说明）
- [ ] 4 页 report（Motivation · Intro · Related · Method · Eval · Conclusion + 每人 bio）
- [ ] 20 min 展示 PPT（6/07）
- [ ] Proposal 1 页（5/14 前交）
- [ ] （可选）Pretrained checkpoints 上 Google Drive / HuggingFace
