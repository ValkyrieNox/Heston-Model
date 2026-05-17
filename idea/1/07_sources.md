# 参考资料与链接

## LOB 生成模型（核心 baseline）
- [Painting the market — arXiv 2509.05107](https://arxiv.org/abs/2509.05107) — 图像域扩散 + inpainting，LOB-Bench 强基线
- [TRADES — arXiv 2502.07071](https://arxiv.org/html/2502.07071v2) — transformer-DDPM，事件域
  - 代码：[LeonardoBerti00/DeepMarket](https://github.com/LeonardoBerti00/DeepMarket)
- [LOB-Bench — arXiv 2502.09172](https://arxiv.org/abs/2502.09172) — ICML 2025 基准
  - 代码：[peernagy/lob_bench](https://github.com/peernagy/lob_bench)
  - 项目页：[lobbench.github.io](https://lobbench.github.io/)
- [DiffVolume — arXiv 2508.08698](https://arxiv.org/html/2508.08698) — 只做 volume 的扩散
- [DiffLOB — arXiv 2602.03776](https://arxiv.org/abs/2602.03776) — 反事实生成
- [LOB Simulation with GANs (Cont, Cucuringu) — SSRN 4512356](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4512356)

## 世界模型 / RL
- [DIAMOND — arXiv 2405.12399](https://arxiv.org/abs/2405.12399) ★ 课程参考论文
  - 代码：[eloialonso/diamond](https://github.com/eloialonso/diamond)
  - 项目页：[diamond-wm.github.io](https://diamond-wm.github.io/)
- [DAWM — arXiv 2509.19538](https://arxiv.org/abs/2509.19538) — diffusion action world model
- [DWM — arXiv 2402.03570](https://arxiv.org/abs/2402.03570) — multi-step 预测
- [Diffusion Models for RL Survey](https://github.com/apexrl/Diff4RLSurvey)

## Flow Matching / Consistency / Mean Flow
- [FM-TS — arXiv 2411.07506](https://arxiv.org/abs/2411.07506) — rectified flow 时间序列
- [TimeFlow — arXiv 2511.07968](https://arxiv.org/abs/2511.07968) — SDE-based 流匹配
- Consistency Models (Song 2023) — 课程参考
- Mean Flow (Geng 2025, NeurIPS) — 课程参考

## LOB 分类 baseline
- [DeepLOB — arXiv 1808.03668](https://arxiv.org/pdf/1808.03668)
  - 代码：[zcakhaa/DeepLOB-...](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)
  - 多模型实现：[Jeonghwan-Cheon/lob-deep-learning](https://github.com/Jeonghwan-Cheon/lob-deep-learning)
- [TLOB — arXiv 2502.15757](https://arxiv.org/html/2502.15757v3) — dual-attention transformer

## 数据集
- [LOBSTER 免费样本](https://lobsterdata.com) — AAPL/AMZN/GOOG/INTC/MSFT
- [FI-2010](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)
- TRADES-LOB（TRADES 官方发布的合成数据）：DeepMarket repo 内

## 课程本地资料
- `/volume/rhxie/waste/aigc/req/2026深度生成分组及考核要求.docx` — 硬性要求
- `/volume/rhxie/waste/aigc/PPT/Lecture 1 Introduction.pdf` — 评分、时间线
- 论文列表（docx 末尾）：Consistency Models / LLaDA / Mean Flows / Denoising basics / SANA / Video gen / 3D GS / World Model 系列
