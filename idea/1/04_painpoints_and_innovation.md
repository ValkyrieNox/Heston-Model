# 现有工作痛点与我们的创新靶点

## 5 个文档化痛点（按优先级）

1. **采样速度慢** — TRADES 6h / 仿真 h；DDIM-1 步劣化 2.6×
2. **长程漂移 (snowballing errors)** — LOB-Bench 实证：prediction horizon 越长模型越差
3. **局部细节缺失** — "Painting the Market" 作者自述"重全局结构、弱 local 细节"；LOB-Bench 显示合成数据**下游分类精度下降**
4. **没有 action-conditioning** — 所有现有 LOB 生成模型都是 open-loop 仿真，无法当 RL world model 用
5. **评测碎片化** — 很多新论文跳过 LOB-Bench；我们全程用 LOB-Bench 自动拿分

## 创新定位（四个贡献对应四个课程主题）

| 编号 | 贡献 | 解决的痛点 | 对应课程 |
|---|---|---|---|
| **C1** | Flow Matching 主干替代 DDPM | #1 速度 | 第 8 周 Flow Matching |
| **C2** | Consistency 蒸馏做 1-step 采样 | #1 速度 | 参考列表：Consistency Models / Mean Flow |
| **C3** | EDM + 两阶段（image coarse → event fine）| #2 漂移、#3 细节 | 第 7 周 Diffusion + DIAMOND 范式 |
| **C4** | Action-conditional + RL agent 真实回测 | #4 可用性、#5 评测严谨性 | 第 12 周 World Model |

## 和量化的结合点
- LOB 是 market microstructure 的标准研究对象
- C4 的 RL agent 在仿真里学做市/短周期择时 → 迁移到真实 LOBSTER 数据做 PnL / Sharpe 回测 = 标准 quant research workflow
- **非硬凑**：DeepLOB（IEEE TSP 2019）已确立 LOB-as-image 范式；DiffLOB / Painting-the-Market 已走图像域扩散

## 最低可交付（MVP / fallback 路径）
只做 C1 + C3 + LOB-Bench 评测 = 一篇完整 4 页 report。C2、C4 是加分项，W5 / W6 若阻塞可砍。
