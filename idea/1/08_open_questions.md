# 待决定项、风险、fallback

## 待决定（需要用户/队友确认）
1. **数据规模**：只用 LOBSTER 免费样本，还是学校有付费更大样本？付费数据能支撑跨股票泛化实验
2. **算力**：至少 1× 24GB GPU（3090/4090/A5000）跑 7 周；2× 更舒服。具体可用什么资源？
3. **队友**：另一位成员是否已定？建议按"Data&Eval × Model&Training"分工

## 主要风险
| 风险 | 触发条件 | 应对 |
|---|---|---|
| TRADES / LOB-Bench 环境跑不起来 | W1 末 | 改用 FI-2010 + DeepLOB 分类赛道做 baseline，放弃"世界模型"线 |
| FM 对 LOB 收敛慢 | W3 训练曲线差 | 回退 DDPM（仍能复现 + EDM 改进）|
| Consistency 蒸馏掉点严重 | W5 NFE=1 差距 >10% | 只报 NFE=4，强调 FM 本身 |
| RL agent 学不到策略 | W6 末 | 只报仿真 fidelity（LOB-Bench 指标），跳过真实回测 |
| TA 不认可 "LOB 当 AIGC" | Proposal 审核 | 备选论据：DeepLOB / DiffLOB / Painting-the-Market 已成系列；或改用 B 选题（Consistency for Path Generation）|

## Fallback 路径（按优先级降级）
1. **完整版**：C1 + C2 + C3 + C4（含真实回测）— 简历级成果
2. **标准版**：C1 + C3 + LOB-Bench 全套评测 — 足以拿高分
3. **最小版**：FM-for-LOB（替换 DDPM）+ 基础 LOB-Bench 指标 — 一篇可交报告
4. **备选选题 B**：Consistency / Mean Flow for financial path generation — 若 A 整体卡壳

## 工程注意事项
- LOBSTER 原始数据每只股票每天几百 MB，预处理存成 `.npy` tensor
- LOB image 窗口 `T=100 ticks` × `L=20 levels` × `C=2` = 4000 floats / 样本，适合 batch 256 以上
- 用 `lightning + hydra + wandb` 作为实验管理底座（TRADES 和 DIAMOND 都用）
- 训练单元：先用 1 只股票 1 天数据跑 overfit 测试，再放开

## Proposal 要写清的 3 件事
1. 为什么"LOB 当图像"不是硬凑（引 DeepLOB + Painting the Market）
2. 四个贡献如何各自对应课程内容（第 7/8/12 周 + Consistency 参考论文）
3. 评测用 LOB-Bench（表明有标准化、可比较的量化结论）
