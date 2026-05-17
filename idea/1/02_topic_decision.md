# 选题决策记录

## 用户约束
- 2 人小组
- 希望与 **quant research** 结合，但**不能硬凑**
- 必须落在课程要求的大选题内
- 1–2 个月可完成
- 技术面参考课程 PPT + 论文列表（但不限）

## 候选对比

### 选题 A ⭐ 选定：Diffusion / Flow Matching World Model for Limit Order Book (LOB)
- **契合方向**：World Model（第12周）+ Diffusion（第7周）+ Flow Matching（第8周）
- **量化契合度**：LOB 是 market microstructure 核心研究对象，**非硬凑**（DeepLOB 等已把 LOB tensor 当图像处理，学界先例充足）
- **可行性**：高，LOB "图像" 通常 40 levels × 100 ticks，单卡 3090/4090 可训
- **风险**：需在 proposal 里论证 "LOB 当图像" 的合理性

### 选题 B（备选）：Consistency / Mean Flow 在金融路径生成
- 直接用课程列表里的 Consistency Models / Mean Flow 论文
- 量化味浓，但 **不在课程 4 大方向内**，需提前找 TA 确认
- 作为 A 的 fallback

### 选题 C（弃选）：纯视觉 World Model 复现
- 零量化味，不满足用户诉求

## 定题：**FlowLOB**
> 把 LOB 视作图像的 Action-Conditional Flow-Matching World Model，通过 Consistency 蒸馏实现 1–4 步快速仿真，用下游 RL agent 的真实回测表现作为端到端评测。

## 为什么这个题目值得做
1. 一次性覆盖课程 4 个核心主题（Diffusion / Flow Matching / Consistency Models / World Model）
2. 直击 2025 年最新 LOB 生成模型的 5 个文档化痛点（见 04 号文件）
3. 2 人分工天然清晰（数据/评测 × 模型/训练）
4. 简历上是量化实习的硬通货（market microstructure + 生成模型 + RL backtest）
5. 开源 baseline 齐全（TRADES、DIAMOND、lob_bench 都有代码）