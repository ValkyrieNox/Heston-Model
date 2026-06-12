# 仓库地图(REPO_MAP)

FinFlow — 基于流匹配的 Heston 世界模型:两阶段 FM teacher + 严格 proper 路径损失微调(sig-MMD / Sig-W1 / Energy,本项目方法)+ 少步蒸馏(CD / Lagrangian flow-map),与 GARCH-t / block-bootstrap / QGAN / DDPM 基线全面对比。

本仓库 = 完整项目工作区(2026-06-12 由 924 GPU 机完整镜像重建):代码 + git 历史 + 实验结果 + 论文 + 分析管线 + 汇报材料。**大型二进制(数据集 npz、训练 checkpoint、tar 包)被 .gitignore 排除,仅存在于本地工作区与本地 archive 备份中;GitHub 上是代码、文档、图表与全部评测 JSON。**

## 目录

| 路径 | 说明 | GitHub 上 |
|---|---|---|
| `finflow/` | 核心库:`models/`(TransitionFM、CD/MeanFlow student)、`training.py`、`distillation/`、`eval/`(stylized facts、pricing、distances、signatures)、`inference/`(rollout、samplers)、`data/`(Heston QE、期权定价)、`baselines/`(QuantGAN)、`pathwise_teacher.py` | ✓ |
| `scripts/` | CLI 入口:两阶段 teacher 训练、`pathwise_teacher_*.py`(combined=本方法)、蒸馏(consistency/mean_flow/flow_map)、rollout(+fewstep)、评测、各 run_*.sh | ✓ |
| `tests/` | pytest 单元测试 | ✓ |
| `idea/` | 设计文档(选题、文献、方法设计、V3 实现、P0–P2 结果) | ✓ |
| `paper/` | 论文:`main.tex`(英)/`main_zh.tex`(中,XeLaTeX)+ references.bib + 成品 PDF | ✓ |
| `analysis/` | 图表管线:`make_figures.py` + `figures/`(fig1–9)+ `eval_json_backup/`(约 200 个评测 JSON,论文数字全部可复算)+ `viz_data/`、`p3_full_parallel_data/`(npz 本地) | ✓(npz/tar 除外) |
| `presentation/` | 课程 PPT + 课程汇报演讲稿(.md/.pdf) | ✓ |
| `server_artifacts/` | 924 服务器仓库外工件:`root_scripts/`(运行脚本+REPRODUCE 副本)、`vizdl/`、`partner/`(单 Heston 评测) | ✓(脚本/JSON;npz/log 本地) |
| `data/` | 数据集 688M(三状态 Markov Heston,seed 20260530) | ✗ 本地 |
| `runs/` | **全部实验结果 17G — 索引见 `runs/README.md`(索引本身已提交)** | ✗ 本地(仅索引) |
| `logs/` | 4 个训练日志 | ✗ 本地 |
| `_924_uncommitted/` | 924 机 p3-tuning 分支当时未提交状态的存档(10 修改+3 未跟踪+patch),见其 README | ✓ |
| 根级文档 | `README.md`、`REPRODUCE.md`(复现命令)、`RESULTS_SUMMARY.md`(结果总表)、`eval_extreme.json`、`new_teacher_distill_summary.md` | ✓ |

## 数据与结果的本地权威副本

- 本地工作区:`runs/`(17G)、`data/`(688M)、analysis 与 server_artifacts 中的 npz
- 最终保险:`../archive/924机_backup/`(924 机逐字节镜像,17.8G,**不在 git 中**)
- 数据可再生:`scripts/generate_heston_data.py --regimes --seed 20260530` + `generate_mc_oracle.py`

## git 说明

- `main` = 最新完整状态(由 `consolidated-all-code` 快进而来,并已合并 `0531-experiments-20260602` 的 signature-kernel 提交 a6a4861)
- 历史分支:`p3-tuning-20260530`(实验期检出)、`0531-experiments-20260602`、`consolidated-all-code`
- 本地配置:`core.autocrlf=false`、`core.filemode=false`(Windows 必需);`finflow/**/__pycache__` 的 .pyc 为历史误提交,保留

## 关键结果速查(raw / cal / kurt;协议见 REPRODUCE.md,定价下限 0.165)

- FM teacher:raw **0.475**(最佳)/ 0.583 / 3.88;combined 微调:cal **0.232**;combined-CD 蒸馏:cal **0.170**(≈下限);FM-CD:kurt **4.48**(最佳形状);B1(sched0.8):0.872/0.348/4.41(最佳平衡)
- 系统性发现:raw ↔ cal/kurt 折中,无单一全胜模型
