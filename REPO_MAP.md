# 仓库地图(REPO_MAP)

本仓库 = 924 GPU 机 `/root/autodl-tmp/Heston-Model` 的完整重建(2026-06-12),代码、git 历史、实验结果、未提交改动全部保留。本文件仅本地可见(已加入 `.git/info/exclude`,不影响 git 状态)。

## 目录

| 路径 | 说明 |
|---|---|
| `finflow/` | 核心库:`models/`(TransitionFM、Consistency/MeanFlow student)、`training.py`、`distillation/`(CD、MeanFlow)、`eval/`(stylized facts、pricing、distances、signatures、reports)、`inference/`(rollout、samplers)、`data/`(Heston QE、期权定价、dataset)、`baselines/`(QuantGAN)、`pathwise_teacher.py`、`transforms.py` |
| `scripts/` | CLI 入口:`train_vol_trans.py`/`train_ret_trans.py`(两阶段 teacher)、`pathwise_teacher_*.py`(路径损失微调,combined=本项目方法)、`distill_consistency/mean_flow/flow_map.py`(蒸馏;flow_map=Lagrangian)、`rollout.py`/`rollout_fewstep.py`(自由滚动/少步)、`evaluate_rollout.py`、各 `run_*.sh` |
| `tests/` | pytest 单元测试(~86 个) |
| `idea/` | 设计文档:`1/`(选题、文献、方法设计)、`2/`(完整计划、V3 实现、P0–P2 结果) |
| `data/` | 数据集(688M,gitignored):三状态 Markov Heston(seed 20260530),与 `runs/experiments/p3_full_parallel/data/` 同源 |
| `runs/` | **全部实验结果(17G,gitignored)— 索引见 `runs/README.md`** |
| `logs/` | 4 个训练日志(原服务器仓库根的 p3_*.log,整理时移入) |
| `_924_uncommitted/` | p3-tuning 分支当时的未提交状态存档(10 修改+3 未跟踪+patch),见其 README |
| 根级文档 | `README.md`(项目说明)、`REPRODUCE.md`(全管线复现命令)、`RESULTS_SUMMARY.md`(结果总表)、`eval_extreme.json`、`new_teacher_distill_summary.md`(E06 独有结果摘要,未入提交) |

## git 状态说明

- **当前检出**:`consolidated-all-code`(`eab4c5f`)= 最新代码超集(Lagrangian flow-map、few-step 采样、combined 训练器都在此分支)
- **分支**:`p3-tuning-20260530`(924 当时检出,48a4b6c)、`0531-experiments-20260602`(含**未推送 commit `a6a4861`**:signature-kernel 路径损失——GitHub 上没有,推送前勿删本仓库)、`main`;远端 = github.com/ValkyrieNox/Heston-Model
- **工作区(有意保留,勿"清理")**:
  - `M scripts/distill_flow_map.py`:924 权威 worktree 上未提交的新版(改动前备份 = 旁边的 `.bak_20260603_negwarm`)
  - `?? new_teacher_distill_summary.md`、`?? _924_uncommitted/`:保全内容,可择机提交
- **本地配置**:`core.autocrlf=false`、`core.filemode=false`(Windows 必需,否则全仓库假"已修改");`finflow/**/__pycache__` 的 24 个 .pyc 是服务器上误提交进 git 的,保留勿删(删除会弄脏状态)
