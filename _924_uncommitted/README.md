# 924 机未提交状态存档(2026-06-12 整理时保全)

本目录原样保存 924 机主仓库(`/root/autodl-tmp/Heston-Model`,当时检出分支 `p3-tuning-20260530`,HEAD `48a4b6c`)工作区中的**全部未提交内容**,逐文件 SHA256 与备份核对。

## p3-tuning-20260530/ 内容

**10 个已修改文件**(相对 `48a4b6c` 的工作区版本,完整文件副本):
finflow/training.py, finflow/data/dataset.py, finflow/distillation/consistency.py,
finflow/distillation/mean_flow.py, scripts/distill_consistency.py, scripts/distill_mean_flow.py,
scripts/train.sh, scripts/train_ret_trans.py, scripts/train_vol_trans.py, tests/test_two_stage_training.py

**3 个未跟踪文件**:
new_teacher_distill_summary.md, scripts/pathwise_teacher_sigmmd.py, scripts/run_p3_tuning_parallel.sh

`uncommitted_vs_48a4b6c.patch` = 10 个修改文件相对 48a4b6c 的统一 diff(可读审阅用;权威内容以文件副本为准)。

## 与 consolidated-all-code(eab4c5f,本仓库当前检出)的包含关系

已被 consolidated 吸收(内容一致,无需处理):
- scripts/distill_consistency.py, scripts/distill_mean_flow.py, scripts/train_ret_trans.py
- scripts/pathwise_teacher_sigmmd.py, scripts/run_p3_tuning_parallel.sh

与 consolidated 版本**不一致**(p3-tuning 线上的独立改动,如需要可手工合并):
- finflow/training.py, finflow/data/dataset.py, finflow/distillation/consistency.py,
  finflow/distillation/mean_flow.py, scripts/train.sh, scripts/train_vol_trans.py,
  tests/test_two_stage_training.py

不在任何提交中(已同时复制到本仓库根目录):
- new_teacher_distill_summary.md

## 另:pathwise worktree 的活跃改动(已直接应用到本仓库工作区)

924 的权威代码 worktree(`Heston-Model-pathwise-3ad5756`,consolidated-all-code 检出)相对 eab4c5f 只有:
- `scripts/distill_flow_map.py` 修改版(已覆盖到本仓库,git status 中可见)
- `scripts/distill_flow_map.py.bak_20260603_negwarm`(改动前备份,已复制)
