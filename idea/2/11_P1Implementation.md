# V3 P1 Implementation Summary

日期：2026-05-19

P1 的目标是补齐方法论层面的训练 / 评估钩子，使 P0 已经可跑通的 V3 管线具备更好的诊断能力和后续正式实验能力。

## 1. Mean Flow Boundary Curriculum

完成内容：

- `MeanFlowDistillConfig` 增加 `boundary_prob_start` / `boundary_prob_end`。
- `scripts/distill_mean_flow.py` 默认启用 P1 curriculum：`0.5 -> 0.1`。
- `mean_flow_loss` 保留原标量 API，同时新增 `mean_flow_loss_components`。
- 训练与验证日志拆分：
  - `*_boundary_loss`
  - `*_identity_loss`
  - `*_boundary_fraction`
  - `boundary_prob`
- 新增 `--identity-residual-eval`，每个 epoch 额外跑纯 identity residual 验证。

## 2. Ret Scheduled Sampling

完成内容：

- `train_ret_trans_fm` 增加 `vol_sampler_checkpoint`。
- `TransitionFMTrainConfig` 增加：
  - `scheduled_sampling_max_prob`
  - `scheduled_sampling_start_epoch`
  - `scheduled_sampling_fm_steps`
- ret 训练时按线性 schedule 替换 condition 中的 `log_v_next`：
  `p(epoch) = max_prob * progress`，默认最大到 `0.5`。
- `scripts/train_ret_trans.py` 增加：
  - `--vol-sampler-checkpoint`
  - `--scheduled-sampling-max-prob`
  - `--scheduled-sampling-start-epoch`
  - `--scheduled-sampling-fm-steps`

## 3. Classifier-Free Guidance

完成内容：

- vol / ret transition dataset 增加 `action_dropout_prob`。
- `scripts/train_vol_trans.py` 和 `scripts/train_ret_trans.py` 默认 `--action-dropout-prob 0.1`。
- FM / Mean Flow / Consistency sampler 均支持 `cfg_w`。
- unconditional branch 通过把 condition 尾部 action one-hot 置零实现。
- `autoregressive_rollout` 和 `scripts/rollout.py` 增加 `cfg_w`。

## 4. Sig-Wasserstein

完成内容：

- 新增 `finflow/eval/signatures.py`。
- 支持将 return paths 转为 `(time, cumulative return)` paths。
- 实现纯 NumPy truncated path signature，depth 限制为 `1..4`。
- `build_full_report` 默认输出 `distances.signature_wasserstein`。
- `scripts/evaluate_rollout.py` 增加 `--signature-depth`，`0` 表示关闭。
- `scripts/run_full_evaluation.sh` summary 表新增 `Sig-W1 mean` 列。

## 5. 测试

新增 / 更新测试覆盖：

- Mean Flow boundary / identity loss components。
- Mean Flow curriculum metrics 和 identity residual eval。
- ret scheduled sampling smoke。
- action dropout dataset 行为。
- sampler CFG 行为。
- Sig-Wasserstein features 和 report integration。

已跑过的局部测试：

```bash
python3 -m pytest \
  tests/test_two_stage_datasets.py \
  tests/test_samplers.py \
  tests/test_rollout.py \
  tests/test_signatures.py \
  tests/test_eval_reports.py \
  tests/test_mean_flow_distill.py \
  tests/test_two_stage_training.py -q
```

结果：`28 passed`。

全量测试：

```bash
python3 -m pytest tests/ -q
```

结果：`82 passed`。

## 6. 下一步

P1 代码层面已经完成。下一步应进入 P2：

1. 跑正式数据规模：50k / 5k / 10k paths，252 steps。
2. 用 P1 hooks 跑中等规模 smoke：
   - ret scheduled sampling 是否缩小 train / rollout gap。
   - MF identity loss 是否随 epoch 降低。
   - CFG `w in {0, 1, 2, 3}` 是否改善 regime separation。
   - Sig-Wasserstein 是否与 marginal / total W1 给出一致排序。
3. 固定正式实验表格和可视化脚本。
