# FinFlow — V3 Heston World-Model Pipeline

A complete V3 implementation of the autoregressive financial world model
described in [idea/2/pipelines.md](idea/2/pipelines.md):

- **Data**: Andersen-QE Heston with optional 3-regime Markov mixture and a
  Carr-Madan FFT pricer for ground-truth option prices.
- **Stage 1 teachers**: two Flow Matching transition kernels
  `p(v_{t+1}|v_t, a_t)` and `p(r_{t+1}|v_{t+1}, v_t, r_t, a_t)`.
- **Stage 2 students**: Mean Flow (Geng 2025, NeurIPS Oral) and Consistency
  Distillation (Song 2023, ICML) 1-NFE generators distilled from each teacher.
- **Inference**: unified samplers (FM teacher / MF / CD) and an autoregressive
  rollout that uses any pair of vol+ret samplers interchangeably.
- **Evaluation**: 5 Cont (2001) stylized facts + marginal/path Wasserstein-1 +
  Carr-Madan vs MC pricing RMSE / MAPE for single-regime Heston data.
- **Baseline**: a minimal Quant GAN (TCN + LSGAN, Wiese 2020-style).

Design docs:
- [idea/2/pipelines.md](idea/2/pipelines.md) — V1 / V2 / V3 framing
- [idea/2/V3_References.md](idea/2/V3_References.md) — literature backing every
  data / model / evaluation choice
- [idea/2/v3_implementation.md](idea/2/v3_implementation.md) — end-to-end V3
  plan, code-progress index, literature reverse-lookup

## Layout

```
finflow/
  data/                    # Heston QE + Carr-Madan + V3 vol/ret datasets
  models/                  # TransitionFM + MeanFlowStudent + ConsistencyStudent
  training.py              # joint trainer + V3 vol/ret trainers (progress bars)
  distillation/            # Mean Flow + Consistency distillation trainers
  inference/               # unified samplers + autoregressive rollout
  eval/                    # stylized facts + distances + pricing + report builder
  baselines/               # Quant GAN (TCN + LSGAN)
scripts/                   # CLI entry points (one per command)
tests/                     # full pytest suite (67 tests)
```

## End-to-end workflow

```bash
pip install numpy torch tqdm pytest scipy

# --- 1) data ---------------------------------------------------------------
python3 scripts/generate_heston_data.py \
  --output data/heston_v3 \
  --n-train 50000 --n-val 5000 --n-test 10000 \
  --steps 252 --regimes --seed 1234

python3 scripts/price_heston_grid.py \
  --output data/heston_v3/option_grid.json

# --- 2) Stage 1: train the two FM teachers --------------------------------
python3 scripts/train_vol_trans.py \
  --data-dir data/heston_v3 --output-dir runs/vol_fm \
  --batch-size 512 --epochs 20 --lr 3e-4

python3 scripts/train_ret_trans.py \
  --data-dir data/heston_v3 --output-dir runs/ret_fm \
  --batch-size 512 --epochs 20 --lr 3e-4

# --- 3) Stage 2: 1-NFE distillation ---------------------------------------
# Mean Flow students (recommended)
python3 scripts/distill_mean_flow.py --stage vol \
  --teacher-checkpoint runs/vol_fm/<run>/checkpoints/best.pt \
  --data-dir data/heston_v3 --epochs 15 --batch-size 512

python3 scripts/distill_mean_flow.py --stage ret \
  --teacher-checkpoint runs/ret_fm/<run>/checkpoints/best.pt \
  --data-dir data/heston_v3 --epochs 15 --batch-size 512

# Consistency Distillation students (comparison baseline)
python3 scripts/distill_consistency.py --stage vol \
  --teacher-checkpoint runs/vol_fm/<run>/checkpoints/best.pt \
  --data-dir data/heston_v3 --epochs 15 --n-discretization 18

python3 scripts/distill_consistency.py --stage ret \
  --teacher-checkpoint runs/ret_fm/<run>/checkpoints/best.pt \
  --data-dir data/heston_v3 --epochs 15 --n-discretization 18

# --- 4) autoregressive rollout --------------------------------------------
python3 scripts/rollout.py \
  --vol-checkpoint runs/mf_vol_distill/<run>/checkpoints/best.pt \
  --ret-checkpoint runs/mf_ret_distill/<run>/checkpoints/best.pt \
  --data-dir data/heston_v3 \
  --output runs/rollout_mf.npz \
  --n-paths 10000 --n-steps 252 --regime-actions

# Same script also works with FM teacher or CD checkpoints (auto-detected).

# --- 5) evaluation --------------------------------------------------------
python3 scripts/evaluate_rollout.py \
  --real data/heston_v3/test.npz \
  --fake runs/rollout_mf.npz \
  --output runs/eval_mf.json \
  --moneynesses 0.85 0.9 0.95 1.0 1.05 \
  --maturities 0.25 0.5 1.0

# Regime-switching data has no single-Heston Carr-Madan reference, so this
# command reports statistical/distance metrics and marks pricing as skipped.
# Drop --regimes during data generation for a closed-form pricing RMSE run.

# --- 6) Quant GAN baseline ------------------------------------------------
python3 scripts/train_quant_gan.py \
  --data-dir data/heston_v3 --output-dir runs/quant_gan \
  --seq-len 252 --epochs 20

python3 scripts/sample_quant_gan.py \
  --checkpoint runs/quant_gan/<run>/checkpoints/best.pt \
  --output runs/quant_gan_paths.npz --n-paths 10000

python3 scripts/evaluate_rollout.py \
  --real data/heston_v3/test.npz \
  --fake runs/quant_gan_paths.npz \
  --output runs/eval_quant_gan.json
```

`num_actions` is auto-read from `metadata.json` at every step. Drop `--regimes`
on data generation to use a single fixed parameter set; everything downstream
adapts automatically.

## Outputs

`generate_heston_data.py` writes per split:

- `{split}.npz`: full simulated paths (`s_paths`, `v_paths`, `log_returns`,
  plus `actions` when `--regimes` is set)
- `{split}_transitions.npz`: flattened one-step transitions
  (`v_t`, `r_t`, `v_next`, `r_next`, `log_v_t`, `log_v_next`, optional
  `action`)
- `metadata.json`: Heston params, regime config, normalization stats,
  `num_actions`, and the canonical transition alignment
  `(v_t, r_{t-1}, a_t) -> (v_{t+1}, r_t)`

Every training / distillation script writes `runs/<stage>/<run_name>/`:

- `config.json` — full run config snapshot
- `metrics.jsonl` — per-epoch loss + epoch wall-clock
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `summary.json` — pointer to checkpoints + history + total wall-clock
- live progress bar with running loss + per-epoch summary line (auto-throttled
  in non-TTY logs)

Checkpoints carry `stage`, `num_actions`, and `extra.kind` so the inference
loader (`load_sampler_from_checkpoint`) auto-dispatches the right sampler
(FM teacher / Mean Flow / Consistency).

## Tests

```bash
python3 -m pytest tests/
```

Covers: Heston QE shape / positivity, regime simulation, Carr-Madan accuracy
(low-vol-of-vol → BS limit + monotonicity + ATM Heston), V3 vol/ret datasets,
single-stage + two-stage FM trainers, Mean Flow model + JVP-based loss +
distillation smoke, Consistency model + distillation smoke, all three samplers,
autoregressive rollout, the 5 stylized facts, Wasserstein distances, MC pricing
vs Carr-Madan for single-regime data, and Quant GAN forward + train + sample.

## Status

All V3 components defined in
[idea/2/v3_implementation.md](idea/2/v3_implementation.md) are implemented:
data + Carr-Madan, Stage 1 teachers, Mean Flow + Consistency distillation,
unified samplers + autoregressive rollout, the full evaluation suite, and a
Quant GAN baseline. Pending V3 future work: signature-based path distances,
classifier-free guidance, scheduled-sampling during ret training, and longer
empirical sweeps reported in the writeup.
