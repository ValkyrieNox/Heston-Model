# Reproduce — FinFlow Heston World-Model (full pipeline)

This branch (`consolidated-all-code`) contains **all code needed to generate every model**
in our comparison: the two-stage Flow-Matching teacher, the Lambert-W variance kernel (LWFM),
the QGAN-style path-loss fine-tuning (pathwise / signature-kernel / energy / combined),
the MeanFlow & Consistency distilled students, and the Quant GAN baseline.

> Data and checkpoints are **not** in git (`.gitignore` excludes `/data/`, `/runs/`, `*.npz`, `*.pt`).
> Use the commands below to regenerate the data deterministically, then train/evaluate.

## 0. Environment
```bash
pip install numpy torch tqdm scipy
```

## 1. Data (deterministic — same seed ⇒ byte-identical to ours)
Our main task is the **3-regime Markov-switching Heston** (this is what every reported model uses):
```bash
python3 scripts/generate_heston_data.py \
  --output data/heston_v3 \
  --n-train 50000 --n-val 5000 --n-test 10000 \
  --steps 252 --regimes --seed 20260530
# MC oracle for pricing (regime data has no closed-form):
python3 scripts/generate_mc_oracle.py \
  --data-dir data/heston_v3 --output data/heston_v3/mc_oracle.npz --n-paths 100000
```
Single-Heston variant (closed-form Carr-Madan pricing; drop `--regimes`):
```bash
python3 scripts/generate_heston_data.py --output data/heston_single \
  --n-train 50000 --n-val 5000 --n-test 10000 --steps 252 --seed 20260530
```

## 2. Stage-1: FM teachers
Baseline FM teacher:
```bash
python3 scripts/train_vol_trans.py --data-dir data/heston_v3 --output-dir runs/vol_fm \
  --batch-size 512 --epochs 20 --lr 3e-4 --action-dropout-prob 0.1
python3 scripts/train_ret_trans.py --data-dir data/heston_v3 --output-dir runs/ret_fm \
  --batch-size 512 --epochs 20 --lr 3e-4 --action-dropout-prob 0.1 \
  --vol-sampler-checkpoint runs/vol_fm/<run>/checkpoints/best.pt --scheduled-sampling-max-prob 0.5
```
**Strong-recipe teacher** (bigger net + large batch + AMP — partner-style, our best teacher recipe):
```bash
python3 scripts/train_vol_trans.py --data-dir data/heston_v3 --output-dir runs/vol_strong \
  --hidden-dim 256 --num-blocks 6 --batch-size 8192 --epochs 20 --lr 1e-3 --use-amp
python3 scripts/train_ret_trans.py --data-dir data/heston_v3 --output-dir runs/ret_strong \
  --hidden-dim 256 --num-blocks 6 --batch-size 8192 --epochs 15 --lr 5e-4 --use-amp \
  --vol-sampler-checkpoint runs/vol_strong/<run>/checkpoints/best.pt --scheduled-sampling-max-prob 0.5
```
**LWFM (Lambert-W Gaussianized variance kernel)** — δ sweep:
```bash
DELTAS="0.03 0.05 0.08 0.12" bash scripts/lwfm_vol_sweep.sh runs/experiments/<exp>
```

## 3. Algorithm fine-tuning on a teacher (axis-2: our path-loss methods)
`scripts/pathwise_teacher_combined.py` swaps the WGAN-GP critic for strictly-proper path losses
(signature-kernel MMD / Sig-W1 / Energy), summed with moment matching. One flag per loss:
```bash
python3 scripts/pathwise_teacher_combined.py \
  --vol-checkpoint <vol_best.pt> --ret-checkpoint <ret_best.pt> \
  --data-dir data/heston_v3 --output-dir runs/combined --run-name combined_0602 \
  --w-sigmmd 200 --w-sigw1 5 --w-energy 10 \
  --moment-weight 1.0 --terminal-weight 1.0 --abs-sum-weight 0.25 --kurtosis-weight 0.1 \
  --epochs 10 --steps-per-epoch 240 --batch-size 512 --fm-n-steps 4 --lr-teacher 5e-6 --freeze-vol
```
Single-loss ablations: `scripts/pathwise_teacher_pathloss.py --path-loss {sig_mmd|sig_w1|energy|none} --path-loss-weight W ...`
Original QGAN-critic version: `scripts/pathwise_teacher_finetune.py ...`

## 4. Stage-2: 1-step distilled students
```bash
python3 scripts/distill_mean_flow.py --stage vol --teacher-checkpoint <vol_best.pt> --data-dir data/heston_v3 --epochs 15 --boundary-prob-start 0.5 --boundary-prob-end 0.1
python3 scripts/distill_mean_flow.py --stage ret --teacher-checkpoint <ret_best.pt> --data-dir data/heston_v3 --epochs 15 --boundary-prob-start 0.5 --boundary-prob-end 0.1
python3 scripts/distill_consistency.py --stage vol --teacher-checkpoint <vol_best.pt> --data-dir data/heston_v3 --epochs 15 --curriculum-kind ict
python3 scripts/distill_consistency.py --stage ret --teacher-checkpoint <ret_best.pt> --data-dir data/heston_v3 --epochs 15 --curriculum-kind ict
```

## 5. Baseline: Quant GAN
```bash
python3 scripts/train_quant_gan.py --data-dir data/heston_v3 --output-dir runs/quant_gan \
  --seq-len 252 --epochs 30 --d-steps-per-g 5 --gradient-penalty-weight 10 --lambert-w-delta 0.1
# (set --moment-penalty-weight 0 for the faithful Wiese-2020 ablation)
python3 scripts/sample_quant_gan.py --checkpoint runs/quant_gan/<run>/checkpoints/best.pt \
  --output runs/qgan_paths.npz --n-paths 10000   # add --no-calibrate-moments for raw
```

## 6. Rollout + Evaluation (THE eval protocol — free-running, identical for all models)
```bash
# free-running autoregressive rollout (generates full paths from noise; NO teacher forcing)
python3 scripts/rollout.py --vol-checkpoint <vol_best.pt> --ret-checkpoint <ret_best.pt> \
  --data-dir data/heston_v3 --output rollout.npz \
  --n-paths 10000 --n-steps 252 --regime-actions --fm-n-steps 20
  # add --calibrate-moments for the moment-calibrated variant
# evaluate (regime ⇒ --mc-oracle; single-Heston ⇒ omit, auto Carr-Madan)
python3 scripts/evaluate_rollout.py --real data/heston_v3/test.npz --fake rollout.npz \
  --data-dir data/heston_v3 --mc-oracle data/heston_v3/mc_oracle.npz \
  --moneynesses 0.85 0.90 0.95 1.00 1.05 --maturities 0.25 0.5 1.0 --signature-depth 3 \
  --output eval.json
```

Key metric keys in `eval.json`: `pricing_fake_vs_mc_oracle.rmse_overall` (regime) or
`pricing_fake_vs_carr_madan.rmse_overall` (single), plus `stylized_facts_comparison`,
`distances` (marginal/total/signature Wasserstein).
