#!/bin/bash
# Orchestrate the comparison experiments requested by the user.
# Runs sequentially in a screen; logs every step. Does NOT use set -e so a
# single failure does not abort the rest.

REPO=/root/autodl-tmp/Heston-Model
BASE=$REPO/runs/experiments/p3_full_parallel
DATA=$BASE/data
ORACLE=$DATA/mc_oracle.npz
TEST=$DATA/test.npz
OUT=$BASE/eval_compare_0601
PY=/root/miniconda3/bin/python
LOG=$OUT/compare.log

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
mkdir -p $OUT/evaluation
cd $REPO

MON="0.85 0.90 0.95 1.00 1.05"
MAT="0.25 0.5 1.0"

evalnpz () {  # $1 = fake npz, $2 = output eval json
  $PY scripts/evaluate_rollout.py --real $TEST --fake "$1" --data-dir $DATA \
    --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT \
    --signature-depth 3 --output "$2" >> $LOG 2>&1
}

echo "[start] $(date -Is)" | tee -a $LOG

# ============================================================
# 1) LWFM d0.05 : regenerate raw + calibrated with identical settings
# ============================================================
VOL=$BASE/training/vol_lwfm_d0.05/vol_lwfm_d0.05/checkpoints/best.pt
RET=$BASE/training/ret_fm/ret_p3_full_parallel/checkpoints/last.pt
echo "[1] lwfm d0.05 raw rollout $(date -Is)" | tee -a $LOG
$PY scripts/rollout.py --vol-checkpoint $VOL --ret-checkpoint $RET --data-dir $DATA \
  --output $OUT/rollout_lwfm_d0.05_raw.npz --n-paths 10000 --n-steps 252 \
  --regime-actions --fm-n-steps 16 --device cuda >> $LOG 2>&1
evalnpz $OUT/rollout_lwfm_d0.05_raw.npz $OUT/evaluation/eval_lwfm_d0.05_raw.json
echo "[1] lwfm d0.05 calibrated rollout $(date -Is)" | tee -a $LOG
$PY scripts/rollout.py --vol-checkpoint $VOL --ret-checkpoint $RET --data-dir $DATA \
  --output $OUT/rollout_lwfm_d0.05_cal.npz --n-paths 10000 --n-steps 252 \
  --regime-actions --fm-n-steps 16 --calibrate-moments --device cuda >> $LOG 2>&1
evalnpz $OUT/rollout_lwfm_d0.05_cal.npz $OUT/evaluation/eval_lwfm_d0.05_cal.json

# ============================================================
# 2) QGAN-ours (existing checkpoint) : fill missing BEST raw
# ============================================================
QO=$BASE/training/quant_gan/quant_gan_p3_full_parallel/checkpoints
echo "[2] qgan-ours best raw sample $(date -Is)" | tee -a $LOG
$PY scripts/sample_quant_gan.py --checkpoint $QO/best.pt \
  --output $OUT/qgan_ours_best_raw.npz --n-paths 10000 --no-calibrate-moments \
  --device cpu --seed 0 >> $LOG 2>&1
evalnpz $OUT/qgan_ours_best_raw.npz $OUT/evaluation/eval_qgan_ours_best_raw.json

# ============================================================
# 3) QGAN-paper-faithful (Wiese 2020): moment penalty OFF, plain head
# ============================================================
echo "[3] TRAIN qgan-paper (moment-penalty 0) $(date -Is)" | tee -a $LOG
$PY scripts/train_quant_gan.py --data-dir $DATA \
  --output-dir $BASE/training/quant_gan_paper --run-name qgan_paper_moment0_0601 \
  --seq-len 252 --epochs 30 --d-steps-per-g 5 --gradient-penalty-weight 10 \
  --lambert-w-delta 0.1 --moment-penalty-weight 0 --device cuda >> $LOG 2>&1
PAPER=$BASE/training/quant_gan_paper/qgan_paper_moment0_0601/checkpoints
echo "[3] qgan-paper sample raw + calibrated (best & last) $(date -Is)" | tee -a $LOG
for tag in best last; do
  $PY scripts/sample_quant_gan.py --checkpoint $PAPER/$tag.pt \
    --output $OUT/qgan_paper_${tag}_raw.npz --n-paths 10000 --no-calibrate-moments \
    --device cpu --seed 0 >> $LOG 2>&1
  evalnpz $OUT/qgan_paper_${tag}_raw.npz $OUT/evaluation/eval_qgan_paper_${tag}_raw.json
  $PY scripts/sample_quant_gan.py --checkpoint $PAPER/$tag.pt \
    --output $OUT/qgan_paper_${tag}_cal.npz --n-paths 10000 \
    --device cpu --seed 0 >> $LOG 2>&1
  evalnpz $OUT/qgan_paper_${tag}_cal.npz $OUT/evaluation/eval_qgan_paper_${tag}_cal.json
done

echo "[done] $(date -Is)" | tee -a $LOG
