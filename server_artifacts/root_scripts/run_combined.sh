#!/bin/bash
# Combined full run: moment (primary, pricing) + gentle sig_mmd + energy + sig_w1
# (stylized-facts nudge), batch 512, 10 epochs, freeze-vol, LWFM teacher. Solo (~28GB).
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
BASE=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
DATA=$BASE/data; ORACLE=$DATA/mc_oracle.npz; TEST=$DATA/test.npz
OUT=$BASE/eval_combined_0602; PY=/root/miniconda3/bin/python; LOG=$OUT/combined.log
mkdir -p $OUT/evaluation; cd $WT
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
VOL=$BASE/training/vol_lwfm_d0.05/vol_lwfm_d0.05/checkpoints/best.pt
RET=$BASE/training/ret_fm/ret_p3_full_parallel/checkpoints/epoch_060.pt
RUN=combined_moment_sigmmd_energy_sigw1_bs512_10ep_0602
echo "[start] $RUN $(date -Is)" >> $LOG
$PY scripts/pathwise_teacher_combined.py --vol-checkpoint $VOL --ret-checkpoint $RET \
  --data-dir $DATA --output-dir $BASE/training/combined_0602 --run-name $RUN \
  --w-sigmmd 200 --w-sigw1 5 --w-energy 10 \
  --moment-weight 1.0 --terminal-weight 1.0 --abs-sum-weight 0.25 --kurtosis-weight 0.1 \
  --epochs 10 --steps-per-epoch 240 --batch-size 512 --fm-n-steps 4 --lr-teacher 5e-6 \
  --freeze-vol --device cuda 2>&1 | tee -a $LOG
CK=$BASE/training/combined_0602/$RUN/checkpoints
for tag in raw cal; do
  EXTRA=""; [ "$tag" = "cal" ] && EXTRA="--calibrate-moments"
  $PY scripts/rollout.py --vol-checkpoint $CK/vol_best.pt --ret-checkpoint $CK/ret_best.pt \
    --data-dir $DATA --output $OUT/rollout_${RUN}_${tag}.npz --n-paths 5000 --n-steps 252 \
    --regime-actions --fm-n-steps 16 $EXTRA --device cuda >> $LOG 2>&1
  $PY scripts/evaluate_rollout.py --real $TEST --fake $OUT/rollout_${RUN}_${tag}.npz --data-dir $DATA \
    --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT --signature-depth 3 \
    --output $OUT/evaluation/eval_${RUN}_${tag}.json >> $LOG 2>&1
done
# also evaluate the LAST checkpoint (pricing-robust cross-check)
for tag in raw cal; do
  EXTRA=""; [ "$tag" = "cal" ] && EXTRA="--calibrate-moments"
  $PY scripts/rollout.py --vol-checkpoint $CK/vol_last.pt --ret-checkpoint $CK/ret_last.pt \
    --data-dir $DATA --output $OUT/rollout_${RUN}_last_${tag}.npz --n-paths 5000 --n-steps 252 \
    --regime-actions --fm-n-steps 16 $EXTRA --device cuda >> $LOG 2>&1
  $PY scripts/evaluate_rollout.py --real $TEST --fake $OUT/rollout_${RUN}_last_${tag}.npz --data-dir $DATA \
    --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT --signature-depth 3 \
    --output $OUT/evaluation/eval_${RUN}_last_${tag}.json >> $LOG 2>&1
done
echo "[done] $RUN $(date -Is)" >> $LOG
