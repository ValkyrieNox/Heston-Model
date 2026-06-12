#!/bin/bash
# E1: signature-kernel MMD pathwise finetune (paper 1). Self-sequences after the
# QGAN-paper compare batch frees the GPU. Fast ablation route: 5 epochs.
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
BASE=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
DATA=$BASE/data; ORACLE=$DATA/mc_oracle.npz; TEST=$DATA/test.npz
OUT=$BASE/eval_e1_sigmmd; PY=/root/miniconda3/bin/python; LOG=$OUT/e1.log
mkdir -p $OUT/evaluation; cd $WT
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
VOL=$BASE/training/vol_lwfm_d0.05/vol_lwfm_d0.05/checkpoints/best.pt
RET=$BASE/training/ret_fm/ret_p3_full_parallel/checkpoints/epoch_060.pt

# wait for the compare batch (QGAN-paper) to finish to avoid GPU contention
echo "[e1] waiting for compare batch to finish $(date -Is)" | tee -a $LOG
for i in $(seq 1 90); do
  if grep -qF '[done]' $BASE/eval_compare_0601/compare.log 2>/dev/null; then break; fi
  sleep 20
done
echo "[e1] start $(date -Is)" | tee -a $LOG

evalnpz () { $PY scripts/evaluate_rollout.py --real $TEST --fake "$1" --data-dir $DATA \
  --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT --signature-depth 3 \
  --output "$2" >> $LOG 2>&1; }

run_one () {  # $1=run-name  $2=sig-mmd-weight
  RUN=$1
  $PY scripts/pathwise_teacher_sigmmd.py --vol-checkpoint $VOL --ret-checkpoint $RET \
    --data-dir $DATA --output-dir $BASE/training/pathwise_sigmmd --run-name $RUN \
    --epochs 5 --steps-per-epoch 240 --batch-size 512 --fm-n-steps 4 --lr-teacher 5e-6 \
    --sig-mmd-weight $2 --freeze-vol --device cuda 2>&1 | tee -a $LOG
  CK=$BASE/training/pathwise_sigmmd/$RUN/checkpoints
  $PY scripts/rollout.py --vol-checkpoint $CK/vol_best.pt --ret-checkpoint $CK/ret_best.pt \
    --data-dir $DATA --output $OUT/rollout_${RUN}_raw.npz --n-paths 5000 --n-steps 252 \
    --regime-actions --fm-n-steps 16 --device cuda >> $LOG 2>&1
  evalnpz $OUT/rollout_${RUN}_raw.npz $OUT/evaluation/eval_${RUN}_raw.json
  $PY scripts/rollout.py --vol-checkpoint $CK/vol_best.pt --ret-checkpoint $CK/ret_best.pt \
    --data-dir $DATA --output $OUT/rollout_${RUN}_cal.npz --n-paths 5000 --n-steps 252 \
    --regime-actions --fm-n-steps 16 --calibrate-moments --device cuda >> $LOG 2>&1
  evalnpz $OUT/rollout_${RUN}_cal.npz $OUT/evaluation/eval_${RUN}_cal.json
}

# E1 = sig-MMD (weight 10);  control = moment-only (weight 0) to isolate paper-1's contribution
run_one e1_sigmmd_w10_5ep 10
run_one e1_control_momentonly_5ep 0
echo "[e1][done] $(date -Is)" | tee -a $LOG
