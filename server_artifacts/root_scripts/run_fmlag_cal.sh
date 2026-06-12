#!/bin/bash
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
B=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
DATA=$B/data; ORACLE=$DATA/mc_oracle.npz; TEST=$DATA/test.npz
OUT=$B/eval_A_0602; PY=/root/miniconda3/bin/python; LOG=$OUT/fmlag_cal.log
cd $WT; export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
V=$B/training/A_flowmap_fmlag/fmlag_vol/checkpoints/best.pt
R=$B/training/A_flowmap_fmlag/fmlag_ret/checkpoints/best.pt
for K in 1 2 4 8; do
  $PY scripts/rollout_fewstep.py --vol-checkpoint $V --ret-checkpoint $R --data-dir $DATA \
    --output $OUT/roll_fmlag_cal_k${K}.npz --student-steps $K --n-paths 5000 --n-steps 252 \
    --regime-actions --calibrate-moments --device cuda >> $LOG 2>&1
  $PY scripts/evaluate_rollout.py --real $TEST --fake $OUT/roll_fmlag_cal_k${K}.npz --data-dir $DATA \
    --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT --signature-depth 3 \
    --output $OUT/evaluation/eval_fmlag_cal_k${K}.json >> $LOG 2>&1
done
echo "[fmlag_cal done]" >> $LOG
