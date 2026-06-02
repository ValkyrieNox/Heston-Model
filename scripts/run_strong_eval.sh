#!/bin/bash
# Evaluate the BARE strong-recipe teacher (Track B stage-1) on our regime data. Axis-1 answer.
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
B=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
DATA=$B/data; ORACLE=$DATA/mc_oracle.npz; TEST=$DATA/test.npz
OUT=$B/eval_b_strongteacher; PY=/root/miniconda3/bin/python; LOG=$OUT/log.txt
mkdir -p $OUT/evaluation; cd $WT
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
VOL=$B/training/b_partner_recipe_teacher_0602/b_partner_recipe_vol_256x6_bs8192_0602/checkpoints/best.pt
RET=$B/training/b_partner_recipe_teacher_0602/b_partner_recipe_ret_256x6_bs8192_sched_0602/checkpoints/best.pt
echo "[start] $(date -Is)" >> $LOG
for tag in raw cal; do
  EXTRA=""; [ "$tag" = "cal" ] && EXTRA="--calibrate-moments"
  $PY scripts/rollout.py --vol-checkpoint $VOL --ret-checkpoint $RET --data-dir $DATA \
    --output $OUT/roll_${tag}.npz --n-paths 5000 --n-steps 252 --regime-actions \
    --fm-n-steps 20 $EXTRA --device cuda >> $LOG 2>&1
  $PY scripts/evaluate_rollout.py --real $TEST --fake $OUT/roll_${tag}.npz --data-dir $DATA \
    --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT --signature-depth 3 \
    --output $OUT/evaluation/eval_strongteacher_${tag}.json >> $LOG 2>&1
done
echo "[done] $(date -Is)" >> $LOG
