#!/bin/bash
# Verify: is partner raw 2.54 an under-integration (fm steps) artifact? Re-roll at fm 20 and 50 (raw).
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
PD=/root/autodl-tmp/partner/data/heston_v3
OUT=/root/autodl-tmp/partner/eval_fmcheck; PY=/root/miniconda3/bin/python; LOG=$OUT/fmcheck.log
mkdir -p $OUT/evaluation; cd $WT
export PYTHONUNBUFFERED=1
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
VOL=/root/autodl-tmp/partner/runs/vol/checkpoints/best.pt
RET=/root/autodl-tmp/partner/runs/ret/checkpoints/best.pt
for fm in 20 50; do
  echo "[fm $fm] $(date -Is)" >> $LOG
  $PY scripts/rollout.py --vol-checkpoint $VOL --ret-checkpoint $RET --data-dir $PD \
    --output $OUT/roll_fm${fm}.npz --n-paths 5000 --n-steps 252 --fm-n-steps $fm --device cuda >> $LOG 2>&1
  $PY scripts/evaluate_rollout.py --real $PD/test.npz --fake $OUT/roll_fm${fm}.npz --data-dir $PD \
    --moneynesses $MON --maturities $MAT --signature-depth 3 \
    --output $OUT/evaluation/eval_fm${fm}.json >> $LOG 2>&1
done
echo "[done] $(date -Is)" >> $LOG
