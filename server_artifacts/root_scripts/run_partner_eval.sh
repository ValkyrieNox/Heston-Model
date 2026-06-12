#!/bin/bash
# Track A1: reproduce partner's FM teacher in OUR pipeline (single-Heston, Carr-Madan exact pricing).
# Runs on CPU so it doesn't contend with the GPU-bound combined run.
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
PD=/root/autodl-tmp/partner/data/heston_v3
OUT=/root/autodl-tmp/partner/eval; PY=/root/miniconda3/bin/python; LOG=$OUT/partner_eval.log
mkdir -p $OUT; cd $WT
export PYTHONUNBUFFERED=1
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
VOL=/root/autodl-tmp/partner/runs/vol/checkpoints/best.pt
RET=/root/autodl-tmp/partner/runs/ret/checkpoints/best.pt
echo "[start] partner teacher eval $(date -Is)" >> $LOG
for tag in raw cal; do
  EXTRA=""; [ "$tag" = "cal" ] && EXTRA="--calibrate-moments"
  $PY scripts/rollout.py --vol-checkpoint $VOL --ret-checkpoint $RET --data-dir $PD \
    --output $OUT/partner_rollout_${tag}.npz --n-paths 10000 --n-steps 252 \
    --fm-n-steps 16 $EXTRA --device cpu >> $LOG 2>&1
  $PY scripts/evaluate_rollout.py --real $PD/test.npz --fake $OUT/partner_rollout_${tag}.npz \
    --data-dir $PD --moneynesses $MON --maturities $MAT --signature-depth 3 \
    --output $OUT/eval_partner_${tag}.json >> $LOG 2>&1
  echo "[eval ${tag} done] $(date -Is)" >> $LOG
done
echo "[done] partner teacher eval $(date -Is)" >> $LOG
