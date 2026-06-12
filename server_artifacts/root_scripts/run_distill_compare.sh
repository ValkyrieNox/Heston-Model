#!/bin/bash
# Distill the COMBINED teacher with MeanFlow and Consistency, eval raw+cal.
# (FM-teacher's MF/CD already exist: eval_champion/eval_calibrated.)
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
B=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
DATA=$B/data; ORACLE=$DATA/mc_oracle.npz; TEST=$DATA/test.npz
OUT=$B/eval_distill_compare_0602; PY=/root/miniconda3/bin/python; LOG=$OUT/log.txt
mkdir -p $OUT/evaluation; cd $WT
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
CVOL=$(ls $B/training/combined_0602/*/checkpoints/vol_best.pt | head -1)
CRET=$(ls $B/training/combined_0602/*/checkpoints/ret_best.pt | head -1)
echo "[distill] teacher vol=$CVOL ret=$CRET" >> $LOG
echo "[distill] waiting for GPU to free $(date -Is)" >> $LOG
for i in $(seq 1 160); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 8000 ]; then break; fi
  sleep 30
done
echo "[distill] GPU free, start $(date -Is)" >> $LOG

distill_eval () {  # $1=kind  $2=run-name
  KIND=$1; RUN=$2
  if [ "$KIND" = "mf" ]; then
    $PY scripts/distill_mean_flow.py --stage vol --teacher-checkpoint $CVOL --data-dir $DATA --output-dir $B/training/distill_$RUN --run-name ${RUN}_vol --epochs 15 --max-train-batches 3000 --boundary-prob-start 0.5 --boundary-prob-end 0.1 --identity-residual-eval --cache-data-device --device cuda >> $LOG 2>&1
    $PY scripts/distill_mean_flow.py --stage ret --teacher-checkpoint $CRET --data-dir $DATA --output-dir $B/training/distill_$RUN --run-name ${RUN}_ret --epochs 15 --max-train-batches 3000 --boundary-prob-start 0.5 --boundary-prob-end 0.1 --identity-residual-eval --cache-data-device --device cuda >> $LOG 2>&1
  else
    $PY scripts/distill_consistency.py --stage vol --teacher-checkpoint $CVOL --data-dir $DATA --output-dir $B/training/distill_$RUN --run-name ${RUN}_vol --epochs 15 --max-train-batches 3000 --curriculum-kind ict --n-min 10 --n-max 160 --huber-c 0.03 --cache-data-device --device cuda >> $LOG 2>&1
    $PY scripts/distill_consistency.py --stage ret --teacher-checkpoint $CRET --data-dir $DATA --output-dir $B/training/distill_$RUN --run-name ${RUN}_ret --epochs 15 --max-train-batches 3000 --curriculum-kind ict --n-min 10 --n-max 160 --huber-c 0.03 --cache-data-device --device cuda >> $LOG 2>&1
  fi
  DVOL=$(ls $B/training/distill_$RUN/${RUN}_vol/checkpoints/best.pt 2>/dev/null)
  DRET=$(ls $B/training/distill_$RUN/${RUN}_ret/checkpoints/best.pt 2>/dev/null)
  for tag in raw cal; do
    EXTRA=""; [ "$tag" = "cal" ] && EXTRA="--calibrate-moments"
    $PY scripts/rollout.py --vol-checkpoint $DVOL --ret-checkpoint $DRET --data-dir $DATA --output $OUT/roll_${RUN}_${tag}.npz --n-paths 5000 --n-steps 252 --regime-actions $EXTRA --device cuda >> $LOG 2>&1
    $PY scripts/evaluate_rollout.py --real $TEST --fake $OUT/roll_${RUN}_${tag}.npz --data-dir $DATA --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT --signature-depth 3 --output $OUT/evaluation/eval_${RUN}_${tag}.json >> $LOG 2>&1
  done
  echo "[distill] $RUN done $(date -Is)" >> $LOG
}
distill_eval cd combined_cd
distill_eval mf combined_mf
echo "[ALLDONE] $(date -Is)" >> $LOG
