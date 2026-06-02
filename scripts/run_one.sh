#!/bin/bash
# $1 run-name $2 path-loss $3 weight $4 batch $5 moment(on/off) $6 epochs $7 steps
RUN=$1; LOSS=$2; W=$3; BS=${4:-96}; MOM=${5:-off}; EP=${6:-5}; ST=${7:-120}
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
BASE=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
DATA=$BASE/data; ORACLE=$DATA/mc_oracle.npz; TEST=$DATA/test.npz
OUT=$BASE/eval_ablation_0601; PY=/root/miniconda3/bin/python; LOG=$OUT/${RUN}.log
mkdir -p $OUT/evaluation; cd $WT
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
VOL=$BASE/training/vol_lwfm_d0.05/vol_lwfm_d0.05/checkpoints/best.pt
RET=$BASE/training/ret_fm/ret_p3_full_parallel/checkpoints/epoch_060.pt
MOMARGS=""; [ "$MOM" = "off" ] && MOMARGS="--moment-weight 0 --terminal-weight 0 --abs-sum-weight 0 --kurtosis-weight 0"
echo "[start] $RUN loss=$LOSS w=$W bs=$BS moment=$MOM ep=$EP st=$ST $(date -Is)" >> $LOG
$PY scripts/pathwise_teacher_pathloss.py --vol-checkpoint $VOL --ret-checkpoint $RET \
  --data-dir $DATA --output-dir $BASE/training/ablation_0601 --run-name $RUN \
  --path-loss $LOSS --path-loss-weight $W --epochs $EP --steps-per-epoch $ST \
  --batch-size $BS --fm-n-steps 4 --lr-teacher 5e-6 --freeze-vol $MOMARGS --device cuda 2>&1 | tee -a $LOG
CK=$BASE/training/ablation_0601/$RUN/checkpoints
for tag in raw cal; do
  EXTRA=""; [ "$tag" = "cal" ] && EXTRA="--calibrate-moments"
  $PY scripts/rollout.py --vol-checkpoint $CK/vol_best.pt --ret-checkpoint $CK/ret_best.pt \
    --data-dir $DATA --output $OUT/rollout_${RUN}_${tag}.npz --n-paths 5000 --n-steps 252 \
    --regime-actions --fm-n-steps 16 $EXTRA --device cuda >> $LOG 2>&1
  $PY scripts/evaluate_rollout.py --real $TEST --fake $OUT/rollout_${RUN}_${tag}.npz --data-dir $DATA \
    --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT --signature-depth 3 \
    --output $OUT/evaluation/eval_${RUN}_${tag}.json >> $LOG 2>&1
done
echo "[done] $RUN $(date -Is)" >> $LOG
