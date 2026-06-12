#!/bin/bash
# B-group: lower teacher raw via higher scheduled-sampling. Reuse existing vol teacher,
# retrain ret only. args: $1 run $2 vol_teacher_ckpt $3 sched_prob $4 hidden $5 blocks $6 epochs $7 gpu
RUN=$1; VOLT=$2; SCHED=$3; HID=$4; BLK=$5; EP=$6; G=${7:-0}
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
B=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
DATA=$B/data; ORACLE=$DATA/mc_oracle.npz; TEST=$DATA/test.npz
OUT=$B/eval_B_0602; PY=/root/miniconda3/bin/python; LOG=$OUT/${RUN}.log
mkdir -p $OUT/evaluation; cd $WT
export CUDA_VISIBLE_DEVICES=$G PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
echo "[start] $RUN vol=$VOLT sched=$SCHED hid=$HID blk=$BLK ep=$EP gpu=$G $(date -Is)" >> $LOG
$PY scripts/train_ret_trans.py --data-dir $DATA --output-dir $B/training/B_0602 --run-name ${RUN}_ret \
  --batch-size 512 --epochs $EP --lr 3e-4 --hidden-dim $HID --num-blocks $BLK \
  --vol-sampler-checkpoint $VOLT --scheduled-sampling-max-prob $SCHED --scheduled-sampling-fm-steps 20 \
  --action-dropout-prob 0.1 --max-train-batches 5000 --cache-data-device --device cuda 2>&1 | tee -a $LOG
RET=$B/training/B_0602/${RUN}_ret/checkpoints/best.pt
for tag in raw cal; do
  EXTRA=""; [ "$tag" = "cal" ] && EXTRA="--calibrate-moments"
  $PY scripts/rollout.py --vol-checkpoint $VOLT --ret-checkpoint $RET --data-dir $DATA \
    --output $OUT/roll_${RUN}_${tag}.npz --n-paths 5000 --n-steps 252 --regime-actions --fm-n-steps 16 $EXTRA --device cuda >> $LOG 2>&1
  $PY scripts/evaluate_rollout.py --real $TEST --fake $OUT/roll_${RUN}_${tag}.npz --data-dir $DATA \
    --mc-oracle $ORACLE --moneynesses $MON --maturities $MAT --signature-depth 3 \
    --output $OUT/evaluation/eval_${RUN}_${tag}.json >> $LOG 2>&1
done
echo "[done] $RUN $(date -Is)" >> $LOG
