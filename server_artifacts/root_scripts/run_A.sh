#!/bin/bash
# A-group on 454 (1 GPU, sequential): Lagrangian distill (FM teacher + strong+combined)
# with NFE sweep 1/2/4/8, plus CD distill of strong+combined.
WT=/root/autodl-tmp/Heston-Model-pathwise-3ad5756
B=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
DATA=$B/data; ORACLE=$DATA/mc_oracle.npz; TEST=$DATA/test.npz
OUT=$B/eval_A_0602; PY=/root/miniconda3/bin/python; LOG=$OUT/A.log
mkdir -p $OUT/evaluation; cd $WT
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MON="0.85 0.90 0.95 1.00 1.05"; MAT="0.25 0.5 1.0"
evalnpz () { $PY scripts/evaluate_rollout.py --real $TEST --fake "$1" --data-dir $DATA --mc-oracle $ORACLE \
  --moneynesses $MON --maturities $MAT --signature-depth 3 --output "$2" >> $LOG 2>&1; }

FMV=$B/training/vol_fm/vol_p3_full_parallel/checkpoints/best.pt
FMR=$B/training/ret_fm/ret_p3_full_parallel/checkpoints/best.pt
SCV=$B/training/b_partner_recipe_pathwise_0602/b_partner_recipe_pathwise_combined_bs512_10ep_0602/checkpoints/vol_best.pt
SCR=$B/training/b_partner_recipe_pathwise_0602/b_partner_recipe_pathwise_combined_bs512_10ep_0602/checkpoints/ret_best.pt

lagrangian () {  # $1 tag  $2 vol_teacher  $3 ret_teacher
  TAG=$1
  $PY scripts/distill_flow_map.py --data-dir $DATA --teacher-checkpoint $2 --stage vol \
    --output-dir $B/training/A_flowmap_$TAG --run-name ${TAG}_vol --epochs 15 --max-train-batches 3000 --cache-data-device --device cuda >> $LOG 2>&1
  $PY scripts/distill_flow_map.py --data-dir $DATA --teacher-checkpoint $3 --stage ret \
    --output-dir $B/training/A_flowmap_$TAG --run-name ${TAG}_ret --epochs 15 --max-train-batches 3000 --cache-data-device --device cuda >> $LOG 2>&1
  V=$B/training/A_flowmap_$TAG/${TAG}_vol/checkpoints/best.pt
  R=$B/training/A_flowmap_$TAG/${TAG}_ret/checkpoints/best.pt
  for K in 1 2 4 8; do
    $PY scripts/rollout_fewstep.py --vol-checkpoint $V --ret-checkpoint $R --data-dir $DATA \
      --output $OUT/roll_${TAG}_k${K}.npz --student-steps $K --n-paths 5000 --n-steps 252 --regime-actions --device cuda >> $LOG 2>&1
    evalnpz $OUT/roll_${TAG}_k${K}.npz $OUT/evaluation/eval_${TAG}_k${K}.json
  done
  echo "[A] lagrangian $TAG done $(date -Is)" >> $LOG
}

echo "[A start] $(date -Is)" >> $LOG
lagrangian fmlag $FMV $FMR
lagrangian sclag $SCV $SCR

# CD distill of strong+combined (1-NFE), raw+cal
$PY scripts/distill_consistency.py --stage vol --teacher-checkpoint $SCV --data-dir $DATA \
  --output-dir $B/training/A_cd_sc --run-name cdsc_vol --epochs 15 --curriculum-kind ict --n-min 10 --n-max 160 --huber-c 0.03 --max-train-batches 3000 --cache-data-device --device cuda >> $LOG 2>&1
$PY scripts/distill_consistency.py --stage ret --teacher-checkpoint $SCR --data-dir $DATA \
  --output-dir $B/training/A_cd_sc --run-name cdsc_ret --epochs 15 --curriculum-kind ict --n-min 10 --n-max 160 --huber-c 0.03 --max-train-batches 3000 --cache-data-device --device cuda >> $LOG 2>&1
CV=$B/training/A_cd_sc/cdsc_vol/checkpoints/best.pt
CR=$B/training/A_cd_sc/cdsc_ret/checkpoints/best.pt
for tag in raw cal; do
  EXTRA=""; [ "$tag" = "cal" ] && EXTRA="--calibrate-moments"
  $PY scripts/rollout.py --vol-checkpoint $CV --ret-checkpoint $CR --data-dir $DATA \
    --output $OUT/roll_cdsc_${tag}.npz --n-paths 5000 --n-steps 252 --regime-actions $EXTRA --device cuda >> $LOG 2>&1
  evalnpz $OUT/roll_cdsc_${tag}.npz $OUT/evaluation/eval_cdsc_${tag}.json
done
echo "[A ALLDONE] $(date -Is)" >> $LOG
