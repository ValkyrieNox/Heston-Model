#!/bin/bash
cd /root/autodl-tmp/Heston-Model-pathwise-3ad5756
P=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
V=$P/training/vol_fm/vol_p3_full_parallel/checkpoints/best.pt
R=$P/training/ret_fm/ret_p3_full_parallel/checkpoints/epoch_060.pt
PY=/root/miniconda3/bin/python
for r in 2 0 1; do
  $PY scripts/rollout.py --vol-checkpoint $V --ret-checkpoint $R \
    --data-dir $P/data --output $P/viz/roll_fm_regime$r.npz \
    --n-paths 1000 --n-steps 252 --constant-action --initial-regime $r \
    --fm-n-steps 16 --device cuda >> /root/ctrl3.log 2>&1
  echo "done regime$r $(date +%T)" >> /root/ctrl3.log
done
echo ALL_DONE >> /root/ctrl3.log
