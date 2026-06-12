#!/bin/bash
L=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel/eval_distill_compare_0602/log.txt
for i in $(seq 1 200); do
  if grep -qF 'ALLDONE' "$L" 2>/dev/null; then echo DISTILL_DONE; exit 0; fi
  sleep 60
done
echo DISTILL_TIMEOUT
