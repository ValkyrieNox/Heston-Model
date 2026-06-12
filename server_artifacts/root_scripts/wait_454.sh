#!/bin/bash
L=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel/eval_A_0602/A.log
for i in $(seq 1 300); do
  if grep -qF 'ALLDONE' "$L" 2>/dev/null; then echo A454_DONE; exit 0; fi
  sleep 60
done
echo A454_TIMEOUT
