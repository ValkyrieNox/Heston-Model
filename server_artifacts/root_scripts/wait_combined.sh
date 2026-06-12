#!/bin/bash
L=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel/eval_combined_0602/combined.log
for i in $(seq 1 300); do
  if grep -qF '[done]' "$L" 2>/dev/null; then echo COMBINED_DONE; exit 0; fi
  sleep 60
done
echo COMBINED_TIMEOUT
