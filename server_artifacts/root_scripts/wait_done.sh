#!/bin/bash
OUT=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel/eval_compare_0601
for i in $(seq 1 130); do
  if grep -qF '[done]' "$OUT/compare.log" 2>/dev/null; then echo BATCH_DONE; exit 0; fi
  sleep 30
done
echo BATCH_TIMEOUT
