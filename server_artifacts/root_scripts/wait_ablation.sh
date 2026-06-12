#!/bin/bash
O=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel/eval_ablation_0601
for i in $(seq 1 240); do
  n=$(grep -lF '[done]' $O/*.log 2>/dev/null | wc -l)
  if [ "$n" -ge 4 ]; then echo ABLATION_DONE; exit 0; fi
  sleep 60
done
echo ABLATION_TIMEOUT
