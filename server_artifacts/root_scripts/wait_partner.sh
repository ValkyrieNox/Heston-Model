#!/bin/bash
L=/root/autodl-tmp/partner/eval/partner_eval.log
for i in $(seq 1 120); do
  if grep -qF '[done]' "$L" 2>/dev/null; then echo PARTNER_EVAL_DONE; exit 0; fi
  sleep 30
done
echo PARTNER_EVAL_TIMEOUT
