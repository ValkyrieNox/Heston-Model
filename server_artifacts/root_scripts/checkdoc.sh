#!/bin/bash
cd /root/autodl-tmp/Heston-Model
echo "== distill_summary on any origin branch? =="
for b in main 0531-tuning-branch2 0601-branch2 0601-for-distill p3-tuning-20260530; do
  c=$(git ls-tree -r --name-only origin/$b 2>/dev/null | grep -c distill_summary)
  echo "  origin/$b: $c"
done
echo "== file size + head =="
ls -la new_teacher_distill_summary.md
echo "----"
head -n 20 new_teacher_distill_summary.md
