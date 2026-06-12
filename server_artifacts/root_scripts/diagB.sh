#!/bin/bash
echo "==== python processes (full cmd) ===="
ps aux | grep -E "python|pathwise|train_" | grep -v grep | sed 's/  */ /g' | cut -c1-220
echo "==== GPU process list ===="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo "==== b_after_a2 screen hardcopy ===="
screen -S b_after_a2 -X hardcopy /tmp/bhard.txt 2>/dev/null; sleep 1
tail -n 25 /tmp/bhard.txt 2>/dev/null | grep -v '^$'
echo "==== any log mentioning track B / strong / b_after ===="
ls -la /root/autodl-tmp/partner/ 2>/dev/null
find /root/autodl-tmp -name '*.log' -newermt '2026-06-02 13:30' 2>/dev/null
