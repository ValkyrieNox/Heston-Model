#!/bin/bash
screen -S distill -X quit 2>/dev/null
sleep 2
P=/root/autodl-tmp/Heston-Model/runs/experiments/p3_full_parallel
rm -rf "$P/training/distill_combined_cd" "$P/training/distill_combined_mf" "$P/eval_distill_compare_0602"
screen -dmS distill bash /root/run_distill_compare.sh
sleep 2
screen -ls
echo relaunched
