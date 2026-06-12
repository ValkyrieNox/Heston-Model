#!/bin/bash
cd /root/autodl-tmp/Heston-Model
echo "== fetch origin (read-only) =="
git fetch origin --quiet && echo "fetched"
echo
echo "== all origin branches =="
git for-each-ref --format='%(refname:short) %(objectname:short)' refs/remotes/origin
echo
echo "== working-tree diff vs current HEAD (48a4b6c) --stat =="
git diff --stat
echo
echo "== does the working tree match origin/0531-tuning-branch2? (empty = identical to that pushed branch) =="
git diff --stat origin/0531-tuning-branch2 -- finflow scripts tests
echo
echo "== does it match origin/0601-branch2? =="
git diff --stat origin/0601-branch2 -- finflow scripts tests
echo
echo "== untracked files (excluding ablation scripts I added) =="
git status --porcelain | grep '^??'
