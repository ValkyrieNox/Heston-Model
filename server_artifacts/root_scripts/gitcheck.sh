#!/bin/bash
for d in /root/autodl-tmp/Heston-Model /root/autodl-tmp/Heston-Model-pathwise-3ad5756 /root/autodl-tmp/Heston-Model-b4e42ed; do
  echo "======================================================"
  echo "REPO: $d"
  if [ ! -d "$d/.git" ] && [ ! -f "$d/.git" ]; then echo "  (not a git repo / worktree)"; continue; fi
  cd "$d"
  echo "-- branch / HEAD:"; git rev-parse --abbrev-ref HEAD; git log --oneline -1
  echo "-- remote:"; git remote -v | head -n 2
  echo "-- status (short):"; git status -sb | head -n 40
  echo "-- ahead/behind vs upstream:"; git rev-list --left-right --count @{u}...HEAD 2>/dev/null || echo "  (no upstream tracking)"
done
