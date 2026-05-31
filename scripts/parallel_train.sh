#!/bin/bash
# Parallel training script for Heston FM teachers
# Usage: bash scripts/parallel_train.sh

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Starting parallel training ==="
echo "Project root: $PROJECT_ROOT"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Create output directories
mkdir -p runs/vol_fm_extreme runs/ret_fm_extreme

# Start vol training in background
echo ""
echo "=== Starting vol teacher training (background) ==="
python scripts/train_vol_trans.py \
  --data-dir data/heston_v3 \
  --output-dir runs/vol_fm_extreme \
  --device cuda \
  > runs/vol_fm_extreme/train.log 2>&1 &
VOL_PID=$!
echo "Vol training PID: $VOL_PID"

# Start ret training in background
echo ""
echo "=== Starting ret teacher training (background) ==="
python scripts/train_ret_trans.py \
  --data-dir data/heston_v3 \
  --output-dir runs/ret_fm_extreme \
  --device cuda \
  > runs/ret_fm_extreme/train.log 2>&1 &
RET_PID=$!
echo "Ret training PID: $RET_PID"

echo ""
echo "=== Both trainings started ==="
echo "Vol PID: $VOL_PID (log: runs/vol_fm_extreme/train.log)"
echo "Ret PID: $RET_PID (log: runs/ret_fm_extreme/train.log)"
echo ""
echo "Monitor GPU: nvidia-smi -l 1"
echo "Monitor vol: tail -f runs/vol_fm_extreme/train.log"
echo "Monitor ret: tail -f runs/ret_fm_extreme/train.log"
echo ""
echo "Waiting for both to complete..."

# Wait for both processes
wait $VOL_PID
VOL_EXIT=$?
echo "Vol training finished with exit code: $VOL_EXIT"

wait $RET_PID
RET_EXIT=$?
echo "Ret training finished with exit code: $RET_EXIT"

if [ $VOL_EXIT -ne 0 ] || [ $RET_EXIT -ne 0 ]; then
  echo "ERROR: One or both trainings failed"
  exit 1
fi

echo ""
echo "=== Both trainings completed successfully ==="
echo ""
echo "Next steps:"
echo "1. Find checkpoint paths:"
echo "   ls runs/vol_fm_extreme/*/checkpoints/best.pt"
echo "   ls runs/ret_fm_extreme/*/checkpoints/best.pt"
echo ""
echo "2. Run rollout:"
echo "   python scripts/rollout.py --vol-checkpoint <vol_path> --ret-checkpoint <ret_path> --data-dir data/heston_v3 --output runs/rollout_extreme.npz --n-paths 1000 --n-steps 252"
echo ""
echo "3. Evaluate:"
echo "   python scripts/evaluate_rollout.py --real data/heston_v3/test.npz --fake runs/rollout_extreme.npz --output runs/eval_extreme.json"
