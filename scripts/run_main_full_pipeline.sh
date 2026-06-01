#!/bin/bash
# Main branch full pipeline with GPU optimization
# Reuses existing data/heston_v3 (regime-switching)
# Usage: bash scripts/run_main_full_pipeline.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/heston_v3"
RUN_TAG="main_$(date +%Y%m%d_%H%M%S)"

echo "=================================================="
echo "  Main Branch Full Pipeline (GPU-optimized)"
echo "=================================================="
echo "Project: $PROJECT_ROOT"
echo "Run tag: $RUN_TAG"
echo "Data dir: $DATA_DIR"
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo ""

# Sanity check: data must exist
if [ ! -f "$DATA_DIR/metadata.json" ]; then
  echo "ERROR: $DATA_DIR/metadata.json not found"
  echo "Please ensure regime-switching data is generated first"
  exit 1
fi

echo "Data metadata (first 20 lines):"
head -20 "$DATA_DIR/metadata.json"
echo ""

mkdir -p runs

# ============================================
# Step 1: Train vol teacher
# ============================================
echo ""
echo "=========================================="
echo "  Step 1/4: Train Vol Teacher"
echo "=========================================="
echo "Config: hidden=256, blocks=6, time_emb=64, batch=8192, lr=1e-3, epochs=20"
echo "GPU optimizations: cache-data-device + use-amp"
echo ""

VOL_OUT="runs/vol_${RUN_TAG}"

python scripts/train_vol_trans.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$VOL_OUT" \
  --batch-size 8192 \
  --epochs 20 \
  --lr 1e-3 \
  --action-dropout-prob 0.1 \
  --hidden-dim 256 \
  --time-embedding-dim 64 \
  --num-blocks 6 \
  --cache-data-device \
  --use-amp \
  --device cuda

# Find vol checkpoint
VOL_RUN_DIR=$(ls -td "$VOL_OUT"/*/ | head -1)
VOL_CKPT="${VOL_RUN_DIR}checkpoints/best.pt"

if [ ! -f "$VOL_CKPT" ]; then
  echo "ERROR: Vol checkpoint not found at $VOL_CKPT"
  exit 1
fi
echo ""
echo "Vol checkpoint: $VOL_CKPT"

# ============================================
# Step 2: Train ret teacher
# ============================================
echo ""
echo "=========================================="
echo "  Step 2/4: Train Ret Teacher"
echo "=========================================="
echo "Config: hidden=256, blocks=6, time_emb=32, batch=8192, lr=5e-4, epochs=15"
echo "Note: ret loss tends to plateau early; using 15 epochs"
echo ""

RET_OUT="runs/ret_${RUN_TAG}"

python scripts/train_ret_trans.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$RET_OUT" \
  --batch-size 8192 \
  --epochs 15 \
  --lr 5e-4 \
  --action-dropout-prob 0.1 \
  --vol-sampler-checkpoint "$VOL_CKPT" \
  --scheduled-sampling-max-prob 0.5 \
  --hidden-dim 256 \
  --time-embedding-dim 32 \
  --num-blocks 6 \
  --cache-data-device \
  --use-amp \
  --device cuda

RET_RUN_DIR=$(ls -td "$RET_OUT"/*/ | head -1)
RET_CKPT="${RET_RUN_DIR}checkpoints/best.pt"

if [ ! -f "$RET_CKPT" ]; then
  echo "ERROR: Ret checkpoint not found at $RET_CKPT"
  exit 1
fi
echo ""
echo "Ret checkpoint: $RET_CKPT"

# ============================================
# Step 3: MC oracle (regime data needs it)
# ============================================
echo ""
echo "=========================================="
echo "  Step 3/4: Generate MC Oracle"
echo "=========================================="

MC_ORACLE="$DATA_DIR/mc_oracle.npz"
if [ ! -f "$MC_ORACLE" ]; then
  echo "Generating MC oracle (100k paths)..."
  python scripts/generate_mc_oracle.py \
    --data-dir "$DATA_DIR" \
    --output "$MC_ORACLE" \
    --n-paths 100000
else
  echo "MC oracle already exists, skipping"
fi

# ============================================
# Step 4: Rollout + Evaluate
# ============================================
echo ""
echo "=========================================="
echo "  Step 4/4: Rollout + Evaluate"
echo "=========================================="

ROLLOUT_OUT="runs/rollout_${RUN_TAG}.npz"
EVAL_OUT="runs/eval_${RUN_TAG}.json"

echo "Rolling out 10000 paths..."
python scripts/rollout.py \
  --vol-checkpoint "$VOL_CKPT" \
  --ret-checkpoint "$RET_CKPT" \
  --data-dir "$DATA_DIR" \
  --output "$ROLLOUT_OUT" \
  --n-paths 10000 \
  --n-steps 252 \
  --regime-actions \
  --cfg-w 2.0

echo ""
echo "Evaluating..."
python scripts/evaluate_rollout.py \
  --real "$DATA_DIR/test.npz" \
  --fake "$ROLLOUT_OUT" \
  --mc-oracle "$MC_ORACLE" \
  --output "$EVAL_OUT" \
  --signature-depth 3 \
  --moneynesses 0.85 0.9 0.95 1.0 1.05 \
  --maturities 0.25 0.5 1.0

# ============================================
# Summary
# ============================================
echo ""
echo "=================================================="
echo "  Pipeline Complete"
echo "=================================================="
echo "Run tag: $RUN_TAG"
echo "Vol:     $VOL_CKPT"
echo "Ret:     $RET_CKPT"
echo "Rollout: $ROLLOUT_OUT"
echo "Eval:    $EVAL_OUT"
echo ""
echo "Key metrics from eval:"
python -c "
import json
with open('$EVAL_OUT') as f:
    e = json.load(f)
sf = e['stylized_facts_comparison']
print(f'  Kurtosis:           real={sf[\"kurtosis_real\"]:.3f}  fake={sf[\"kurtosis_fake\"]:.3f}  diff={sf[\"kurtosis_abs_diff\"]:.3f}')
print(f'  Vol clustering L1:  {sf[\"absolute_return_acf_l1\"]:.4f}')
print(f'  Leverage L1:        {sf[\"leverage_correlation_l1\"]:.4f}')
print(f'  Tail index diff:    {sf[\"tail_index_abs_diff\"]:.3f}')
d = e['distances']
print(f'  Marginal W mean:    {d[\"marginal_wasserstein_mean\"]:.5f}')
print(f'  Signature W mean:   {d[\"signature_wasserstein\"][\"mean\"]:.5f}')
if 'pricing_fake_vs_mc_oracle' in e:
    p = e['pricing_fake_vs_mc_oracle']
    print(f'  Pricing RMSE:       {p[\"rmse_overall\"]:.3f}')
    print(f'  Pricing MAPE:       {p[\"mape_overall\"]:.4f}  ({p[\"mape_overall\"]*100:.1f}%)')
"
echo ""
echo "View full results: cat $EVAL_OUT | python -m json.tool"
