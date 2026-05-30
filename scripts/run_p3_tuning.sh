#!/usr/bin/env bash
# =============================================================================
# run_p3_tuning.sh — one-command P3 tuning campaign for the GPU box.
#
# Optimises the project's ORIGINAL results (two-stage regime-conditioned FM
# teacher + Mean Flow 1-NFE students), not the QGAN baseline. Just run:
#
#     bash scripts/run_p3_tuning.sh
#
# Every knob below is overridable via environment variables, e.g.:
#     DEVICE=cuda:1 FM_EPOCHS=80 bash scripts/run_p3_tuning.sh
#
# The pipeline is resumable: finished data/checkpoints are skipped on re-run.
# Phases:
#   A. full pipeline (data + teachers w/ epoch snapshots + MF/CD/QGAN) via train.sh
#   B. MC pricing oracle (regime-data ground truth)
#   C. pricing-aware checkpoint selection for the ret teacher
#   D. re-distill Mean Flow students from the SELECTED teacher
#   E. champion rollout + full evaluation table
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# ----------------------------- configuration --------------------------------
EXP_NAME="${EXP_NAME:-p3_full}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260530}"

# data scale
N_TRAIN="${N_TRAIN:-50000}"
N_VAL="${N_VAL:-5000}"
N_TEST="${N_TEST:-10000}"
STEPS="${STEPS:-252}"

# teacher capacity + schedule (the levers that matter for the original result)
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
LR="${LR:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LR_MIN="${LR_MIN:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-5}"

# epochs
FM_EPOCHS="${FM_EPOCHS:-60}"
MF_EPOCHS="${MF_EPOCHS:-25}"
CD_EPOCHS="${CD_EPOCHS:-25}"
QGAN_EPOCHS="${QGAN_EPOCHS:-30}"

# scheduled sampling is proven to give no benefit on the teacher -> off
RET_SCHEDULED_MAX_PROB="${RET_SCHEDULED_MAX_PROB:-0}"

# evaluation / selection
MC_PATHS="${MC_PATHS:-100000}"
SELECT_PATHS="${SELECT_PATHS:-3000}"
EVAL_PATHS="${EVAL_PATHS:-10000}"
CFG_W="${CFG_W:-0}"
MONEYNESS="${MONEYNESS:-0.85 0.90 0.95 1.00 1.05}"
MATURITIES="${MATURITIES:-0.25 0.5 1.0}"

# targeted Mean-Flow refinement LR (down from 3e-4 to tame extreme tails)
MF_REFINE_LR="${MF_REFINE_LR:-2e-4}"
MF_REFINE_EPOCHS="${MF_REFINE_EPOCHS:-20}"

# ----------------------------- derived paths --------------------------------
RUN_DIR="runs/experiments/$EXP_NAME"
DATA_DIR="$RUN_DIR/data"
TRAIN_DIR="$RUN_DIR/training"
ORACLE="$DATA_DIR/mc_oracle.npz"

VOL_CKPT="$TRAIN_DIR/vol_fm/vol_${EXP_NAME}/checkpoints"
RET_CKPT="$TRAIN_DIR/ret_fm/ret_${EXP_NAME}/checkpoints"
SELECTION_JSON="$RUN_DIR/selection_ret.json"
MF_SELECT_DIR="$RUN_DIR/mf_select"
EVAL_DIR="$RUN_DIR/eval_champion"

log() { printf '\n\033[1m[p3] %s\033[0m\n' "$*"; }

# ============================================================================
log "Phase A: full pipeline (data + teachers + MF/CD/QGAN)  exp=$EXP_NAME device=$DEVICE"
scripts/train.sh \
  --experiment-name "$EXP_NAME" \
  --seed "$SEED" \
  --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --fm-epochs "$FM_EPOCHS" --mf-epochs "$MF_EPOCHS" --cd-epochs "$CD_EPOCHS" \
  --qgan-epochs "$QGAN_EPOCHS" \
  --hidden-dim "$HIDDEN_DIM" --num-blocks "$NUM_BLOCKS" \
  --lr "$LR" --lr-schedule "$LR_SCHEDULE" --lr-min "$LR_MIN" \
  --save-every-epochs "$SAVE_EVERY_EPOCHS" \
  --ret-scheduled-max-prob "$RET_SCHEDULED_MAX_PROB" \
  --device "$DEVICE"

# ============================================================================
log "Phase B: MC pricing oracle ($MC_PATHS paths)"
if [[ -f "$ORACLE" ]]; then
  echo "[skip] $ORACLE exists"
else
  python3 scripts/generate_mc_oracle.py \
    --data-dir "$DATA_DIR" --output "$ORACLE" \
    --n-paths "$MC_PATHS" --device "$DEVICE"
fi

# ============================================================================
log "Phase C: pricing-aware checkpoint selection (ret teacher)"
python3 scripts/select_checkpoint.py \
  --sweep-stage ret \
  --fixed-vol-checkpoint "$VOL_CKPT/last.pt" \
  --sweep-checkpoints "$RET_CKPT"/epoch_*.pt "$RET_CKPT/last.pt" "$RET_CKPT/best.pt" \
  --data-dir "$DATA_DIR" --mc-oracle "$ORACLE" \
  --rank-by pricing_rmse --n-paths "$SELECT_PATHS" --cfg-w "$CFG_W" \
  --moneynesses $MONEYNESS --maturities $MATURITIES \
  --output "$SELECTION_JSON" --device "$DEVICE"

BEST_RET="$(python3 -c "import json,sys; print(json.load(open('$SELECTION_JSON'))['best']['checkpoint'])")"
log "selected ret teacher: $BEST_RET"

# ============================================================================
log "Phase D: re-distill Mean Flow students from the selected teacher"
python3 scripts/distill_mean_flow.py --stage vol \
  --teacher-checkpoint "$VOL_CKPT/last.pt" --data-dir "$DATA_DIR" \
  --output-dir "$MF_SELECT_DIR/vol" --run-name mf_vol_select \
  --epochs "$MF_REFINE_EPOCHS" --lr "$MF_REFINE_LR" --batch-size "$BATCH_SIZE" \
  --boundary-prob-start 0.5 --boundary-prob-end 0.1 \
  --identity-residual-eval --device "$DEVICE"

python3 scripts/distill_mean_flow.py --stage ret \
  --teacher-checkpoint "$BEST_RET" --data-dir "$DATA_DIR" \
  --output-dir "$MF_SELECT_DIR/ret" --run-name mf_ret_select \
  --epochs "$MF_REFINE_EPOCHS" --lr "$MF_REFINE_LR" --batch-size "$BATCH_SIZE" \
  --boundary-prob-start 0.5 --boundary-prob-end 0.1 \
  --identity-residual-eval --device "$DEVICE"

MF_VOL="$MF_SELECT_DIR/vol/mf_vol_select/checkpoints/best.pt"
MF_RET="$MF_SELECT_DIR/ret/mf_ret_select/checkpoints/best.pt"

# ============================================================================
log "Phase E: champion rollouts + full evaluation"
mkdir -p "$EVAL_DIR"

# champion FM teacher (vol last + pricing-selected ret)
python3 scripts/rollout.py \
  --vol-checkpoint "$VOL_CKPT/last.pt" --ret-checkpoint "$BEST_RET" \
  --data-dir "$DATA_DIR" --output "$EVAL_DIR/rollout_fm.npz" \
  --n-paths "$EVAL_PATHS" --n-steps "$STEPS" --regime-actions --cfg-w "$CFG_W" \
  --device "$DEVICE"

# refined Mean Flow 1-NFE student
python3 scripts/rollout.py \
  --vol-checkpoint "$MF_VOL" --ret-checkpoint "$MF_RET" \
  --data-dir "$DATA_DIR" --output "$EVAL_DIR/rollout_mf.npz" \
  --n-paths "$EVAL_PATHS" --n-steps "$STEPS" --regime-actions --cfg-w "$CFG_W" \
  --device "$DEVICE"

MC_ORACLE="$ORACLE" MONEYNESS="$MONEYNESS" MATURITIES="$MATURITIES" \
  scripts/run_full_evaluation.sh "$EVAL_DIR" "$DATA_DIR/test.npz" "$EVAL_DIR/evaluation"

log "DONE. Champion comparison table:"
cat "$EVAL_DIR/evaluation/summary.md"
echo
echo "[p3] selected ret teacher : $BEST_RET"
echo "[p3] selection ranking    : $SELECTION_JSON"
echo "[p3] evaluation summary    : $EVAL_DIR/evaluation/summary.md"
