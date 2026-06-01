#!/usr/bin/env bash
# Fine-tune the current best teacher with a QGAN-style path critic.
#
# This writes only to eval_pathwise/ and training/pathwise_teacher/, leaving
# eval_lwfm/, eval_champion/, and eval_calibrated/ untouched.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="${1:-runs/experiments/p3_full_parallel}"
EXP_NAME="$(basename "$EXP_DIR")"
DATA="$EXP_DIR/data"
TRAIN="$EXP_DIR/training"
OUT="$EXP_DIR/eval_pathwise"
EVAL="$OUT/evaluation"
ORACLE="$DATA/mc_oracle.npz"
mkdir -p "$EVAL"

DEVICE="${DEVICE:-cuda}"
DELTA="${DELTA:-0.05}"
N_STEPS="${N_STEPS:-252}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-3}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
FM_N_STEPS="${FM_N_STEPS:-8}"
LR_TEACHER="${LR_TEACHER:-1e-5}"
LR_CRITIC="${LR_CRITIC:-2e-4}"
CRITIC_STEPS="${CRITIC_STEPS:-3}"
COMPILE_MODELS="${COMPILE_MODELS:-0}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
EVAL_PATHS="${EVAL_PATHS:-10000}"
MONEYNESS="${MONEYNESS:-0.85 0.90 0.95 1.00 1.05}"
MATURITIES="${MATURITIES:-0.25 0.5 1.0}"
read -r -a MN <<< "$MONEYNESS"
read -r -a MT <<< "$MATURITIES"

first_match() {
  local f
  for f in $1; do
    [[ -e "$f" ]] && { echo "$f"; return; }
  done
  echo ""
}

VOL="${VOL:-$TRAIN/vol_lwfm_d${DELTA}/vol_lwfm_d${DELTA}/checkpoints/best.pt}"
RET="${RET:-$(python3 -c "import json;print(json.load(open('$EXP_DIR/selection_ret.json'))['best']['checkpoint'])" 2>/dev/null || true)}"
[[ -z "${RET:-}" || ! -e "$RET" ]] && RET="$(first_match "$TRAIN/ret_fm/ret_${EXP_NAME}/checkpoints/last.pt $TRAIN/ret_fm/ret_${EXP_NAME}/checkpoints/best.pt")"
[[ -e "$VOL" ]] || { echo "missing VOL checkpoint: $VOL" >&2; exit 1; }
[[ -e "$RET" ]] || { echo "missing RET checkpoint: $RET" >&2; exit 1; }

RUN_NAME="${RUN_NAME:-pathwise_lwfm_d${DELTA}}"
COMPILE_ARGS=()
[[ "$COMPILE_MODELS" == "1" ]] && COMPILE_ARGS=(--compile-models --compile-mode "$COMPILE_MODE")
python3 scripts/pathwise_teacher_finetune.py \
  --vol-checkpoint "$VOL" \
  --ret-checkpoint "$RET" \
  --data-dir "$DATA" \
  --output-dir "$TRAIN/pathwise_teacher" \
  --run-name "$RUN_NAME" \
  --n-steps "$N_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --fm-n-steps "$FM_N_STEPS" \
  --lr-teacher "$LR_TEACHER" \
  --lr-critic "$LR_CRITIC" \
  --critic-steps "$CRITIC_STEPS" \
  --device "$DEVICE" \
  "${COMPILE_ARGS[@]}"

PW="$TRAIN/pathwise_teacher/$RUN_NAME/checkpoints"
python3 scripts/rollout.py \
  --vol-checkpoint "$PW/vol_best.pt" \
  --ret-checkpoint "$PW/ret_best.pt" \
  --data-dir "$DATA" \
  --output "$OUT/rollout_${RUN_NAME}.npz" \
  --n-paths "$EVAL_PATHS" \
  --n-steps "$N_STEPS" \
  --regime-actions \
  --device "$DEVICE"

python3 scripts/evaluate_rollout.py \
  --real "$DATA/test.npz" \
  --fake "$OUT/rollout_${RUN_NAME}.npz" \
  --data-dir "$DATA" \
  --mc-oracle "$ORACLE" \
  --moneynesses "${MN[@]}" \
  --maturities "${MT[@]}" \
  --signature-depth 3 \
  --output "$EVAL/eval_${RUN_NAME}.json"

python3 scripts/summarize_cfg_sweep.py "$EVAL" || true
