#!/usr/bin/env bash
# Parallel P3 tuning campaign for one GPU.
#
# This keeps the same experiment semantics as run_p3_tuning.sh, but starts
# independent downstream jobs as soon as their checkpoints/data are available.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

EXP_NAME="${EXP_NAME:-p3_full_parallel}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-20260530}"

N_TRAIN="${N_TRAIN:-50000}"
N_VAL="${N_VAL:-5000}"
N_TEST="${N_TEST:-10000}"
STEPS="${STEPS:-252}"

HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
LR="${LR:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LR_MIN="${LR_MIN:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
CACHE_DATA_DEVICE="${CACHE_DATA_DEVICE:-1}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-5}"

FM_EPOCHS="${FM_EPOCHS:-60}"
MF_EPOCHS="${MF_EPOCHS:-25}"
CD_EPOCHS="${CD_EPOCHS:-25}"
QGAN_EPOCHS="${QGAN_EPOCHS:-30}"
RET_SCHEDULED_MAX_PROB="${RET_SCHEDULED_MAX_PROB:-0}"

MC_PATHS="${MC_PATHS:-100000}"
SELECT_PATHS="${SELECT_PATHS:-3000}"
EVAL_PATHS="${EVAL_PATHS:-10000}"
CFG_W="${CFG_W:-0}"
MONEYNESS="${MONEYNESS:-0.85 0.90 0.95 1.00 1.05}"
MATURITIES="${MATURITIES:-0.25 0.5 1.0}"

MF_REFINE_LR="${MF_REFINE_LR:-2e-4}"
MF_REFINE_EPOCHS="${MF_REFINE_EPOCHS:-20}"

RUN_DIR="runs/experiments/$EXP_NAME"
DATA_DIR="$RUN_DIR/data"
TRAIN_DIR="$RUN_DIR/training"
META_DIR="$RUN_DIR/metadata"
LOG_DIR="$META_DIR/logs/parallel"
ORACLE="$DATA_DIR/mc_oracle.npz"

VOL_CKPT="$TRAIN_DIR/vol_fm/vol_${EXP_NAME}/checkpoints"
RET_CKPT="$TRAIN_DIR/ret_fm/ret_${EXP_NAME}/checkpoints"
SELECTION_JSON="$RUN_DIR/selection_ret.json"
MF_SELECT_DIR="$RUN_DIR/mf_select"
EVAL_DIR="$RUN_DIR/eval_champion"

CACHE_DATA_ARGS=()
if [[ "$CACHE_DATA_DEVICE" == "1" ]]; then
  CACHE_DATA_ARGS=(--cache-data-device)
fi

mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$META_DIR" "$LOG_DIR"

log() { printf '\n\033[1m[p3-par] %s\033[0m\n' "$*"; }

quote_cmd() {
  printf "%q " "$@"
}

run_step() {
  local name="$1"
  local marker="$2"
  shift 2
  local log_path="$LOG_DIR/${name}.log"
  if [[ -e "$marker" ]]; then
    echo "[skip] $name -> $marker"
    return 0
  fi
  echo "[run] $name"
  {
    echo "step=$name"
    printf "command="
    quote_cmd "$@"
    printf "\n\n"
  } > "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

PIDS=()
NAMES=()

run_bg() {
  local name="$1"
  local marker="$2"
  shift 2
  local log_path="$LOG_DIR/${name}.log"
  if [[ -e "$marker" ]]; then
    echo "[skip] $name -> $marker"
    return 0
  fi
  echo "[bg] $name -> $log_path"
  {
    echo "step=$name"
    printf "command="
    quote_cmd "$@"
    printf "\n\n"
  } > "$log_path"
  "$@" >> "$log_path" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

wait_bg() {
  local rc=0
  local i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "[done] ${NAMES[$i]}"
    else
      echo "[failed] ${NAMES[$i]} (pid=${PIDS[$i]})" >&2
      rc=1
    fi
  done
  PIDS=()
  NAMES=()
  return "$rc"
}

log "Phase A0: data exp=$EXP_NAME device=$DEVICE batch=$BATCH_SIZE cache=$CACHE_DATA_DEVICE"
run_step data "$DATA_DIR/metadata.json" \
  python3 scripts/generate_heston_data.py \
    --output "$DATA_DIR" \
    --n-train "$N_TRAIN" \
    --n-val "$N_VAL" \
    --n-test "$N_TEST" \
    --steps "$STEPS" \
    --regimes \
    --seed "$SEED"

log "Phase A1: start data-only jobs while vol teacher trains"
run_bg mc_oracle "$ORACLE" \
  python3 scripts/generate_mc_oracle.py \
    --data-dir "$DATA_DIR" --output "$ORACLE" \
    --n-paths "$MC_PATHS"

run_bg quant_gan "$TRAIN_DIR/quant_gan/quant_gan_${EXP_NAME}/checkpoints/best.pt" \
  python3 scripts/train_quant_gan.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/quant_gan" \
    --run-name "quant_gan_${EXP_NAME}" \
    --seq-len "$STEPS" \
    --epochs "$QGAN_EPOCHS" \
    --batch-size "${QGAN_BATCH_SIZE:-128}" \
    --d-steps-per-g "${QGAN_D_STEPS_PER_G:-5}" \
    --gradient-penalty-weight "${QGAN_GRADIENT_PENALTY_WEIGHT:-10}" \
    --lambert-w-delta "${QGAN_LAMBERT_W_DELTA:-0.1}" \
    --device "$DEVICE"

run_step vol_fm "$VOL_CKPT/best.pt" \
  python3 scripts/train_vol_trans.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/vol_fm" \
    --run-name "vol_${EXP_NAME}" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$FM_EPOCHS" \
    --lr "$LR" \
    --lr-schedule "$LR_SCHEDULE" \
    --lr-min "$LR_MIN" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-blocks "$NUM_BLOCKS" \
    "${CACHE_DATA_ARGS[@]}" \
    --save-every-epochs "$SAVE_EVERY_EPOCHS" \
    --action-dropout-prob "${ACTION_DROPOUT_PROB:-0.1}" \
    --device "$DEVICE"

log "Phase A2: ret teacher while vol-dependent students train"
run_bg mf_vol "$TRAIN_DIR/mf_vol/mf_vol_${EXP_NAME}/checkpoints/best.pt" \
  python3 scripts/distill_mean_flow.py --stage vol \
    --teacher-checkpoint "$VOL_CKPT/best.pt" --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/mf_vol" --run-name "mf_vol_${EXP_NAME}" \
    --epochs "$MF_EPOCHS" --batch-size "$BATCH_SIZE" \
    "${CACHE_DATA_ARGS[@]}" \
    --boundary-prob-start 0.5 --boundary-prob-end 0.1 \
    --identity-residual-eval --device "$DEVICE"

run_bg cd_vol "$TRAIN_DIR/cd_vol/cd_vol_${EXP_NAME}/checkpoints/best.pt" \
  python3 scripts/distill_consistency.py --stage vol \
    --teacher-checkpoint "$VOL_CKPT/best.pt" --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/cd_vol" --run-name "cd_vol_${EXP_NAME}" \
    --epochs "$CD_EPOCHS" --batch-size "$BATCH_SIZE" \
    "${CACHE_DATA_ARGS[@]}" \
    --curriculum-kind ict --n-min 10 --n-max 160 --huber-c 0.03 \
    --device "$DEVICE"

run_bg mf_select_vol "$MF_SELECT_DIR/vol/mf_vol_select/checkpoints/best.pt" \
  python3 scripts/distill_mean_flow.py --stage vol \
    --teacher-checkpoint "$VOL_CKPT/last.pt" --data-dir "$DATA_DIR" \
    --output-dir "$MF_SELECT_DIR/vol" --run-name mf_vol_select \
    --epochs "$MF_REFINE_EPOCHS" --lr "$MF_REFINE_LR" --batch-size "$BATCH_SIZE" \
    "${CACHE_DATA_ARGS[@]}" \
    --boundary-prob-start 0.5 --boundary-prob-end 0.1 \
    --identity-residual-eval --device "$DEVICE"

run_step ret_fm "$RET_CKPT/best.pt" \
  python3 scripts/train_ret_trans.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/ret_fm" \
    --run-name "ret_${EXP_NAME}" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$FM_EPOCHS" \
    --lr "$LR" \
    --lr-schedule "$LR_SCHEDULE" \
    --lr-min "$LR_MIN" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-blocks "$NUM_BLOCKS" \
    "${CACHE_DATA_ARGS[@]}" \
    --save-every-epochs "$SAVE_EVERY_EPOCHS" \
    --action-dropout-prob "${ACTION_DROPOUT_PROB:-0.1}" \
    --vol-sampler-checkpoint "$VOL_CKPT/best.pt" \
    --scheduled-sampling-max-prob "$RET_SCHEDULED_MAX_PROB" \
    --device "$DEVICE"

log "Phase A3: ret-dependent baseline students"
run_bg mf_ret "$TRAIN_DIR/mf_ret/mf_ret_${EXP_NAME}/checkpoints/best.pt" \
  python3 scripts/distill_mean_flow.py --stage ret \
    --teacher-checkpoint "$RET_CKPT/best.pt" --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/mf_ret" --run-name "mf_ret_${EXP_NAME}" \
    --epochs "$MF_EPOCHS" --batch-size "$BATCH_SIZE" \
    "${CACHE_DATA_ARGS[@]}" \
    --boundary-prob-start 0.5 --boundary-prob-end 0.1 \
    --identity-residual-eval --device "$DEVICE"

run_bg cd_ret "$TRAIN_DIR/cd_ret/cd_ret_${EXP_NAME}/checkpoints/best.pt" \
  python3 scripts/distill_consistency.py --stage ret \
    --teacher-checkpoint "$RET_CKPT/best.pt" --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/cd_ret" --run-name "cd_ret_${EXP_NAME}" \
    --epochs "$CD_EPOCHS" --batch-size "$BATCH_SIZE" \
    "${CACHE_DATA_ARGS[@]}" \
    --curriculum-kind ict --n-min 10 --n-max 160 --huber-c 0.03 \
    --device "$DEVICE"

wait_bg

log "Phase B/C: pricing-aware checkpoint selection"
run_step selection "$SELECTION_JSON" \
  python3 scripts/select_checkpoint.py \
    --sweep-stage ret \
    --fixed-vol-checkpoint "$VOL_CKPT/last.pt" \
    --sweep-checkpoints "$RET_CKPT"/epoch_*.pt "$RET_CKPT/last.pt" "$RET_CKPT/best.pt" \
    --data-dir "$DATA_DIR" --mc-oracle "$ORACLE" \
    --rank-by pricing_rmse --n-paths "$SELECT_PATHS" --cfg-w "$CFG_W" \
    --moneynesses $MONEYNESS --maturities $MATURITIES \
    --output "$SELECTION_JSON" --device "$DEVICE"

BEST_RET="$(python3 -c "import json; print(json.load(open('$SELECTION_JSON'))['best']['checkpoint'])")"
log "selected ret teacher: $BEST_RET"

log "Phase D: selected ret Mean Flow"
run_step mf_select_ret "$MF_SELECT_DIR/ret/mf_ret_select/checkpoints/best.pt" \
  python3 scripts/distill_mean_flow.py --stage ret \
    --teacher-checkpoint "$BEST_RET" --data-dir "$DATA_DIR" \
    --output-dir "$MF_SELECT_DIR/ret" --run-name mf_ret_select \
    --epochs "$MF_REFINE_EPOCHS" --lr "$MF_REFINE_LR" --batch-size "$BATCH_SIZE" \
    "${CACHE_DATA_ARGS[@]}" \
    --boundary-prob-start 0.5 --boundary-prob-end 0.1 \
    --identity-residual-eval --device "$DEVICE"

log "Phase E: champion rollouts + full evaluation"
mkdir -p "$EVAL_DIR"
python3 scripts/rollout.py \
  --vol-checkpoint "$VOL_CKPT/last.pt" --ret-checkpoint "$BEST_RET" \
  --data-dir "$DATA_DIR" --output "$EVAL_DIR/rollout_fm.npz" \
  --n-paths "$EVAL_PATHS" --n-steps "$STEPS" --regime-actions --cfg-w "$CFG_W" \
  --device "$DEVICE"

python3 scripts/rollout.py \
  --vol-checkpoint "$MF_SELECT_DIR/vol/mf_vol_select/checkpoints/best.pt" \
  --ret-checkpoint "$MF_SELECT_DIR/ret/mf_ret_select/checkpoints/best.pt" \
  --data-dir "$DATA_DIR" --output "$EVAL_DIR/rollout_mf.npz" \
  --n-paths "$EVAL_PATHS" --n-steps "$STEPS" --regime-actions --cfg-w "$CFG_W" \
  --device "$DEVICE"

MC_ORACLE="$ORACLE" MONEYNESS="$MONEYNESS" MATURITIES="$MATURITIES" \
  scripts/run_full_evaluation.sh "$EVAL_DIR" "$DATA_DIR/test.npz" "$EVAL_DIR/evaluation"

log "DONE. Champion comparison table:"
cat "$EVAL_DIR/evaluation/summary.md"
