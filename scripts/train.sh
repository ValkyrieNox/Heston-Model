#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/train.sh [options]

Defaults run the P2-medium experiment. Results are organized as:

  runs/experiments/<experiment-name>/
    metadata/      invocation, config, per-step commands, logs, timings
    data/          generated Heston data
    training/      FM / MF / CD / Quant GAN training runs

The script is resumable: if a target data file or checkpoint exists, that step
is skipped and the skip is recorded in metadata/train_steps.tsv.

Options:
  --experiment-root DIR            default: runs/experiments
  --experiment-name NAME           default: p2_medium
  --run-dir DIR                    default: EXPERIMENT_ROOT/EXPERIMENT_NAME
  --data-dir DIR                   default: RUN_DIR/data
  --tag NAME                       default: EXPERIMENT_NAME
  --device DEVICE                  default: auto
  --seed INT                       default: 20260520
  --n-train INT                    default: 5000
  --n-val INT                      default: 1000
  --n-test INT                     default: 1000
  --steps INT                      default: 252
  --batch-size INT                 default: 512
  --fm-epochs INT                  default: 10
  --mf-epochs INT                  default: 10
  --cd-epochs INT                  default: 10
  --qgan-epochs INT                default: 15
  --lr FLOAT                       default: 3e-4
  --lr-schedule MODE               constant|cosine, default: constant
  --lr-min FLOAT                   eta_min for cosine, default: 0.0
  --hidden-dim INT                 FM teacher width, default: 128
  --num-blocks INT                 FM teacher depth, default: 4
  --save-every-epochs INT          dump epoch_XXX.pt every N (0=off), default: 0
  --action-dropout-prob FLOAT      default: 0.1
  --ret-scheduled-max-prob FLOAT   default: 0.5
  --ret-scheduled-start-epoch INT  default: 1
  --qgan-batch-size INT            default: 128
  --qgan-d-steps-per-g INT         default: 5
  --qgan-gradient-penalty FLOAT    default: 10
  --qgan-lambert-w-delta FLOAT     default: 0.1
  -h, --help

Examples:
  scripts/train.sh
  scripts/train.sh --experiment-name p2_full --n-train 50000 --n-val 5000 --n-test 10000
  scripts/train.sh --experiment-name p2_ss02 --ret-scheduled-max-prob 0.2
  scripts/train.sh --run-dir /tmp/custom_experiment --tag custom_experiment
USAGE
}

quote_cmd() {
  printf "%q " "$@"
}

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

ORIGINAL_ARGS=("$@")

EXPERIMENT_ROOT="runs/experiments"
EXPERIMENT_NAME="p2_medium"
RUN_DIR=""
DATA_DIR=""
TAG=""
DEVICE="auto"
SEED="20260520"
N_TRAIN="5000"
N_VAL="1000"
N_TEST="1000"
STEPS="252"
BATCH_SIZE="512"
FM_EPOCHS="10"
MF_EPOCHS="10"
CD_EPOCHS="10"
QGAN_EPOCHS="15"
LR="3e-4"
LR_SCHEDULE="constant"
LR_MIN="0.0"
HIDDEN_DIM="128"
NUM_BLOCKS="4"
SAVE_EVERY_EPOCHS="0"
ACTION_DROPOUT_PROB="0.1"
RET_SCHEDULED_MAX_PROB="0.5"
RET_SCHEDULED_START_EPOCH="1"
QGAN_BATCH_SIZE="128"
QGAN_D_STEPS_PER_G="5"
QGAN_GRADIENT_PENALTY_WEIGHT="10"
QGAN_LAMBERT_W_DELTA="0.1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --n-train) N_TRAIN="$2"; shift 2 ;;
    --n-val) N_VAL="$2"; shift 2 ;;
    --n-test) N_TEST="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --fm-epochs) FM_EPOCHS="$2"; shift 2 ;;
    --mf-epochs) MF_EPOCHS="$2"; shift 2 ;;
    --cd-epochs) CD_EPOCHS="$2"; shift 2 ;;
    --qgan-epochs) QGAN_EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --lr-schedule) LR_SCHEDULE="$2"; shift 2 ;;
    --lr-min) LR_MIN="$2"; shift 2 ;;
    --hidden-dim) HIDDEN_DIM="$2"; shift 2 ;;
    --num-blocks) NUM_BLOCKS="$2"; shift 2 ;;
    --save-every-epochs) SAVE_EVERY_EPOCHS="$2"; shift 2 ;;
    --action-dropout-prob) ACTION_DROPOUT_PROB="$2"; shift 2 ;;
    --ret-scheduled-max-prob) RET_SCHEDULED_MAX_PROB="$2"; shift 2 ;;
    --ret-scheduled-start-epoch) RET_SCHEDULED_START_EPOCH="$2"; shift 2 ;;
    --qgan-batch-size) QGAN_BATCH_SIZE="$2"; shift 2 ;;
    --qgan-d-steps-per-g) QGAN_D_STEPS_PER_G="$2"; shift 2 ;;
    --qgan-gradient-penalty) QGAN_GRADIENT_PENALTY_WEIGHT="$2"; shift 2 ;;
    --qgan-lambert-w-delta) QGAN_LAMBERT_W_DELTA="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

RUN_DIR="${RUN_DIR:-$EXPERIMENT_ROOT/$EXPERIMENT_NAME}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/data}"
TAG="${TAG:-$EXPERIMENT_NAME}"
TRAIN_DIR="$RUN_DIR/training"
META_DIR="$RUN_DIR/metadata"
LOG_DIR="$META_DIR/logs/train"

VOL_RUN="vol_${TAG}"
RET_RUN="ret_${TAG}"
MF_VOL_RUN="mf_vol_${TAG}"
MF_RET_RUN="mf_ret_${TAG}"
CD_VOL_RUN="cd_vol_${TAG}"
CD_RET_RUN="cd_ret_${TAG}"
QGAN_RUN="quant_gan_${TAG}"

mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$META_DIR" "$LOG_DIR"
TRAIN_STARTED_AT="$(timestamp_utc)"

{
  echo "phase=train"
  echo "started_at_utc=$TRAIN_STARTED_AT"
  echo "root_dir=$ROOT_DIR"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  printf "command="
  quote_cmd "$SCRIPT_DIR/train.sh" "${ORIGINAL_ARGS[@]}"
  printf "\n"
} > "$META_DIR/train_invocation.txt"

python3 - "$META_DIR/train_config.json" <<PY
import json
import sys

config = {
    "phase": "train",
    "started_at_utc": "$TRAIN_STARTED_AT",
    "experiment_root": "$EXPERIMENT_ROOT",
    "experiment_name": "$EXPERIMENT_NAME",
    "run_dir": "$RUN_DIR",
    "data_dir": "$DATA_DIR",
    "training_dir": "$TRAIN_DIR",
    "tag": "$TAG",
    "device": "$DEVICE",
    "seed": int("$SEED"),
    "n_train": int("$N_TRAIN"),
    "n_val": int("$N_VAL"),
    "n_test": int("$N_TEST"),
    "steps": int("$STEPS"),
    "batch_size": int("$BATCH_SIZE"),
    "fm_epochs": int("$FM_EPOCHS"),
    "mf_epochs": int("$MF_EPOCHS"),
    "cd_epochs": int("$CD_EPOCHS"),
    "qgan_epochs": int("$QGAN_EPOCHS"),
    "lr": "$LR",
    "lr_schedule": "$LR_SCHEDULE",
    "lr_min": "$LR_MIN",
    "hidden_dim": int("$HIDDEN_DIM"),
    "num_blocks": int("$NUM_BLOCKS"),
    "save_every_epochs": int("$SAVE_EVERY_EPOCHS"),
    "action_dropout_prob": "$ACTION_DROPOUT_PROB",
    "ret_scheduled_max_prob": "$RET_SCHEDULED_MAX_PROB",
    "ret_scheduled_start_epoch": int("$RET_SCHEDULED_START_EPOCH"),
    "qgan_batch_size": int("$QGAN_BATCH_SIZE"),
    "qgan_d_steps_per_g": int("$QGAN_D_STEPS_PER_G"),
    "qgan_gradient_penalty": "$QGAN_GRADIENT_PENALTY_WEIGHT",
    "qgan_lambert_w_delta": "$QGAN_LAMBERT_W_DELTA",
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\\n")
PY

printf "step\tstatus\tstarted_at_utc\tfinished_at_utc\tduration_s\tmarker\tcommand\n" \
  > "$META_DIR/train_steps.tsv"

record_step() {
  local step="$1"
  local status="$2"
  local started_at="$3"
  local finished_at="$4"
  local duration="$5"
  local marker="$6"
  shift 6
  {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t" \
      "$step" "$status" "$started_at" "$finished_at" "$duration" "$marker"
    quote_cmd "$@"
    printf "\n"
  } >> "$META_DIR/train_steps.tsv"
}

run_if_missing() {
  local step="$1"
  local marker="$2"
  shift 2
  local started_at
  local finished_at
  local start_s
  local end_s
  local duration
  local log_path="$LOG_DIR/${step}.log"
  if [[ -e "$marker" ]]; then
    started_at="$(timestamp_utc)"
    record_step "$step" "skip" "$started_at" "$started_at" "0" "$marker" "$@"
    echo "[skip] $step -> $marker"
    return 0
  fi

  started_at="$(timestamp_utc)"
  start_s="$(date +%s)"
  echo "[run] $step"
  {
    echo "step=$step"
    echo "started_at_utc=$started_at"
    echo "marker=$marker"
    printf "command="
    quote_cmd "$@"
    printf "\n\n"
  } > "$log_path"

  set +e
  "$@" 2>&1 | tee -a "$log_path"
  local rc=${PIPESTATUS[0]}
  set -e

  finished_at="$(timestamp_utc)"
  end_s="$(date +%s)"
  duration="$((end_s - start_s))"
  if [[ "$rc" -eq 0 ]]; then
    record_step "$step" "done" "$started_at" "$finished_at" "$duration" "$marker" "$@"
    {
      echo
      echo "finished_at_utc=$finished_at"
      echo "duration_s=$duration"
      echo "status=done"
    } >> "$log_path"
  else
    record_step "$step" "failed:$rc" "$started_at" "$finished_at" "$duration" "$marker" "$@"
    {
      echo
      echo "finished_at_utc=$finished_at"
      echo "duration_s=$duration"
      echo "status=failed"
      echo "exit_code=$rc"
    } >> "$log_path"
    exit "$rc"
  fi
}

run_if_missing data "$DATA_DIR/metadata.json" \
  python3 scripts/generate_heston_data.py \
    --output "$DATA_DIR" \
    --n-train "$N_TRAIN" \
    --n-val "$N_VAL" \
    --n-test "$N_TEST" \
    --steps "$STEPS" \
    --regimes \
    --seed "$SEED"

run_if_missing vol_fm "$TRAIN_DIR/vol_fm/$VOL_RUN/checkpoints/best.pt" \
  python3 scripts/train_vol_trans.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/vol_fm" \
    --run-name "$VOL_RUN" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$FM_EPOCHS" \
    --lr "$LR" \
    --lr-schedule "$LR_SCHEDULE" \
    --lr-min "$LR_MIN" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-blocks "$NUM_BLOCKS" \
    --save-every-epochs "$SAVE_EVERY_EPOCHS" \
    --action-dropout-prob "$ACTION_DROPOUT_PROB" \
    --device "$DEVICE"

run_if_missing ret_fm "$TRAIN_DIR/ret_fm/$RET_RUN/checkpoints/best.pt" \
  python3 scripts/train_ret_trans.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/ret_fm" \
    --run-name "$RET_RUN" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$FM_EPOCHS" \
    --lr "$LR" \
    --lr-schedule "$LR_SCHEDULE" \
    --lr-min "$LR_MIN" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-blocks "$NUM_BLOCKS" \
    --save-every-epochs "$SAVE_EVERY_EPOCHS" \
    --action-dropout-prob "$ACTION_DROPOUT_PROB" \
    --vol-sampler-checkpoint "$TRAIN_DIR/vol_fm/$VOL_RUN/checkpoints/best.pt" \
    --scheduled-sampling-max-prob "$RET_SCHEDULED_MAX_PROB" \
    --scheduled-sampling-start-epoch "$RET_SCHEDULED_START_EPOCH" \
    --device "$DEVICE"

run_if_missing mf_vol "$TRAIN_DIR/mf_vol/$MF_VOL_RUN/checkpoints/best.pt" \
  python3 scripts/distill_mean_flow.py \
    --stage vol \
    --teacher-checkpoint "$TRAIN_DIR/vol_fm/$VOL_RUN/checkpoints/best.pt" \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/mf_vol" \
    --run-name "$MF_VOL_RUN" \
    --epochs "$MF_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --boundary-prob-start 0.5 \
    --boundary-prob-end 0.1 \
    --identity-residual-eval \
    --device "$DEVICE"

run_if_missing mf_ret "$TRAIN_DIR/mf_ret/$MF_RET_RUN/checkpoints/best.pt" \
  python3 scripts/distill_mean_flow.py \
    --stage ret \
    --teacher-checkpoint "$TRAIN_DIR/ret_fm/$RET_RUN/checkpoints/best.pt" \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/mf_ret" \
    --run-name "$MF_RET_RUN" \
    --epochs "$MF_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --boundary-prob-start 0.5 \
    --boundary-prob-end 0.1 \
    --identity-residual-eval \
    --device "$DEVICE"

run_if_missing cd_vol "$TRAIN_DIR/cd_vol/$CD_VOL_RUN/checkpoints/best.pt" \
  python3 scripts/distill_consistency.py \
    --stage vol \
    --teacher-checkpoint "$TRAIN_DIR/vol_fm/$VOL_RUN/checkpoints/best.pt" \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/cd_vol" \
    --run-name "$CD_VOL_RUN" \
    --epochs "$CD_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --curriculum-kind ict \
    --n-min 10 \
    --n-max 160 \
    --huber-c 0.03 \
    --device "$DEVICE"

run_if_missing cd_ret "$TRAIN_DIR/cd_ret/$CD_RET_RUN/checkpoints/best.pt" \
  python3 scripts/distill_consistency.py \
    --stage ret \
    --teacher-checkpoint "$TRAIN_DIR/ret_fm/$RET_RUN/checkpoints/best.pt" \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/cd_ret" \
    --run-name "$CD_RET_RUN" \
    --epochs "$CD_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --curriculum-kind ict \
    --n-min 10 \
    --n-max 160 \
    --huber-c 0.03 \
    --device "$DEVICE"

run_if_missing quant_gan "$TRAIN_DIR/quant_gan/$QGAN_RUN/checkpoints/best.pt" \
  python3 scripts/train_quant_gan.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$TRAIN_DIR/quant_gan" \
    --run-name "$QGAN_RUN" \
    --seq-len "$STEPS" \
    --epochs "$QGAN_EPOCHS" \
    --batch-size "$QGAN_BATCH_SIZE" \
    --d-steps-per-g "$QGAN_D_STEPS_PER_G" \
    --gradient-penalty-weight "$QGAN_GRADIENT_PENALTY_WEIGHT" \
    --lambert-w-delta "$QGAN_LAMBERT_W_DELTA" \
    --device "$DEVICE"

TRAIN_FINISHED_AT="$(timestamp_utc)"
{
  echo "phase=train"
  echo "started_at_utc=$TRAIN_STARTED_AT"
  echo "finished_at_utc=$TRAIN_FINISHED_AT"
  echo "run_dir=$RUN_DIR"
  echo "data_dir=$DATA_DIR"
  echo "training_dir=$TRAIN_DIR"
  echo "metadata_dir=$META_DIR"
} > "$META_DIR/train_finished.txt"

echo
echo "[done] training artifacts are under $TRAIN_DIR"
echo "[done] training metadata is under $META_DIR"
