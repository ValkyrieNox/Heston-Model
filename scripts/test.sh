#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  scripts/test.sh [options]

Defaults evaluate the P2-medium experiment produced by scripts/train.sh.
Results are organized as:

  runs/experiments/<experiment-name>/
    metadata/      invocation, config, per-step commands, logs, timings
    oracle/        MC oracle paths
    rollouts/      FM / MF / CD / Quant GAN generated paths
    evaluation/    JSON reports and summary markdown tables

The script is resumable: existing oracle, rollout, and evaluation files are
skipped and recorded in metadata/test_steps.tsv.

Options:
  --experiment-root DIR       default: runs/experiments
  --experiment-name NAME      default: p2_medium
  --run-dir DIR               default: EXPERIMENT_ROOT/EXPERIMENT_NAME
  --data-dir DIR              default: RUN_DIR/data
  --tag NAME                  default: EXPERIMENT_NAME
  --device DEVICE             default: auto
  --qgan-sample-device DEVICE default: cpu
  --n-rollout INT             default: 10000
  --n-oracle INT              default: 100000
  --steps INT                 default: 252
  --fm-n-steps INT            default: 20
  --cfg-weights "LIST"        default: "0 0.5 1 2"
  --best-mf-cfg FLOAT         default: 0.5
  --cd-cfg-w FLOAT            default: 0
  --moneyness "LIST"          default: "0.85 0.90 0.95 1.00 1.05"
  --maturities "LIST"         default: "0.25 0.5 1.0"
  --signature-depth INT       default: 3
  --action-seed INT           default: 20260520
  --noise-seed INT            default: 20260520
  --qgan-seed INT             default: 20260520
  --oracle-seed INT           default: 20260521
  -h, --help

Examples:
  scripts/test.sh
  scripts/test.sh --experiment-name p2_full --n-rollout 50000 --n-oracle 200000
  scripts/test.sh --experiment-name p2_medium --cfg-weights "0 0.25 0.5 0.75 1"
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
QGAN_SAMPLE_DEVICE="cpu"
N_ROLLOUT="10000"
N_ORACLE="100000"
STEPS="252"
FM_N_STEPS="20"
CFG_WEIGHTS="0 0.5 1 2"
BEST_MF_CFG="0.5"
CD_CFG_W="0"
MONEYNESS="0.85 0.90 0.95 1.00 1.05"
MATURITIES="0.25 0.5 1.0"
SIGNATURE_DEPTH="3"
ACTION_SEED="20260520"
NOISE_SEED="20260520"
QGAN_SEED="20260520"
ORACLE_SEED="20260521"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-root) EXPERIMENT_ROOT="$2"; shift 2 ;;
    --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --qgan-sample-device) QGAN_SAMPLE_DEVICE="$2"; shift 2 ;;
    --n-rollout) N_ROLLOUT="$2"; shift 2 ;;
    --n-oracle) N_ORACLE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --fm-n-steps) FM_N_STEPS="$2"; shift 2 ;;
    --cfg-weights) CFG_WEIGHTS="$2"; shift 2 ;;
    --best-mf-cfg) BEST_MF_CFG="$2"; shift 2 ;;
    --cd-cfg-w) CD_CFG_W="$2"; shift 2 ;;
    --moneyness) MONEYNESS="$2"; shift 2 ;;
    --maturities) MATURITIES="$2"; shift 2 ;;
    --signature-depth) SIGNATURE_DEPTH="$2"; shift 2 ;;
    --action-seed) ACTION_SEED="$2"; shift 2 ;;
    --noise-seed) NOISE_SEED="$2"; shift 2 ;;
    --qgan-seed) QGAN_SEED="$2"; shift 2 ;;
    --oracle-seed) ORACLE_SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

RUN_DIR="${RUN_DIR:-$EXPERIMENT_ROOT/$EXPERIMENT_NAME}"
DATA_DIR="${DATA_DIR:-$RUN_DIR/data}"
TAG="${TAG:-$EXPERIMENT_NAME}"
TRAIN_DIR="$RUN_DIR/training"
ROLLOUT_DIR="$RUN_DIR/rollouts"
ORACLE_DIR="$RUN_DIR/oracle"
EVAL_DIR="$RUN_DIR/evaluation"
META_DIR="$RUN_DIR/metadata"
LOG_DIR="$META_DIR/logs/test"

VOL_FM="$TRAIN_DIR/vol_fm/vol_${TAG}/checkpoints/best.pt"
RET_FM="$TRAIN_DIR/ret_fm/ret_${TAG}/checkpoints/best.pt"
MF_VOL="$TRAIN_DIR/mf_vol/mf_vol_${TAG}/checkpoints/best.pt"
MF_RET="$TRAIN_DIR/mf_ret/mf_ret_${TAG}/checkpoints/best.pt"
CD_VOL="$TRAIN_DIR/cd_vol/cd_vol_${TAG}/checkpoints/best.pt"
CD_RET="$TRAIN_DIR/cd_ret/cd_ret_${TAG}/checkpoints/best.pt"
QGAN="$TRAIN_DIR/quant_gan/quant_gan_${TAG}/checkpoints/best.pt"
ORACLE="$ORACLE_DIR/mc_oracle.npz"

mkdir -p "$RUN_DIR" "$ROLLOUT_DIR" "$ORACLE_DIR" "$EVAL_DIR" "$META_DIR" "$LOG_DIR"
TEST_STARTED_AT="$(timestamp_utc)"

{
  echo "phase=test"
  echo "started_at_utc=$TEST_STARTED_AT"
  echo "root_dir=$ROOT_DIR"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  printf "command="
  quote_cmd "$SCRIPT_DIR/test.sh" "${ORIGINAL_ARGS[@]}"
  printf "\n"
} > "$META_DIR/test_invocation.txt"

python3 - "$META_DIR/test_config.json" <<PY
import json
import sys

config = {
    "phase": "test",
    "started_at_utc": "$TEST_STARTED_AT",
    "experiment_root": "$EXPERIMENT_ROOT",
    "experiment_name": "$EXPERIMENT_NAME",
    "run_dir": "$RUN_DIR",
    "data_dir": "$DATA_DIR",
    "training_dir": "$TRAIN_DIR",
    "rollout_dir": "$ROLLOUT_DIR",
    "oracle_dir": "$ORACLE_DIR",
    "evaluation_dir": "$EVAL_DIR",
    "tag": "$TAG",
    "device": "$DEVICE",
    "qgan_sample_device": "$QGAN_SAMPLE_DEVICE",
    "n_rollout": int("$N_ROLLOUT"),
    "n_oracle": int("$N_ORACLE"),
    "steps": int("$STEPS"),
    "fm_n_steps": int("$FM_N_STEPS"),
    "cfg_weights": "$CFG_WEIGHTS".split(),
    "best_mf_cfg": "$BEST_MF_CFG",
    "cd_cfg_w": "$CD_CFG_W",
    "moneyness": "$MONEYNESS".split(),
    "maturities": "$MATURITIES".split(),
    "signature_depth": int("$SIGNATURE_DEPTH"),
    "action_seed": int("$ACTION_SEED"),
    "noise_seed": int("$NOISE_SEED"),
    "qgan_seed": int("$QGAN_SEED"),
    "oracle_seed": int("$ORACLE_SEED"),
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\\n")
PY

printf "step\tstatus\tstarted_at_utc\tfinished_at_utc\tduration_s\tmarker\tcommand\n" \
  > "$META_DIR/test_steps.tsv"

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
  } >> "$META_DIR/test_steps.tsv"
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

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required file: $path" >&2
    echo "Run scripts/train.sh first, or pass matching --run-dir/--tag." >&2
    exit 1
  fi
}

copy_if_missing() {
  local step="$1"
  local src="$2"
  local dst="$3"
  if [[ -e "$dst" ]]; then
    local now
    now="$(timestamp_utc)"
    record_step "$step" "skip" "$now" "$now" "0" "$dst" cp "$src" "$dst"
    echo "[skip] $step -> $dst"
    return 0
  fi
  require_file "$src"
  run_if_missing "$step" "$dst" python3 - "$src" "$dst" <<'PY'
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
src_json = src.with_suffix(".json")
if src_json.exists():
    shutil.copy2(src_json, dst.with_suffix(".json"))
PY
}

evaluate_if_missing() {
  local step="$1"
  local fake="$2"
  local output="$3"
  shift 3
  run_if_missing "$step" "$output" \
    python3 scripts/evaluate_rollout.py \
      --real "$DATA_DIR/test.npz" \
      --fake "$fake" \
      --data-dir "$DATA_DIR" \
      --mc-oracle "$ORACLE" \
      --output "$output" \
      --signature-depth "$SIGNATURE_DEPTH" \
      --moneynesses "${MONEYNESS_ARGS[@]}" \
      --maturities "${MATURITY_ARGS[@]}" \
      "$@"
}

require_file "$DATA_DIR/test.npz"
require_file "$VOL_FM"
require_file "$RET_FM"
require_file "$MF_VOL"
require_file "$MF_RET"
require_file "$CD_VOL"
require_file "$CD_RET"
require_file "$QGAN"

read -r -a MONEYNESS_ARGS <<< "$MONEYNESS"
read -r -a MATURITY_ARGS <<< "$MATURITIES"

run_if_missing mc_oracle "$ORACLE" \
  python3 scripts/generate_mc_oracle.py \
    --data-dir "$DATA_DIR" \
    --output "$ORACLE" \
    --n-paths "$N_ORACLE" \
    --seed "$ORACLE_SEED"

run_if_missing rollout_fm "$ROLLOUT_DIR/rollout_fm.npz" \
  python3 scripts/rollout.py \
    --vol-checkpoint "$VOL_FM" \
    --ret-checkpoint "$RET_FM" \
    --data-dir "$DATA_DIR" \
    --output "$ROLLOUT_DIR/rollout_fm.npz" \
    --n-paths "$N_ROLLOUT" \
    --n-steps "$STEPS" \
    --regime-actions \
    --action-seed "$ACTION_SEED" \
    --noise-seed "$NOISE_SEED" \
    --fm-n-steps "$FM_N_STEPS" \
    --device "$DEVICE"

for cfg_w in $CFG_WEIGHTS; do
  run_if_missing "rollout_mf_cfg${cfg_w}" "$ROLLOUT_DIR/rollout_mf_cfg${cfg_w}.npz" \
    python3 scripts/rollout.py \
      --vol-checkpoint "$MF_VOL" \
      --ret-checkpoint "$MF_RET" \
      --data-dir "$DATA_DIR" \
      --output "$ROLLOUT_DIR/rollout_mf_cfg${cfg_w}.npz" \
      --n-paths "$N_ROLLOUT" \
      --n-steps "$STEPS" \
      --regime-actions \
      --action-seed "$ACTION_SEED" \
      --noise-seed "$NOISE_SEED" \
      --cfg-w "$cfg_w" \
      --device "$DEVICE"
done
copy_if_missing rollout_mf_best "$ROLLOUT_DIR/rollout_mf_cfg${BEST_MF_CFG}.npz" "$ROLLOUT_DIR/rollout_mf.npz"

run_if_missing rollout_cd "$ROLLOUT_DIR/rollout_cd.npz" \
  python3 scripts/rollout.py \
    --vol-checkpoint "$CD_VOL" \
    --ret-checkpoint "$CD_RET" \
    --data-dir "$DATA_DIR" \
    --output "$ROLLOUT_DIR/rollout_cd.npz" \
    --n-paths "$N_ROLLOUT" \
    --n-steps "$STEPS" \
    --regime-actions \
    --action-seed "$ACTION_SEED" \
    --noise-seed "$NOISE_SEED" \
    --cfg-w "$CD_CFG_W" \
    --device "$DEVICE"

run_if_missing sample_quant_gan "$ROLLOUT_DIR/quant_gan_paths.npz" \
  python3 scripts/sample_quant_gan.py \
    --checkpoint "$QGAN" \
    --output "$ROLLOUT_DIR/quant_gan_paths.npz" \
    --n-paths "$N_ROLLOUT" \
    --device "$QGAN_SAMPLE_DEVICE" \
    --seed "$QGAN_SEED"

evaluate_if_missing eval_fm "$ROLLOUT_DIR/rollout_fm.npz" "$EVAL_DIR/eval_fm.json"
evaluate_if_missing eval_mf "$ROLLOUT_DIR/rollout_mf.npz" "$EVAL_DIR/eval_mf.json"
for cfg_w in $CFG_WEIGHTS; do
  evaluate_if_missing "eval_mf_cfg${cfg_w}" \
    "$ROLLOUT_DIR/rollout_mf_cfg${cfg_w}.npz" "$EVAL_DIR/eval_mf_cfg${cfg_w}.json"
done
evaluate_if_missing eval_cd "$ROLLOUT_DIR/rollout_cd.npz" "$EVAL_DIR/eval_cd.json"
evaluate_if_missing eval_quant_gan "$ROLLOUT_DIR/quant_gan_paths.npz" "$EVAL_DIR/eval_quant_gan.json"

run_if_missing cfg_sweep_summary "$EVAL_DIR/cfg_sweep_summary.txt" \
  python3 scripts/summarize_cfg_sweep.py \
    --run-dir "$EVAL_DIR" \
    --output "$EVAL_DIR/cfg_sweep_summary.txt" \
    --cfg-weights "$CFG_WEIGHTS"

run_if_missing model_comparison_summary "$EVAL_DIR/model_comparison_summary.md" \
  python3 scripts/summarize_model_comparison.py \
    --eval-dir "$EVAL_DIR" \
    --output "$EVAL_DIR/model_comparison_summary.md"

TEST_FINISHED_AT="$(timestamp_utc)"
{
  echo "phase=test"
  echo "started_at_utc=$TEST_STARTED_AT"
  echo "finished_at_utc=$TEST_FINISHED_AT"
  echo "run_dir=$RUN_DIR"
  echo "data_dir=$DATA_DIR"
  echo "training_dir=$TRAIN_DIR"
  echo "rollout_dir=$ROLLOUT_DIR"
  echo "oracle_dir=$ORACLE_DIR"
  echo "evaluation_dir=$EVAL_DIR"
  echo "metadata_dir=$META_DIR"
} > "$META_DIR/test_finished.txt"

echo
echo "[done] evaluation artifacts are under $EVAL_DIR"
echo "[done] test metadata is under $META_DIR"
