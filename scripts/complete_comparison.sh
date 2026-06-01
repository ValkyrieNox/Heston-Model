#!/usr/bin/env bash
# =============================================================================
# complete_comparison.sh — finish the FM / MF / CD / QGAN comparison for an
# ALREADY-TRAINED experiment, WITHOUT retraining anything.
#
# run_p3_tuning.sh only rolls out FM + MF in eval_champion/. This script adds
# the two missing competitors (Consistency students + Quant GAN, both best and
# last) on the SAME data + MC oracle, then prints one unified table so you can
# see whether the FM teacher is really the best model.
#
# Usage (on the GPU box that still has the .pt checkpoints):
#     bash scripts/complete_comparison.sh [EXPERIMENT_DIR]
#     # default EXPERIMENT_DIR = runs/experiments/p3_full_parallel
#
# Overridable: DEVICE, EVAL_PATHS, STEPS, CFG_W, MONEYNESS, MATURITIES.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="${1:-runs/experiments/p3_full_parallel}"
DEVICE="${DEVICE:-cuda}"
EVAL_PATHS="${EVAL_PATHS:-10000}"
STEPS="${STEPS:-252}"
CFG_W="${CFG_W:-0}"
MONEYNESS="${MONEYNESS:-0.85 0.90 0.95 1.00 1.05}"
MATURITIES="${MATURITIES:-0.25 0.5 1.0}"

DATA="$EXP_DIR/data"
TRAIN="$EXP_DIR/training"
E="$EXP_DIR/eval_champion"
EVAL="$E/evaluation"
ORACLE="$DATA/mc_oracle.npz"
mkdir -p "$EVAL"

read -r -a MN <<< "$MONEYNESS"
read -r -a MT <<< "$MATURITIES"

log() { printf '\n\033[1m[cmp] %s\033[0m\n' "$*"; }

first_match() {  # echo first existing path matching a glob, or empty
  local f
  for f in $1; do [[ -e "$f" ]] && { echo "$f"; return 0; }; done
  echo ""
}

eval_one() {  # fake_npz  out_json
  python3 scripts/evaluate_rollout.py \
    --real "$DATA/test.npz" --fake "$1" --data-dir "$DATA" \
    --mc-oracle "$ORACLE" \
    --moneynesses "${MN[@]}" --maturities "${MT[@]}" \
    --signature-depth 3 --output "$2" >/dev/null
}

# --- Consistency (CD) students --------------------------------------------
CD_VOL="$(first_match "$TRAIN/cd_vol/*/checkpoints/best.pt")"
CD_RET="$(first_match "$TRAIN/cd_ret/*/checkpoints/best.pt")"
if [[ -n "$CD_VOL" && -n "$CD_RET" ]]; then
  log "rollout + eval: Consistency (CD) student"
  python3 scripts/rollout.py \
    --vol-checkpoint "$CD_VOL" --ret-checkpoint "$CD_RET" \
    --data-dir "$DATA" --output "$E/rollout_cd.npz" \
    --n-paths "$EVAL_PATHS" --n-steps "$STEPS" --regime-actions --cfg-w "$CFG_W" \
    --device "$DEVICE"
  eval_one "$E/rollout_cd.npz" "$EVAL/eval_cd.json"
else
  echo "[skip] no CD checkpoints under $TRAIN/cd_{vol,ret}"
fi

# --- Quant GAN baseline (best + last) -------------------------------------
QG_DIR="$(first_match "$TRAIN/quant_gan/*/checkpoints")"
if [[ -n "$QG_DIR" ]]; then
  log "sample + eval: Quant GAN (best, calibrated)"
  python3 scripts/sample_quant_gan.py \
    --checkpoint "$QG_DIR/best.pt" --output "$E/quant_gan_paths.npz" \
    --n-paths "$EVAL_PATHS" --seed 0
  eval_one "$E/quant_gan_paths.npz" "$EVAL/eval_quant_gan.json"

  if [[ -e "$QG_DIR/last.pt" ]]; then
    log "sample + eval: Quant GAN (last, calibrated)"
    python3 scripts/sample_quant_gan.py \
      --checkpoint "$QG_DIR/last.pt" --output "$E/quant_gan_last_paths.npz" \
      --n-paths "$EVAL_PATHS" --seed 0
    eval_one "$E/quant_gan_last_paths.npz" "$EVAL/eval_quant_gan_last.json"
  fi
else
  echo "[skip] no Quant GAN checkpoints under $TRAIN/quant_gan"
fi

# --- unified table ---------------------------------------------------------
log "unified comparison table"
python3 - "$EVAL" <<'PY'
import json, math, sys
from pathlib import Path

evdir = Path(sys.argv[1])
# label -> eval json filename
rows_spec = [
    ("FM teacher (multi-step)", "eval_fm.json"),
    ("Mean Flow (1-NFE)",       "eval_mf.json"),
    ("Consistency (1-NFE)",     "eval_cd.json"),
    ("Quant GAN (best)",        "eval_quant_gan.json"),
    ("Quant GAN (last)",        "eval_quant_gan_last.json"),
]

def f(x, d=4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(x) else f"{x:.{d}f}"

def pricing(r):
    return r.get("pricing_fake_vs_mc_oracle") or r.get("pricing_fake_vs_carr_madan") or {}

table = []
for label, fn in rows_spec:
    p = evdir / fn
    if not p.exists():
        continue
    r = json.loads(p.read_text())
    d = r.get("distances", {})
    pr = pricing(r)
    sig = d.get("signature_wasserstein")
    sigm = sig.get("mean") if isinstance(sig, dict) else d.get("signature_wasserstein_mean")
    sf = r.get("stylized_facts_comparison", {})
    table.append([
        label,
        f(d.get("marginal_wasserstein_mean")),
        f(d.get("total_return_wasserstein")),
        f(sigm),
        f(pr.get("rmse_overall")),
        f(pr.get("mape_overall")),
        f(sf.get("kurtosis_fake"), 2),
    ])

hdr = ["Model", "Marg-W1", "Total-W1", "Sig-W1", "Price-RMSE", "Price-MAPE", "kurt"]
widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(hdr)]
def line(cells):
    return "  ".join(c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
                      for i, c in enumerate(cells))
print(line(hdr))
print("  ".join("-" * w for w in widths))
for row in table:
    print(line(row))

# markdown copy
md = evdir / "comparison_full.md"
mlines = ["# Full model comparison (same data + MC oracle)", "",
          "| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|"]
for row in table:
    mlines.append("| " + " | ".join(row) + " |")
md.write_text("\n".join(mlines) + "\n")
print(f"\n[cmp] markdown table -> {md}")
PY