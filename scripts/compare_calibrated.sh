#!/usr/bin/env bash
# =============================================================================
# compare_calibrated.sh — FAIR comparison with all models moment-calibrated.
#
# The existing eval_champion/ table is UNFAIR: the flow models (FM/MF/CD) are
# raw, but Quant GAN gets a sampling-time moment calibration (pooled return
# mean/std pinned to the data). This script levels the field: it re-rolls the
# flow models WITH the same calibration (--calibrate-moments) and re-samples
# Quant GAN (also calibrated), then builds one "all-calibrated" table.
#
# It writes ONLY to  <EXP_DIR>/eval_calibrated/  and NEVER touches the frozen
# eval_champion/ results.
#
# Usage (on the GPU box with the .pt checkpoints):
#     bash scripts/compare_calibrated.sh [EXPERIMENT_DIR]
#     # default EXPERIMENT_DIR = runs/experiments/p3_full_parallel
#
# Overridable: DEVICE, EVAL_PATHS, STEPS, CFG_W, MONEYNESS, MATURITIES.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="${1:-runs/experiments/p3_full_parallel}"
EXP_NAME="$(basename "$EXP_DIR")"
DEVICE="${DEVICE:-cuda}"
EVAL_PATHS="${EVAL_PATHS:-10000}"
STEPS="${STEPS:-252}"
CFG_W="${CFG_W:-0}"
MONEYNESS="${MONEYNESS:-0.85 0.90 0.95 1.00 1.05}"
MATURITIES="${MATURITIES:-0.25 0.5 1.0}"

DATA="$EXP_DIR/data"
TRAIN="$EXP_DIR/training"
OUT="$EXP_DIR/eval_calibrated"          # <-- separate dir; eval_champion untouched
EVAL="$OUT/evaluation"
ORACLE="$DATA/mc_oracle.npz"
mkdir -p "$EVAL"

read -r -a MN <<< "$MONEYNESS"
read -r -a MT <<< "$MATURITIES"
log() { printf '\n\033[1m[cal] %s\033[0m\n' "$*"; }
first_match() { local f; for f in $1; do [[ -e "$f" ]] && { echo "$f"; return; }; done; echo ""; }

eval_one() {  # fake_npz  out_json
  python3 scripts/evaluate_rollout.py \
    --real "$DATA/test.npz" --fake "$1" --data-dir "$DATA" --mc-oracle "$ORACLE" \
    --moneynesses "${MN[@]}" --maturities "${MT[@]}" \
    --signature-depth 3 --output "$2" >/dev/null
}

roll_cal() {  # vol_ckpt  ret_ckpt  out_npz  (rolls out WITH --calibrate-moments)
  python3 scripts/rollout.py \
    --vol-checkpoint "$1" --ret-checkpoint "$2" \
    --data-dir "$DATA" --output "$3" \
    --n-paths "$EVAL_PATHS" --n-steps "$STEPS" --regime-actions --cfg-w "$CFG_W" \
    --calibrate-moments --device "$DEVICE"
}

# --- resolve checkpoints ---------------------------------------------------
FM_VOL="$(first_match "$TRAIN/vol_fm/vol_${EXP_NAME}/checkpoints/last.pt")"
FM_RET="$(python3 -c "import json;print(json.load(open('$EXP_DIR/selection_ret.json'))['best']['checkpoint'])" 2>/dev/null || true)"
[[ -z "${FM_RET:-}" || ! -e "$FM_RET" ]] && FM_RET="$(first_match "$TRAIN/ret_fm/ret_${EXP_NAME}/checkpoints/epoch_060.pt $TRAIN/ret_fm/ret_${EXP_NAME}/checkpoints/last.pt")"
MF_VOL="$(first_match "$EXP_DIR/mf_select/vol/*/checkpoints/best.pt")"
MF_RET="$(first_match "$EXP_DIR/mf_select/ret/*/checkpoints/best.pt")"
CD_VOL="$(first_match "$TRAIN/cd_vol/*/checkpoints/best.pt")"
CD_RET="$(first_match "$TRAIN/cd_ret/*/checkpoints/best.pt")"
QG_DIR="$(first_match "$TRAIN/quant_gan/*/checkpoints")"

# --- calibrated flow rollouts ---------------------------------------------
if [[ -n "$FM_VOL" && -n "$FM_RET" ]]; then
  log "FM teacher  (calibrated)  vol=$(basename "$FM_VOL") ret=$(basename "$FM_RET")"
  roll_cal "$FM_VOL" "$FM_RET" "$OUT/rollout_fm.npz"; eval_one "$OUT/rollout_fm.npz" "$EVAL/eval_fm.json"
fi
if [[ -n "$MF_VOL" && -n "$MF_RET" ]]; then
  log "Mean Flow   (calibrated)"
  roll_cal "$MF_VOL" "$MF_RET" "$OUT/rollout_mf.npz"; eval_one "$OUT/rollout_mf.npz" "$EVAL/eval_mf.json"
fi
if [[ -n "$CD_VOL" && -n "$CD_RET" ]]; then
  log "Consistency (calibrated)"
  roll_cal "$CD_VOL" "$CD_RET" "$OUT/rollout_cd.npz"; eval_one "$OUT/rollout_cd.npz" "$EVAL/eval_cd.json"
fi

# --- Quant GAN (already moment-calibrated by default) ----------------------
if [[ -n "$QG_DIR" ]]; then
  log "Quant GAN best/last (calibrated)"
  python3 scripts/sample_quant_gan.py --checkpoint "$QG_DIR/best.pt" \
    --output "$OUT/quant_gan_paths.npz" --n-paths "$EVAL_PATHS" --seed 0
  eval_one "$OUT/quant_gan_paths.npz" "$EVAL/eval_quant_gan.json"
  if [[ -e "$QG_DIR/last.pt" ]]; then
    python3 scripts/sample_quant_gan.py --checkpoint "$QG_DIR/last.pt" \
      --output "$OUT/quant_gan_last_paths.npz" --n-paths "$EVAL_PATHS" --seed 0
    eval_one "$OUT/quant_gan_last_paths.npz" "$EVAL/eval_quant_gan_last.json"
  fi
fi

# --- unified all-calibrated table -----------------------------------------
log "all-calibrated comparison table"
python3 - "$EVAL" "ALL MODELS MOMENT-CALIBRATED" "comparison_calibrated.md" <<'PY'
import json, math, sys
from pathlib import Path
evdir, title, mdname = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
spec = [("FM teacher (multi-step)","eval_fm.json"),("Mean Flow (1-NFE)","eval_mf.json"),
        ("Consistency (1-NFE)","eval_cd.json"),("Quant GAN (best)","eval_quant_gan.json"),
        ("Quant GAN (last)","eval_quant_gan_last.json")]
def f(x,d=4):
    try: x=float(x)
    except (TypeError,ValueError): return "n/a"
    return "n/a" if not math.isfinite(x) else f"{x:.{d}f}"
def pr(r): return r.get("pricing_fake_vs_mc_oracle") or r.get("pricing_fake_vs_carr_madan") or {}
rows=[]
for label,fn in spec:
    p=evdir/fn
    if not p.exists(): continue
    r=json.loads(p.read_text()); d=r.get("distances",{}); pp=pr(r)
    sig=d.get("signature_wasserstein"); sigm=sig.get("mean") if isinstance(sig,dict) else d.get("signature_wasserstein_mean")
    sf=r.get("stylized_facts_comparison",{})
    rows.append([label,f(d.get("marginal_wasserstein_mean")),f(d.get("total_return_wasserstein")),
                 f(sigm),f(pp.get("rmse_overall")),f(pp.get("mape_overall")),f(sf.get("kurtosis_fake"),2)])
hdr=["Model","Marg-W1","Total-W1","Sig-W1","Price-RMSE","Price-MAPE","kurt"]
w=[max(len(h),*(len(r[i]) for r in rows)) for i,h in enumerate(hdr)]
def line(c): return "  ".join(x.ljust(w[i]) if i==0 else x.rjust(w[i]) for i,x in enumerate(c))
print(f"\n[{title}]"); print(line(hdr)); print("  ".join("-"*x for x in w))
for r in rows: print(line(r))
md=evdir/mdname
ml=[f"# {title} (same data + MC oracle)","","| "+" | ".join(hdr)+" |","|"+"|".join(["---"]+["---:"]*(len(hdr)-1))+"|"]
ml+=["| "+" | ".join(r)+" |" for r in rows]
md.write_text("\n".join(ml)+"\n"); print(f"\n[cal] -> {md}")
PY
echo
echo "[cal] calibrated outputs in : $OUT  (eval_champion/ untouched)"