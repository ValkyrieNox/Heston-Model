#!/usr/bin/env bash
# =============================================================================
# lwfm_vol_sweep.sh — Lambert-W Flow Matching: sweep the heavy-tail strength of
# the VARIANCE kernel to fix the FM teacher's thin-tail / under-dispersion gap.
#
# Diagnosis (see chat / 13_P2MediumResults): Heston return tails come almost
# entirely from volatility mixing (r|v is conditionally Gaussian, kurt~3). The
# FM teacher under-disperses the variance, so return kurtosis is 3.9 vs 4.6.
# This sweeps a Lambert-W Gaussianization delta on the vol-kernel TARGET
# (QGAN's heavy-tail trick, moved to where it belongs in a two-stage model),
# pairs each vol kernel with the EXISTING best ret teacher, rolls out, and
# ranks by how well the return kurtosis matches the real data + pricing.
#
# Trains ONLY new vol kernels (into runs/.../training/vol_lwfm_d<delta>/), reuses
# the existing ret teacher + data + MC oracle. Writes to a NEW eval dir; never
# touches eval_champion/ or eval_calibrated/.
#
# Usage (GPU box):
#   bash scripts/lwfm_vol_sweep.sh [EXPERIMENT_DIR]
#   DELTAS="0.03 0.05 0.08 0.12" bash scripts/lwfm_vol_sweep.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

EXP_DIR="${1:-runs/experiments/p3_full_parallel}"
EXP_NAME="$(basename "$EXP_DIR")"
DEVICE="${DEVICE:-cuda}"
DELTAS="${DELTAS:-0.03 0.05 0.08 0.12}"

# vol-kernel training hyperparams (match your champion teacher run)
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_BLOCKS="${NUM_BLOCKS:-6}"
FM_EPOCHS="${FM_EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LR="${LR:-2e-4}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
LR_MIN="${LR_MIN:-1e-5}"
ACTION_DROPOUT_PROB="${ACTION_DROPOUT_PROB:-0.1}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
CACHE_DATA_DEVICE="${CACHE_DATA_DEVICE:-1}"

EVAL_PATHS="${EVAL_PATHS:-10000}"
STEPS="${STEPS:-252}"
CFG_W="${CFG_W:-0}"
MONEYNESS="${MONEYNESS:-0.85 0.90 0.95 1.00 1.05}"
MATURITIES="${MATURITIES:-0.25 0.5 1.0}"

DATA="$EXP_DIR/data"
TRAIN="$EXP_DIR/training"
OUT="$EXP_DIR/eval_lwfm"
EVAL="$OUT/evaluation"
ORACLE="$DATA/mc_oracle.npz"
mkdir -p "$EVAL"
read -r -a MN <<< "$MONEYNESS"; read -r -a MT <<< "$MATURITIES"
log() { printf '\n\033[1m[lwfm] %s\033[0m\n' "$*"; }
first_match() { local f; for f in $1; do [[ -e "$f" ]] && { echo "$f"; return; }; done; echo ""; }

# existing best ret teacher (pricing-selected; falls back to last/epoch_060)
RET="$(python3 -c "import json;print(json.load(open('$EXP_DIR/selection_ret.json'))['best']['checkpoint'])" 2>/dev/null || true)"
[[ -z "${RET:-}" || ! -e "$RET" ]] && RET="$(first_match "$TRAIN/ret_fm/ret_${EXP_NAME}/checkpoints/last.pt $TRAIN/ret_fm/ret_${EXP_NAME}/checkpoints/best.pt")"
[[ -z "$RET" ]] && { echo "no ret teacher found under $TRAIN/ret_fm" >&2; exit 1; }
log "ret teacher (fixed): $RET"

# real return kurtosis (target to match)
REAL_KURT="$(python3 -c "
import numpy as np
z=np.load('$DATA/test.npz')['log_returns'].ravel()
print(f'{(((z-z.mean())/z.std())**4).mean():.4f}')
")"
log "real return kurtosis target = $REAL_KURT"

for D in $DELTAS; do
  RUN="vol_lwfm_d${D}"
  CKPT="$TRAIN/$RUN/$RUN/checkpoints/best.pt"
  if [[ ! -e "$CKPT" ]]; then
    log "train vol kernel  delta=$D"
    CACHE_ARGS=()
    [[ "$CACHE_DATA_DEVICE" == "1" ]] && CACHE_ARGS=(--cache-data-device)
    python3 scripts/train_vol_trans.py --data-dir "$DATA" \
      --output-dir "$TRAIN/$RUN" --run-name "$RUN" \
      --epochs "$FM_EPOCHS" --batch-size "$BATCH_SIZE" \
      --lr "$LR" --lr-schedule "$LR_SCHEDULE" --lr-min "$LR_MIN" \
      --hidden-dim "$HIDDEN_DIM" --num-blocks "$NUM_BLOCKS" \
      --save-every-epochs "$SAVE_EVERY_EPOCHS" \
      --action-dropout-prob "$ACTION_DROPOUT_PROB" \
      "${CACHE_ARGS[@]}" \
      --lambert-w-delta "$D" --device "$DEVICE"
  fi
  log "rollout + eval  delta=$D"
  python3 scripts/rollout.py --vol-checkpoint "$CKPT" --ret-checkpoint "$RET" \
    --data-dir "$DATA" --output "$OUT/rollout_d${D}.npz" \
    --n-paths "$EVAL_PATHS" --n-steps "$STEPS" --regime-actions --cfg-w "$CFG_W" \
    --device "$DEVICE"
  python3 scripts/evaluate_rollout.py --real "$DATA/test.npz" --fake "$OUT/rollout_d${D}.npz" \
    --data-dir "$DATA" --mc-oracle "$ORACLE" \
    --moneynesses "${MN[@]}" --maturities "${MT[@]}" \
    --signature-depth 3 --output "$EVAL/eval_d${D}.json" >/dev/null
done

log "Lambert-W vol-delta sweep (target kurt=$REAL_KURT)"
python3 - "$EVAL" "$REAL_KURT" $DELTAS <<'PY'
import json, math, sys
from pathlib import Path
evdir, real_kurt = Path(sys.argv[1]), float(sys.argv[2])
deltas = sys.argv[3:]
def f(x,d=4):
    try: x=float(x)
    except (TypeError,ValueError): return "n/a"
    return "n/a" if not math.isfinite(x) else f"{x:.{d}f}"
rows=[]
for D in deltas:
    p=evdir/f"eval_d{D}.json"
    if not p.exists(): continue
    r=json.loads(p.read_text()); dd=r.get("distances",{})
    pr=r.get("pricing_fake_vs_mc_oracle") or r.get("pricing_fake_vs_carr_madan") or {}
    sf=r.get("stylized_facts_comparison",{})
    k=sf.get("kurtosis_fake")
    rows.append({"delta":D,"kurt":k,"kurt_gap":abs(float(k)-real_kurt) if k else 9e9,
                 "total_w1":dd.get("total_return_wasserstein"),
                 "sig_w1":(dd.get("signature_wasserstein") or {}).get("mean") if isinstance(dd.get("signature_wasserstein"),dict) else None,
                 "rmse":pr.get("rmse_overall"),"mape":pr.get("mape_overall")})
rows.sort(key=lambda x:(x["rmse"] is None, x["rmse"] if x["rmse"] is not None else 9e9))
hdr=["delta","kurt(→%.2f)"%real_kurt,"Total-W1","Sig-W1","Price-RMSE","Price-MAPE"]
print("  ".join(h.rjust(11) for h in hdr))
for x in rows:
    print("  ".join(s.rjust(11) for s in [x["delta"],f(x["kurt"],2),f(x["total_w1"]),f(x["sig_w1"]),f(x["rmse"]),f(x["mape"])]))
best=rows[0] if rows else None
md=evdir/"comparison_lwfm.md"
ml=["# Lambert-W variance-kernel delta sweep (real kurt=%.2f)"%real_kurt,"",
    "| delta | kurt | Total-W1 | Sig-W1 | Price-RMSE | Price-MAPE |","|---|---:|---:|---:|---:|---:|"]
ml+=["| %s | %s | %s | %s | %s | %s |"%(x["delta"],f(x["kurt"],2),f(x["total_w1"]),f(x["sig_w1"]),f(x["rmse"]),f(x["mape"])) for x in rows]
md.write_text("\n".join(ml)+"\n")
if best: print(f"\n[lwfm] best by pricing RMSE: delta={best['delta']}  (table -> {md})")
PY
echo
echo "[lwfm] outputs in $OUT  (eval_champion/ and eval_calibrated/ untouched)"
