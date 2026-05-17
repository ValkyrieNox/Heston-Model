# FinFlow — V3 Heston World-Model Pipeline

Flow Matching for the one-step transition kernel of the Heston stochastic
volatility model, plus a Carr-Madan FFT pricer for option-pricing ground truth.
The end goal is a Mean-Flow-distilled 1-NFE autoregressive world model on
synthetic financial paths.

Design docs live under [idea/2/](idea/2/):
- [idea/2/pipelines.md](idea/2/pipelines.md) — V1 / V2 / V3 framing
- [idea/2/V3_References.md](idea/2/V3_References.md) — literature backing every
  piece of V3 (data / models / evaluation)
- [idea/2/v3_implementation.md](idea/2/v3_implementation.md) — end-to-end V3
  implementation plan and code-progress reverse index

## Layout

```
finflow/
  data/
    heston.py            # Andersen QE + QE-M, single + regime-switching simulators
    option_pricing.py    # Heston char fn + Carr-Madan FFT pricer
    dataset.py           # legacy joint + V3 vol/ret transition datasets
  models/
    transition_fm.py     # FiLM-MLP Conditional Flow Matching backbone
  training.py            # legacy joint trainer + V3 vol/ret trainers
scripts/
  generate_heston_data.py  # path simulation CLI (with optional --regimes)
  train_vol_trans.py       # Stage 1a:  p(v_{t+1} | v_t, a_t)
  train_ret_trans.py       # Stage 1b:  p(r_{t+1} | v_{t+1}, v_t, r_t, a_t)
  eval_transition_fm.py    # legacy joint-stage eval
  price_heston_grid.py     # Carr-Madan ground-truth option grid
tests/                     # pytest suite
```

## Quick start

```bash
pip install numpy torch pytest

# 1. Generate the V3 dataset (3-regime Markov mix: normal / high_vol / crash).
python3 scripts/generate_heston_data.py \
  --output data/heston_v3 \
  --n-train 50000 --n-val 5000 --n-test 10000 \
  --steps 252 --regimes --seed 1234

# 2. Train Stage 1a: variance transition kernel.
python3 scripts/train_vol_trans.py \
  --data-dir data/heston_v3 \
  --output-dir runs/vol_trans_fm \
  --batch-size 512 --epochs 20 --lr 3e-4

# 3. Train Stage 1b: return transition kernel (teacher-forced on v_{t+1}).
python3 scripts/train_ret_trans.py \
  --data-dir data/heston_v3 \
  --output-dir runs/ret_trans_fm \
  --batch-size 512 --epochs 20 --lr 3e-4

# 4. (Optional) Carr-Madan ground-truth prices on the 15-point (K, T) grid.
python3 scripts/price_heston_grid.py \
  --output data/heston_v3/option_grid.json
```

Drop `--regimes` to use a single fixed parameter set; `num_actions` is auto-read
from `metadata.json` by the training scripts so no other flag has to change.

For a fast smoke run:

```bash
python3 scripts/generate_heston_data.py --output /tmp/heston_smoke \
  --n-train 64 --n-val 16 --n-test 16 --steps 12 --regimes --seed 0
python3 scripts/train_vol_trans.py --data-dir /tmp/heston_smoke \
  --output-dir /tmp/runs_vol --epochs 2 --batch-size 16 --hidden-dim 32 \
  --time-embedding-dim 16 --num-blocks 2 --device cpu
```

## Outputs

`generate_heston_data.py` writes per split:

- `{split}.npz`: full simulated paths (`s_paths`, `v_paths`, `log_returns`,
  plus `actions` when `--regimes` is set)
- `{split}_transitions.npz`: flattened one-step transitions
  (`v_t`, `r_t`, `v_next`, `r_next`, `log_v_t`, `log_v_next`, optional
  `action`)
- `metadata.json`: Heston params, regime config, normalization stats,
  `num_actions`, and the canonical transition alignment
  `(v_t, r_{t-1}, a_t) -> (v_{t+1}, r_t)`

Training writes `runs/<stage>/<run_name>/`:

- `config.json` — full run config snapshot
- `metrics.jsonl` — per-epoch train / val loss
- `checkpoints/best.pt`, `checkpoints/last.pt`
- `summary.json` — pointer to checkpoints + history

Checkpoints carry `stage` and `num_actions` so the eval entry points can
type-check (`evaluate_two_stage_checkpoint(..., stage="vol"|"ret")`).

## Tests

```bash
python3 -m pytest tests/
```

Covers Heston QE shape / positivity, regime simulation, Carr-Madan accuracy
(low-vol-of-vol → BS limit + monotonicity), transition-dataset layout, and
smoke trainers for all three stages (joint / vol / ret).

## Status

`finflow/` currently implements data generation, the Carr-Madan ground-truth
pricer, and the V3 Stage 1a / Stage 1b Flow Matching teachers. Mean Flow / CD
distillation, stylized-fact evaluation, and the autoregressive rollout are the
next milestones — see [idea/2/v3_implementation.md](idea/2/v3_implementation.md)
for the full punch list and the literature reverse index.
