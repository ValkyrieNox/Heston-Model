#!/usr/bin/env python3
"""Pricing-aware checkpoint selection for the two-stage FM teacher / students.

Validation loss is a poor proxy for path-level and option-pricing quality in
this pipeline (see idea/2/13_P2MediumResults.md: the ret teacher ``last.pt``
beats ``best.pt`` by ~5x on Total W1). This tool sweeps a set of candidate
checkpoints for ONE stage while holding the other stage fixed, runs a short
autoregressive rollout for each, evaluates it, and ranks the candidates by a
generation/pricing metric instead of validation loss.

Example
-------
    python3 scripts/select_checkpoint.py \
      --sweep-stage ret \
      --fixed-vol-checkpoint runs/sweep/vol/checkpoints/last.pt \
      --sweep-checkpoints runs/sweep/ret/checkpoints/epoch_*.pt \
                          runs/sweep/ret/checkpoints/last.pt \
      --data-dir runs/experiments/p3_full/data \
      --mc-oracle runs/experiments/p3_full/data/mc_oracle.npz \
      --rank-by pricing_rmse \
      --n-paths 3000 \
      --output runs/experiments/p3_full/selection_ret.json

The fixed counterpart for a ``ret`` sweep is the vol checkpoint and vice versa.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"

RANK_KEYS = {
    "pricing_rmse": ("pricing", "rmse_overall"),
    "pricing_mape": ("pricing", "mape_overall"),
    "total_w1": ("distances", "total_return_wasserstein"),
    "marginal_w1": ("distances", "marginal_wasserstein_mean"),
    "sig_w1": ("distances", "signature_wasserstein_mean"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-stage", choices=("vol", "ret"), required=True,
                   help="which stage's checkpoints are being compared")
    p.add_argument("--sweep-checkpoints", type=Path, nargs="+", required=True,
                   help="candidate checkpoints for the swept stage (shell-globbed)")
    p.add_argument("--fixed-vol-checkpoint", type=Path, default=None,
                   help="fixed vol checkpoint (required when --sweep-stage ret)")
    p.add_argument("--fixed-ret-checkpoint", type=Path, default=None,
                   help="fixed ret checkpoint (required when --sweep-stage vol)")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--real", type=Path, default=None,
                   help="real test npz; default: DATA_DIR/test.npz")
    p.add_argument("--mc-oracle", type=Path, default=None,
                   help="MC oracle npz for regime-data pricing")
    p.add_argument("--rank-by", choices=sorted(RANK_KEYS), default="pricing_rmse")
    p.add_argument("--n-paths", type=int, default=3000,
                   help="rollout paths per candidate (small for speed)")
    p.add_argument("--n-steps", type=int, default=252)
    p.add_argument("--cfg-w", type=float, default=0.0)
    p.add_argument("--regime-actions", action="store_true", default=True)
    p.add_argument("--no-regime-actions", dest="regime_actions", action="store_false")
    p.add_argument("--signature-depth", type=int, default=3)
    p.add_argument("--moneynesses", nargs="+", type=float,
                   default=[0.85, 0.9, 0.95, 1.0, 1.05])
    p.add_argument("--maturities", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=Path, default=None,
                   help="write ranked results JSON here")
    p.add_argument("--workdir", type=Path, default=None,
                   help="dir for intermediate rollouts; default: a temp dir")
    return p.parse_args()


def _pricing_block(report: dict) -> dict:
    return (
        report.get("pricing_fake_vs_mc_oracle")
        or report.get("pricing_fake_vs_carr_madan")
        or {}
    )


def _extract(report: dict, rank_by: str) -> float | None:
    group, key = RANK_KEYS[rank_by]
    block = _pricing_block(report) if group == "pricing" else report.get(group, {})
    val = block.get(key)
    # signature_wasserstein may be nested as {"mean": ...}
    if val is None and group == "distances" and key == "signature_wasserstein_mean":
        sig = report.get("distances", {}).get("signature_wasserstein")
        if isinstance(sig, dict):
            val = sig.get("mean")
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _run(cmd: list[str]) -> None:
    print("    $ " + " ".join(str(c) for c in cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    args = parse_args()
    real = args.real or (args.data_dir / "test.npz")

    if args.sweep_stage == "ret" and args.fixed_vol_checkpoint is None:
        sys.exit("--fixed-vol-checkpoint is required when --sweep-stage ret")
    if args.sweep_stage == "vol" and args.fixed_ret_checkpoint is None:
        sys.exit("--fixed-ret-checkpoint is required when --sweep-stage vol")

    workdir_ctx = (
        tempfile.TemporaryDirectory() if args.workdir is None else None
    )
    workdir = Path(args.workdir) if args.workdir else Path(workdir_ctx.name)
    workdir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    candidates = list(dict.fromkeys(args.sweep_checkpoints))  # dedup, keep order
    print(f"[select] sweeping {len(candidates)} {args.sweep_stage} checkpoints, "
          f"ranking by {args.rank_by}", file=sys.stderr)

    for idx, ckpt in enumerate(candidates):
        if not Path(ckpt).exists():
            print(f"  [skip] missing: {ckpt}", file=sys.stderr)
            continue
        tag = f"{idx:02d}_{Path(ckpt).stem}"
        rollout_npz = workdir / f"rollout_{tag}.npz"
        eval_json = workdir / f"eval_{tag}.json"

        if args.sweep_stage == "ret":
            vol_ckpt, ret_ckpt = args.fixed_vol_checkpoint, ckpt
        else:
            vol_ckpt, ret_ckpt = ckpt, args.fixed_ret_checkpoint

        rollout_cmd = [
            sys.executable, str(SCRIPT_DIR / "rollout.py"),
            "--vol-checkpoint", str(vol_ckpt),
            "--ret-checkpoint", str(ret_ckpt),
            "--data-dir", str(args.data_dir),
            "--output", str(rollout_npz),
            "--n-paths", str(args.n_paths),
            "--n-steps", str(args.n_steps),
            "--cfg-w", str(args.cfg_w),
            "--device", args.device,
        ]
        if args.regime_actions:
            rollout_cmd.append("--regime-actions")

        eval_cmd = [
            sys.executable, str(SCRIPT_DIR / "evaluate_rollout.py"),
            "--real", str(real),
            "--fake", str(rollout_npz),
            "--data-dir", str(args.data_dir),
            "--output", str(eval_json),
            "--signature-depth", str(args.signature_depth),
            "--moneynesses", *[str(m) for m in args.moneynesses],
            "--maturities", *[str(m) for m in args.maturities],
        ]
        if args.mc_oracle is not None:
            eval_cmd += ["--mc-oracle", str(args.mc_oracle)]

        print(f"  [{idx + 1}/{len(candidates)}] {ckpt}", file=sys.stderr)
        _run(rollout_cmd)
        _run(eval_cmd)

        report = json.loads(eval_json.read_text(encoding="utf-8"))
        dist = report.get("distances", {})
        pricing = _pricing_block(report)
        row = {
            "checkpoint": str(ckpt),
            "rank_metric": args.rank_by,
            "rank_value": _extract(report, args.rank_by),
            "total_w1": dist.get("total_return_wasserstein"),
            "marginal_w1": dist.get("marginal_wasserstein_mean"),
            "pricing_rmse": pricing.get("rmse_overall"),
            "pricing_mape": pricing.get("mape_overall"),
            "eval_json": str(eval_json),
        }
        results.append(row)
        print(f"      {args.rank_by}={row['rank_value']} "
              f"total_w1={row['total_w1']} pricing_rmse={row['pricing_rmse']}",
              file=sys.stderr)

    ranked = sorted(
        results,
        key=lambda r: (r["rank_value"] is None, r["rank_value"]
                       if r["rank_value"] is not None else float("inf")),
    )

    print("\n=== ranking (best first) ===", file=sys.stderr)
    print(f"{'rank':>4}  {args.rank_by:>14}  {'total_w1':>10}  "
          f"{'pricing_rmse':>12}  checkpoint", file=sys.stderr)
    for i, r in enumerate(ranked, 1):
        rv = "n/a" if r["rank_value"] is None else f"{r['rank_value']:.4f}"
        tw = "n/a" if r["total_w1"] is None else f"{r['total_w1']:.4f}"
        pr = "n/a" if r["pricing_rmse"] is None else f"{r['pricing_rmse']:.4f}"
        print(f"{i:>4}  {rv:>14}  {tw:>10}  {pr:>12}  {r['checkpoint']}",
              file=sys.stderr)

    out = {
        "sweep_stage": args.sweep_stage,
        "rank_by": args.rank_by,
        "n_paths": args.n_paths,
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[select] wrote {args.output}", file=sys.stderr)
    print(json.dumps(out["best"], indent=2))

    if workdir_ctx is not None:
        workdir_ctx.cleanup()


if __name__ == "__main__":
    main()