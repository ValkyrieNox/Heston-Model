#!/usr/bin/env python3
"""Pricing-aware checkpoint/NFE selection for the joint FM teacher.

This sweeps candidate joint checkpoints against a list of FM ODE step counts,
runs short raw rollouts, evaluates each rollout with the standard evaluator,
and ranks by a path/pricing metric instead of transition validation loss.
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
    "abs_total_w1": ("distances", "abs_total_return_wasserstein"),
    "marginal_w1": ("distances", "marginal_wasserstein_mean"),
    "kurtosis_diff": ("stylized_facts_comparison", "kurtosis_abs_diff"),
    "abs_acf_l1": ("stylized_facts_comparison", "absolute_return_acf_l1"),
    "leverage_l1": ("stylized_facts_comparison", "leverage_correlation_l1"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--nfe-steps", type=int, nargs="+", default=[15, 20, 25, 30])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--real", type=Path, default=None,
                   help="real test npz; default: DATA_DIR/test.npz")
    p.add_argument("--mc-oracle", type=Path, default=None)
    p.add_argument("--rank-by", choices=sorted(RANK_KEYS), default="pricing_rmse")
    p.add_argument("--n-paths", type=int, default=3000)
    p.add_argument("--n-steps", type=int, default=252)
    p.add_argument("--cfg-w", type=float, default=0.0)
    p.add_argument("--fm-solver", choices=("euler", "heun"), default="euler")
    p.add_argument("--regime-actions", action="store_true", default=True)
    p.add_argument("--no-regime-actions", dest="regime_actions", action="store_false")
    p.add_argument("--signature-depth", type=int, default=0)
    p.add_argument("--moneynesses", nargs="+", type=float,
                   default=[0.85, 0.9, 0.95, 1.0, 1.05])
    p.add_argument("--maturities", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    p.add_argument("--calibrate-moments", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--workdir", type=Path, default=None,
                   help="dir for intermediate rollouts/eval JSON; default: temp dir")
    p.add_argument("--reuse-existing", action="store_true",
                   help="skip rollout/eval commands when the expected files already exist")
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
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_tag(path: Path, nfe: int, index: int) -> str:
    parent = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    return f"{index:03d}_{parent}_{path.stem}_nfe{nfe}".replace("/", "_")


def _run(cmd: list[str]) -> None:
    print("    $ " + " ".join(str(c) for c in cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    args = parse_args()
    real = args.real or (args.data_dir / "test.npz")
    candidates = list(dict.fromkeys(args.checkpoints))
    nfes = list(dict.fromkeys(args.nfe_steps))

    workdir_ctx = tempfile.TemporaryDirectory() if args.workdir is None else None
    workdir = Path(args.workdir) if args.workdir else Path(workdir_ctx.name)
    workdir.mkdir(parents=True, exist_ok=True)

    print(
        f"[select-joint] {len(candidates)} checkpoints x {len(nfes)} NFE values, "
        f"ranking by {args.rank_by}",
        file=sys.stderr,
    )
    results: list[dict] = []

    for ckpt_index, ckpt in enumerate(candidates):
        if not ckpt.exists():
            print(f"  [skip] missing checkpoint: {ckpt}", file=sys.stderr)
            continue
        for nfe in nfes:
            tag = _safe_tag(ckpt, nfe, ckpt_index)
            rollout_npz = workdir / f"rollout_{tag}.npz"
            eval_json = workdir / f"eval_{tag}.json"

            rollout_cmd = [
                sys.executable, str(SCRIPT_DIR / "rollout_joint.py"),
                "--checkpoint", str(ckpt),
                "--data-dir", str(args.data_dir),
                "--output", str(rollout_npz),
                "--n-paths", str(args.n_paths),
                "--n-steps", str(args.n_steps),
                "--fm-n-steps", str(nfe),
                "--fm-solver", args.fm_solver,
                "--cfg-w", str(args.cfg_w),
                "--device", args.device,
            ]
            if args.regime_actions:
                rollout_cmd.append("--regime-actions")
            if args.calibrate_moments:
                rollout_cmd.append("--calibrate-moments")

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

            print(f"  [{len(results) + 1}] ckpt={ckpt} nfe={nfe}", file=sys.stderr)
            if not (args.reuse_existing and rollout_npz.exists()):
                _run(rollout_cmd)
            if not (args.reuse_existing and eval_json.exists()):
                _run(eval_cmd)

            report = json.loads(eval_json.read_text(encoding="utf-8"))
            dist = report.get("distances", {})
            pricing = _pricing_block(report)
            facts = report.get("fake_facts", {})
            stylized = report.get("stylized_facts_comparison", {})
            row = {
                "checkpoint": str(ckpt),
                "nfe": nfe,
                "rank_metric": args.rank_by,
                "rank_value": _extract(report, args.rank_by),
                "pricing_rmse": pricing.get("rmse_overall"),
                "pricing_mape": pricing.get("mape_overall"),
                "rmse_per_maturity": pricing.get("rmse_per_maturity"),
                "kurtosis_fake": facts.get("kurtosis"),
                "kurtosis_diff": stylized.get("kurtosis_abs_diff"),
                "abs_acf_l1": stylized.get("absolute_return_acf_l1"),
                "leverage_l1": stylized.get("leverage_correlation_l1"),
                "total_w1": dist.get("total_return_wasserstein"),
                "abs_total_w1": dist.get("abs_total_return_wasserstein"),
                "marginal_w1": dist.get("marginal_wasserstein_mean"),
                "rollout_npz": str(rollout_npz),
                "eval_json": str(eval_json),
            }
            results.append(row)
            print(
                f"      {args.rank_by}={row['rank_value']} "
                f"pricing_rmse={row['pricing_rmse']} total_w1={row['total_w1']}",
                file=sys.stderr,
            )

    ranked = sorted(
        results,
        key=lambda r: (
            r["rank_value"] is None,
            r["rank_value"] if r["rank_value"] is not None else float("inf"),
        ),
    )
    out = {
        "rank_by": args.rank_by,
        "n_paths": args.n_paths,
        "n_steps": args.n_steps,
        "nfe_steps": nfes,
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[select-joint] wrote {args.output}", file=sys.stderr)

    print(json.dumps(out["best"], indent=2))
    if workdir_ctx is not None:
        workdir_ctx.cleanup()


if __name__ == "__main__":
    main()
