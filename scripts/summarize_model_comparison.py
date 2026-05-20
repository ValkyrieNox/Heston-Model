#!/usr/bin/env python3
"""Build a markdown comparison table from model evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _fmt(value: object, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [
        ("FM teacher", args.eval_dir / "eval_fm.json"),
        ("Mean Flow best CFG", args.eval_dir / "eval_mf.json"),
        ("Consistency", args.eval_dir / "eval_cd.json"),
        ("Quant GAN", args.eval_dir / "eval_quant_gan.json"),
    ]

    lines = [
        "# Model Comparison",
        "",
        "| Model | Marginal W1 mean | Total-return W1 | Sig-W1 mean | Pricing RMSE | Pricing MAPE | JSON |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for label, path in models:
        report = json.loads(path.read_text(encoding="utf-8"))
        distances = report.get("distances", {})
        signature = distances.get("signature_wasserstein") or {}
        pricing = report.get("pricing_fake_vs_mc_oracle") or {}
        lines.append(
            "| "
            + " | ".join([
                label,
                _fmt(distances.get("marginal_wasserstein_mean")),
                _fmt(distances.get("total_return_wasserstein")),
                _fmt(signature.get("mean")),
                _fmt(pricing.get("rmse_overall")),
                _fmt(pricing.get("mape_overall")),
                str(path),
            ])
            + " |"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
