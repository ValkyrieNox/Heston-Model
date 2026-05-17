#!/usr/bin/env python3
"""Train V3 Stage 1b return transition kernel: p(r_{t+1} | v_{t+1}, v_t, r_t, a_t).

Training uses teacher-forced ground-truth v_{t+1} from the Heston data; at
inference time v_{t+1} is supplied by the Stage 1a sampler.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finflow.training import (
    TransitionFMTrainConfig,
    TwoStageFMModelConfig,
    load_num_actions,
    train_ret_trans_fm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/heston_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ret_trans_fm"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-actions", type=int, default=None,
                        help="Override action dim; default auto-detected from metadata.")

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--time-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-embedding-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_actions = args.num_actions
    if num_actions is None:
        num_actions = load_num_actions(args.data_dir)
    model_config = TwoStageFMModelConfig(
        state_dim=1,
        condition_dim=3 + num_actions,
        hidden_dim=args.hidden_dim,
        time_embedding_dim=args.time_embedding_dim,
        num_blocks=args.num_blocks,
    )
    train_config = TransitionFMTrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        time_eps=args.time_eps,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    summary = train_ret_trans_fm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_actions=num_actions,
        model_config=model_config,
        train_config=train_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
