from pathlib import Path

from finflow.data import (
    DEFAULT_REGIMES,
    DEFAULT_TRANSITION_MATRIX,
    generate_heston_dataset,
)
from finflow.training import (
    TransitionFMTrainConfig,
    TwoStageFMModelConfig,
    evaluate_two_stage_checkpoint,
    train_ret_trans_fm,
    train_vol_trans_fm,
)


_SHARED_TRAIN_KW = dict(
    batch_size=4, epochs=2, lr=1e-3, weight_decay=0.0,
    grad_clip_norm=1.0, seed=7, device="cpu",
    max_train_batches=2, max_val_batches=1,
)


def _generate_smoke_data(tmp_path: Path) -> tuple[Path, int]:
    data_dir = tmp_path / "data"
    metadata = generate_heston_dataset(
        data_dir, n_train=4, n_val=2, n_test=2, n_steps=6,
        regimes=DEFAULT_REGIMES, transition_matrix=DEFAULT_TRANSITION_MATRIX,
        initial_regime=0, seed=21, save_transitions=True,
    )
    return data_dir, metadata["num_actions"]


def test_vol_trans_fm_smoke(tmp_path: Path) -> None:
    data_dir, num_actions = _generate_smoke_data(tmp_path)
    summary = train_vol_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_vol",
        run_name="vol_smoke",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=1 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(**_SHARED_TRAIN_KW),
    )
    run_dir = Path(summary["run_dir"])
    best_ckpt = Path(summary["checkpoints"]["best"])
    assert run_dir.exists() and best_ckpt.exists()
    assert summary["stage"] == "vol"
    assert summary["num_actions"] == num_actions

    eval_result = evaluate_two_stage_checkpoint(
        best_ckpt, data_dir=data_dir, stage="vol", split="val",
        batch_size=2, device="cpu",
    )
    assert eval_result["loss"] >= 0.0
    assert eval_result["stage"] == "vol"


def test_ret_trans_fm_smoke(tmp_path: Path) -> None:
    data_dir, num_actions = _generate_smoke_data(tmp_path)
    summary = train_ret_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_ret",
        run_name="ret_smoke",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=3 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(**_SHARED_TRAIN_KW),
    )
    run_dir = Path(summary["run_dir"])
    best_ckpt = Path(summary["checkpoints"]["best"])
    assert run_dir.exists() and best_ckpt.exists()
    assert summary["stage"] == "ret"

    eval_result = evaluate_two_stage_checkpoint(
        best_ckpt, data_dir=data_dir, stage="ret", split="val",
        batch_size=2, device="cpu",
    )
    assert eval_result["loss"] >= 0.0
    assert eval_result["stage"] == "ret"
