from pathlib import Path
import json

from finflow.data import (
    DEFAULT_REGIMES,
    DEFAULT_TRANSITION_MATRIX,
    generate_heston_dataset,
)
from finflow.training import (
    TransitionFMTrainConfig,
    TwoStageFMModelConfig,
    evaluate_two_stage_checkpoint,
    load_checkpoint,
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


def test_vol_trans_fm_cache_data_device_smoke(tmp_path: Path) -> None:
    data_dir, num_actions = _generate_smoke_data(tmp_path)
    summary = train_vol_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_vol_cache",
        run_name="vol_cache",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=1 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(
            **{
                **_SHARED_TRAIN_KW,
                "epochs": 1,
                "max_train_batches": 1,
                "cache_data_device": True,
            }
        ),
    )
    assert Path(summary["checkpoints"]["best"]).exists()
    assert summary["history"][0]["global_step"] == 1


def test_vol_trans_fm_lambert_w_delta_is_checkpointed(tmp_path: Path) -> None:
    data_dir, num_actions = _generate_smoke_data(tmp_path)
    summary = train_vol_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_vol_lw",
        run_name="vol_lw",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=1 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(
            **{
                **_SHARED_TRAIN_KW,
                "epochs": 1,
                "max_train_batches": 1,
            }
        ),
        lambert_w_delta=0.05,
    )
    ckpt = load_checkpoint(summary["checkpoints"]["best"])
    assert ckpt["extra"]["lambert_w_delta"] == 0.05


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


def test_ret_trans_fm_scheduled_sampling_smoke(tmp_path: Path) -> None:
    data_dir, num_actions = _generate_smoke_data(tmp_path)
    vol_summary = train_vol_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_vol_sched",
        run_name="vol_sched",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=1 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(
            **{**_SHARED_TRAIN_KW, "epochs": 1, "max_train_batches": 1}
        ),
    )
    ret_summary = train_ret_trans_fm(
        data_dir=data_dir,
        output_dir=tmp_path / "runs_ret_sched",
        run_name="ret_sched",
        num_actions=num_actions,
        model_config=TwoStageFMModelConfig(
            state_dim=1, condition_dim=3 + num_actions,
            hidden_dim=16, time_embedding_dim=8, num_blocks=2,
        ),
        train_config=TransitionFMTrainConfig(
            **{
                **_SHARED_TRAIN_KW,
                "epochs": 2,
                "max_train_batches": 1,
                "scheduled_sampling_max_prob": 0.5,
            }
        ),
        vol_sampler_checkpoint=vol_summary["checkpoints"]["best"],
    )

    history = ret_summary["history"]
    assert history[0]["scheduled_sampling_prob"] == 0.25
    assert history[-1]["scheduled_sampling_prob"] == 0.5
    config = json.loads((Path(ret_summary["run_dir"]) / "config.json").read_text(encoding="utf-8"))
    assert config["scheduled_sampling_enabled"] is True
