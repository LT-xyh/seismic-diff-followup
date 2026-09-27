from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.reproducibility import assert_release_paths, resolved_config_hash
from bg_pdr_fm.training.benchmark_config import benchmark_config_hash, load_benchmark_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "bg_pdr_fm" / "configs" / "release" / "aaai27"


def test_release_train_config_matches_paper_protocol() -> None:
    conf = load_benchmark_config(CONFIG_ROOT / "train.yaml")

    assert list(conf.data.openfwi_datasets) == [
        "FlatVelA",
        "FlatVelB",
        "CurveVelA",
        "CurveVelB",
        "FlatFaultA",
        "FlatFaultB",
        "CurveFaultA",
        "CurveFaultB",
    ]
    assert list(conf.data.split_fractions) == [0.7, 0.2, 0.1]
    assert conf.data.split_seed == 42
    assert conf.data.well_seed == 1234
    assert conf.training.seed == 2027
    assert conf.model.codec_type == "identity"
    assert conf.model.hidden_channels == 16
    assert conf.model.background_backend == "unet_direct"
    assert conf.model.background_hidden_channels == 192
    assert conf.model.residual_backend == "unet_fm_film"
    assert conf.model.residual_backend_hidden_channels == 128
    assert conf.model.residual_num_inference_steps == 50
    assert conf.quality_gate.mode == "none"
    assert conf.training.max_epochs == 100
    assert conf.training.batch_size == 64
    assert conf.training.precision == "bf16-mixed"
    assert conf.training.devices == 4
    assert conf.training.strategy == "ddp"
    assert conf.training.joint.lambda_contrastive == 0.01
    assert conf.training.load_stage_checkpoint is None
    assert conf.training.joint.warm_start_checkpoints.contrastive is None
    assert conf.training.joint.warm_start_checkpoints.background is None
    assert conf.training.joint.warm_start_checkpoints.residual is None


def test_release_profiles_resolve_only_relative_paths_and_stable_hashes() -> None:
    for name in ("base", "train", "smoke", "evaluate", "ablation"):
        path = CONFIG_ROOT / f"{name}.yaml"
        conf = load_benchmark_config(path)

        assert_release_paths(conf, REPO_ROOT)
        _assert_no_absolute_strings(OmegaConf.to_container(conf, resolve=True))
        assert benchmark_config_hash(path) == resolved_config_hash(conf)


def test_smoke_evaluate_and_ablation_profiles_preserve_release_contracts() -> None:
    train = load_benchmark_config(CONFIG_ROOT / "train.yaml")
    smoke = load_benchmark_config(CONFIG_ROOT / "smoke.yaml")
    evaluate = load_benchmark_config(CONFIG_ROOT / "evaluate.yaml")
    ablation = load_benchmark_config(CONFIG_ROOT / "ablation.yaml")

    expected_smoke_model = OmegaConf.to_container(train.model, resolve=True)
    expected_smoke_model["residual_num_inference_steps"] = 2
    assert OmegaConf.to_container(smoke.model, resolve=True) == expected_smoke_model
    assert OmegaConf.to_container(smoke.loss, resolve=True) == OmegaConf.to_container(train.loss, resolve=True)
    assert OmegaConf.to_container(smoke.contrastive, resolve=True) == OmegaConf.to_container(train.contrastive, resolve=True)
    assert smoke.data.name == "synthetic"
    assert smoke.training.max_epochs == 1
    assert smoke.training.limit_train_batches == 1
    assert smoke.training.limit_val_batches == 1
    assert smoke.training.num_workers == 0

    assert evaluate.evaluation.split == "test"
    assert evaluate.evaluation.shuffle is False
    assert evaluate.evaluation.allow_untrained is False
    assert evaluate.evaluation.checkpoints.full

    assert ablation.training.ablation.residual_target_source == "lowpass"
    assert ablation.ablation.concat_full_field.benchmark_variant == "concat_fm"
    assert ablation.ablation.prediction_consistency.residual_background_source == "predicted"


def test_native_entrypoints_load_extended_release_profiles(monkeypatch) -> None:
    from bg_pdr_fm.evaluation import evaluate_bg_pdr_fm
    from bg_pdr_fm.training import train_bg_pdr_fm

    captured: dict[str, Any] = {}
    monkeypatch.setattr(train_bg_pdr_fm, "run_one_stage", lambda conf: captured.setdefault("train", conf))
    monkeypatch.setattr(evaluate_bg_pdr_fm, "run_evaluation", lambda conf: captured.setdefault("evaluate", conf))

    train_bg_pdr_fm.main(str(CONFIG_ROOT / "train.yaml"))
    evaluate_bg_pdr_fm.main(str(CONFIG_ROOT / "evaluate.yaml"))

    assert captured["train"].training.stage == "joint_full"
    assert captured["train"].model.background_hidden_channels == 192
    assert captured["evaluate"].evaluation.split == "test"
    assert captured["evaluate"].data.split_seed == 42


def _assert_no_absolute_strings(value: Any) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_absolute_strings(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_strings(child)
    elif isinstance(value, str):
        assert not Path(value).is_absolute()
        assert not PureWindowsPath(value).is_absolute()
