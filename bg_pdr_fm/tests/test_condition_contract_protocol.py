from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from bg_pdr_fm.evaluation.visualize_condition_contract import (
    _draw_role_panel,
    aggregate_complete_modality_routing,
    build_subset_galleries,
    build_single_checkpoint_metadata,
    plot_figure3,
    role_separation_metrics,
    run_condition_contract_visualization,
    single_checkpoint_role_rows,
    select_best_records_by_mean_ssim,
    strict_derangement_indices,
    symmetric_recall_at_one,
    _validate_formal_full_seed_set,
    _write_best_sample_csv,
)
from bg_pdr_fm.evaluation.dispatch_condition_contract_evaluation import (
    build_evaluation_config,
    prepare_evaluation_configs,
    validate_paired_metric_identities,
)
from bg_pdr_fm.lightning import BGPDRFMLightning
from bg_pdr_fm.training.dispatch_condition_contract_protocol import (
    build_stage_configs,
    write_manifest,
)
from bg_pdr_fm.models.encoder import PhysicsDecoupledEncoder
from bg_pdr_fm.training.train_bg_pdr_fm import apply_training_seed


def _stage_config(stage: str) -> object:
    return OmegaConf.create(
        {
            "data": {"split_seed": 42, "well_seed": 1234},
            "training": {
                "stage": stage,
                "max_epochs": 100,
                "batch_size": 64,
                "load_stage_checkpoint": "old.ckpt",
                "joint": {
                    "lambda_contrastive": 0.01,
                    "warm_start_checkpoints": {
                        "contrastive": "old_contrastive.ckpt",
                        "background": "old_background.ckpt",
                        "residual": "old_residual.ckpt",
                    },
                },
                "logging": {"log_dir": "old/logs", "log_version": "old", "add_timestamp": True},
                "checkpoint": {"dirpath": "old/checkpoints", "filename": "old-{epoch}"},
            },
            "diagnostics": {"enabled": False, "output_dir": "old/diagnostics"},
            "evaluation": {
                "output_dir": "old/evaluation",
                "checkpoint": "old_evaluation.ckpt",
                "checkpoints": {
                    "full": "old_full.ckpt",
                    "contrastive": "old_contrastive.ckpt",
                    "background": "old_background.ckpt",
                    "residual": "old_residual.ckpt",
                },
            },
        }
    )


def test_apply_training_seed_replays_torch_randomness():
    apply_training_seed(2027)
    first = torch.rand(6)

    apply_training_seed(2027)
    repeated = torch.rand(6)

    apply_training_seed(3407)
    different = torch.rand(6)

    torch.testing.assert_close(first, repeated)
    assert not torch.allclose(first, different)


def test_apply_training_seed_replays_model_initialization():
    apply_training_seed(2027)
    first = torch.nn.Conv2d(3, 5, kernel_size=3).weight.detach().clone()

    apply_training_seed(2027)
    repeated = torch.nn.Conv2d(3, 5, kernel_size=3).weight.detach().clone()

    apply_training_seed(3407)
    different = torch.nn.Conv2d(3, 5, kernel_size=3).weight.detach().clone()

    torch.testing.assert_close(first, repeated)
    assert not torch.allclose(first, different)


def test_apply_training_seed_replays_dataloader_shuffle_order():
    dataset = TensorDataset(torch.arange(24))

    apply_training_seed(2027)
    first = torch.cat([batch[0] for batch in DataLoader(dataset, batch_size=4, shuffle=True)]).tolist()
    apply_training_seed(2027)
    repeated = torch.cat([batch[0] for batch in DataLoader(dataset, batch_size=4, shuffle=True)]).tolist()
    apply_training_seed(3407)
    different = torch.cat([batch[0] for batch in DataLoader(dataset, batch_size=4, shuffle=True)]).tolist()

    assert first == repeated
    assert first != different


def test_no_contract_stage_configs_do_not_reference_full_checkpoints(tmp_path):
    configs = build_stage_configs(
        regime="no_contract",
        seed=2027,
        run_dir=tmp_path / "no_contract" / "seed_2027",
        contrastive_config=_stage_config("contrastive"),
        background_config=_stage_config("background"),
        residual_config=_stage_config("residual"),
        joint_config=_stage_config("joint_full"),
        epochs=3,
        batch_size=8,
        accumulate_grad_batches=2,
    )

    assert list(configs) == ["background", "residual", "joint_full"]
    assert configs["background"].training.load_stage_checkpoint is None
    assert str(configs["residual"].training.load_stage_checkpoint).endswith("background/stage_checkpoints/background_last.ckpt")
    assert configs["joint_full"].training.load_stage_checkpoint is None
    warm = configs["joint_full"].training.joint.warm_start_checkpoints
    assert warm.contrastive is None
    assert str(warm.background).endswith("background/stage_checkpoints/background_last.ckpt")
    assert str(warm.residual).endswith("residual/stage_checkpoints/residual_last.ckpt")
    assert configs["joint_full"].training.joint.lambda_contrastive == 0.0
    assert all("old_" not in json.dumps(OmegaConf.to_container(conf, resolve=True)) for conf in configs.values())


def test_full_stage_configs_only_reference_same_seed_stage_outputs(tmp_path):
    configs = build_stage_configs(
        regime="full",
        seed=7777,
        run_dir=tmp_path / "full" / "seed_7777",
        contrastive_config=_stage_config("contrastive"),
        background_config=_stage_config("background"),
        residual_config=_stage_config("residual"),
        joint_config=_stage_config("joint_full"),
        epochs=3,
        batch_size=8,
        accumulate_grad_batches=2,
    )

    assert list(configs) == ["contrastive", "background", "residual", "joint_full"]
    assert str(configs["background"].training.load_stage_checkpoint).endswith("contrastive/stage_checkpoints/contrastive_last.ckpt")
    assert str(configs["residual"].training.load_stage_checkpoint).endswith("background/stage_checkpoints/background_last.ckpt")
    warm = configs["joint_full"].training.joint.warm_start_checkpoints
    assert str(warm.contrastive).endswith("contrastive/stage_checkpoints/contrastive_last.ckpt")
    assert str(warm.background).endswith("background/stage_checkpoints/background_last.ckpt")
    assert str(warm.residual).endswith("residual/stage_checkpoints/residual_last.ckpt")
    assert configs["joint_full"].training.joint.lambda_contrastive == 0.01
    for conf in configs.values():
        assert conf.training.seed == 7777
        assert conf.training.batch_size == 8
        assert conf.training.accumulate_grad_batches == 2
        assert conf.data.split_seed == 42
        assert conf.data.well_seed == 1234


def test_manifest_records_final_checkpoint_provenance(tmp_path):
    checkpoint = tmp_path / "joint_full" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    output = write_manifest(
        run_dir=tmp_path,
        regime="full",
        seed=2027,
        stage_configs={"joint_full": tmp_path / "joint_full.yaml"},
        stage_checkpoints={"joint_full": checkpoint},
        sampler_seed=1234,
        test_record_count=33_600,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["regime"] == "full"
    assert manifest["seed"] == 2027
    assert manifest["sampler_seed"] == 1234
    assert manifest["test_record_count"] == 33_600
    assert manifest["stage_checkpoints"]["joint_full"]["sha256"]


def test_strict_derangement_never_keeps_a_sample_in_place():
    indices = strict_derangement_indices(8, offset=3)

    assert np.array_equal(np.sort(indices), np.arange(8))
    assert np.all(indices != np.arange(8))


def test_subset_galleries_stay_within_subset_and_available_records():
    dataset_ids = np.repeat(np.asarray([0, 1]), 8)
    all_modalities = np.ones(16, dtype=bool)
    all_modalities[[1, 10]] = False

    galleries = build_subset_galleries(
        dataset_ids,
        all_modalities,
        gallery_size=3,
        galleries_per_subset=2,
        seed=42,
    )

    assert len(galleries) == 4
    for gallery in galleries:
        assert gallery.shape == (3,)
        assert len(set(gallery.tolist())) == 3
        assert all_modalities[gallery].all()
        assert len(np.unique(dataset_ids[gallery])) == 1


def test_symmetric_recall_at_one_is_symmetric_and_leaves_diagonal_nan():
    embeddings = {
        "pstm": np.eye(4, dtype=np.float32),
        "horizon": np.eye(4, dtype=np.float32),
        "rms": np.eye(4, dtype=np.float32),
        "well": np.eye(4, dtype=np.float32),
    }

    matrix = symmetric_recall_at_one(embeddings, np.arange(4))

    assert np.isnan(np.diag(matrix)).all()
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.allclose(matrix[~np.eye(4, dtype=bool)], 1.0)


def test_role_separation_metrics_are_bounded():
    background = torch.ones(2, 4, 8, 8)
    structural = torch.zeros(2, 4, 8, 8)
    structural[..., ::2, ::2] = 1.0

    metrics = role_separation_metrics(background, structural, kernel_size=3)

    assert 0.0 <= metrics["eta_b"] <= 1.0
    assert 0.0 <= metrics["eta_s"] <= 1.0
    assert 0.0 <= metrics["chi_bs"] <= 1.0


def test_encoder_routing_weights_match_normalized_learned_reliability():
    encoder = PhysicsDecoupledEncoder(base_channels=8, feature_channels=4)
    reliability = torch.tensor(
        [[[0.2, 0.9], [0.8, 0.3], [0.6, 0.4], [0.1, 0.2]]], dtype=torch.float32
    )
    available = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)

    structural, numerical = encoder.routing_weights(reliability, available)

    torch.testing.assert_close(structural.sum(dim=1), torch.ones(1))
    torch.testing.assert_close(numerical.sum(dim=1), torch.ones(1))
    assert structural[0, 2].item() == 0.0
    assert numerical[0, 2].item() == 0.0
    torch.testing.assert_close(
        structural,
        torch.tensor([[0.2 / 1.1, 0.8 / 1.1, 0.0, 0.1 / 1.1]]),
    )
    torch.testing.assert_close(
        numerical,
        torch.tensor([[0.9 / 1.4, 0.3 / 1.4, 0.0, 0.2 / 1.4]]),
    )


def _synthetic_seed_diagnostics_for_single_mode():
    from bg_pdr_fm.evaluation.visualize_condition_contract import SeedDiagnostics

    sample_count = 16
    dataset_ids = np.repeat(np.arange(2, dtype=np.int64), sample_count // 2)
    sample_indices = np.arange(sample_count, dtype=np.int64)
    availability = np.ones((sample_count, 4), dtype=bool)
    embeddings = {
        role: {
            modality: np.eye(sample_count, 8, dtype=np.float32)
            for modality in ("migrated_image", "horizon", "rms_vel", "well_log")
        }
        for role in ("background", "structural")
    }
    return SeedDiagnostics(
        seed=0,
        checkpoint="single.ckpt",
        checkpoint_info={"epoch": 99, "loaded_keys": 12},
        dataset_ids=dataset_ids,
        sample_indices=sample_indices,
        availability=availability,
        embeddings=embeddings,
        anchor_embeddings={
            "background": np.ones((sample_count, 8), dtype=np.float32),
            "structural": np.ones((sample_count, 8), dtype=np.float32),
        },
        eta_b=np.linspace(0.6, 0.9, sample_count).astype(np.float32),
        eta_s=np.linspace(0.5, 0.85, sample_count).astype(np.float32),
        chi_bs=np.linspace(0.1, 0.3, sample_count).astype(np.float32),
        routing_background=np.tile(np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32), (sample_count, 1)),
        routing_structural=np.tile(np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32), (sample_count, 1)),
        dataset_names=["FlatVelA", "FlatVelB"],
    )


def test_single_checkpoint_role_rows_preserve_all_sample_identities():
    data = _synthetic_seed_diagnostics_for_single_mode()

    rows = single_checkpoint_role_rows(data)

    assert len(rows) == 16
    assert {(row["dataset_id"], row["source_sample_index"]) for row in rows} == {
        (int(dataset_id), int(sample_index))
        for dataset_id, sample_index in zip(data.dataset_ids, data.sample_indices)
    }
    assert {row["evidence_scope"] for row in rows} == {"single_checkpoint_diagnostic"}


def test_single_checkpoint_routing_uses_all_complete_records_and_writes_8x4_shape():
    data = _synthetic_seed_diagnostics_for_single_mode()
    data.availability[1, 3] = False
    data.routing_background[1, 3] = 0.0
    data.routing_structural[1, 3] = 0.0
    data.routing_background[1, :3] = np.asarray([0.2, 0.3, 0.5], dtype=np.float32)
    data.routing_structural[1, :3] = np.asarray([0.5, 0.3, 0.2], dtype=np.float32)

    matrices, raw_rows, summary_rows = aggregate_complete_modality_routing(data)

    assert matrices["background"].shape == (2, 4)
    assert matrices["structural"].shape == (2, 4)
    assert len(summary_rows) == 2 * 2 * 4
    assert len(raw_rows) == (16 - 1) * 2 * 4
    assert all(row["record_type"] == "raw" for row in raw_rows)
    assert all(row["record_type"] == "summary" for row in summary_rows)
    assert np.allclose(matrices["background"].sum(axis=1), 1.0)
    assert np.allclose(matrices["structural"].sum(axis=1), 1.0)


def test_single_checkpoint_metadata_declares_noncausal_scope_and_input_hashes(tmp_path):
    paths = {}
    for name in ("checkpoint.ckpt", "config.yaml", "held_out_manifest.csv", "merged_metrics.csv"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    metadata = build_single_checkpoint_metadata(
        checkpoint_path=paths["checkpoint.ckpt"],
        config_path=paths["config.yaml"],
        manifest_path=paths["held_out_manifest.csv"],
        metrics_path=paths["merged_metrics.csv"],
        checkpoint_info={"epoch": 99, "loaded_keys": 1070},
        sample_counts={"manifest": 33600, "metrics": 33600, "diagnostics": 33600},
        split_audit={"train_val_overlap": 0, "test_count": 33600},
        gallery_config={"gallery_count": 80, "gallery_size": 128, "seed": 42},
    )

    assert metadata["evidence_scope"] == "single_checkpoint_diagnostic"
    assert metadata["causal_ablation"] is False
    assert metadata["checkpoint"]["epoch"] == 99
    assert metadata["sample_counts"]["manifest"] == 33600
    assert set(metadata["input_sha256"]) == {"checkpoint", "config", "manifest", "metrics"}


def test_best_sample_selection_is_deterministic_and_tie_breaks_by_source_index():
    rows = [
        {"seed": 2027, "dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 10, "ssim": 0.80, "all_modalities_available": True},
        {"seed": 3407, "dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 10, "ssim": 0.70, "all_modalities_available": True},
        {"seed": 7777, "dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 10, "ssim": 0.90, "all_modalities_available": True},
        {"seed": 2027, "dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 11, "ssim": 0.79, "all_modalities_available": True},
        {"seed": 3407, "dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 11, "ssim": 0.80, "all_modalities_available": True},
        {"seed": 7777, "dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 11, "ssim": 0.80, "all_modalities_available": True},
        {"seed": 2027, "dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 20, "ssim": 0.90, "all_modalities_available": True},
        {"seed": 3407, "dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 20, "ssim": 0.90, "all_modalities_available": True},
        {"seed": 7777, "dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 20, "ssim": 0.90, "all_modalities_available": True},
        {"seed": 2027, "dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 21, "ssim": 0.90, "all_modalities_available": True},
        {"seed": 3407, "dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 21, "ssim": 0.90, "all_modalities_available": True},
        {"seed": 7777, "dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 21, "ssim": 0.90, "all_modalities_available": True},
        {"seed": 2027, "dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 99, "ssim": 0.99, "all_modalities_available": False},
    ]

    selected = select_best_records_by_mean_ssim(rows, required_seeds=(2027, 3407, 7777))

    assert [(row["dataset_id"], row["source_sample_index"]) for row in selected] == [(0, 10), (1, 20)]
    assert np.isclose(selected[0]["mean_ssim"], 0.8)
    assert np.isclose(selected[1]["mean_ssim"], 0.9)


def test_best_sample_csv_records_per_seed_checkpoint_hashes(tmp_path):
    selected = [
        {
            "dataset_id": 0,
            "dataset_name": "FlatVelA",
            "source_sample_index": 17,
            "seed_ssim": {"2027": 0.91, "3407": 0.92, "7777": 0.90},
            "mean_ssim": 0.91,
            "all_modalities_available": True,
            "modality_availability": [True, True, True, True],
            "selection_rule": "highest mean Full SSIM",
        }
    ]
    hashes = {2027: "hash-2027", 3407: "hash-3407", 7777: "hash-7777"}

    output = _write_best_sample_csv(
        tmp_path,
        selected,
        (2027, 3407, 7777),
        checkpoint_sha256=hashes,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["checkpoint_sha256_seed_2027"] == "hash-2027"
    assert row["checkpoint_sha256_seed_3407"] == "hash-3407"
    assert row["checkpoint_sha256_seed_7777"] == "hash-7777"


def test_formal_figure_requires_the_registered_full_seed_set():
    assert _validate_formal_full_seed_set((7777, 2027, 3407)) == (2027, 3407, 7777)
    with pytest.raises(ValueError, match="require Full seeds"):
        _validate_formal_full_seed_set((2027,))


def test_condition_contract_evaluation_config_loads_joint_checkpoint_only(tmp_path):
    config = build_evaluation_config(
        _stage_config("joint_full"),
        regime="full",
        seed=2027,
        checkpoint=tmp_path / "joint_full_last.ckpt",
        output_dir=tmp_path / "evaluation",
    )

    assert config.training.stage == "joint_full"
    assert config.training.load_stage_checkpoint is None
    assert config.training.joint.warm_start_checkpoints.contrastive is None
    assert config.training.joint.warm_start_checkpoints.background is None
    assert config.training.joint.warm_start_checkpoints.residual is None
    assert config.evaluation.checkpoints.full == str(tmp_path / "joint_full_last.ckpt")
    assert config.evaluation.checkpoints.contrastive is None
    assert config.evaluation.output_dir == str(tmp_path / "evaluation")
    assert config.data.post_split_datasets is None


def test_condition_contract_evaluation_prepare_rejects_prepared_manifest(tmp_path):
    run_dir = tmp_path / "condition_contract" / "full" / "seed_2027"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"regime": "full", "seed": 2027, "status": "prepared"}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected 'completed'"):
        prepare_evaluation_configs(
            condition_root=tmp_path / "condition_contract",
            output_root=tmp_path / "evaluations",
            regimes=("full",),
            seeds=(2027,),
        )


def test_condition_contract_evaluation_rejects_paired_identity_mismatch(tmp_path):
    fields = ["dataset_id", "dataset_name", "source_sample_index", "ssim"]
    rows = [
        {"dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 4, "ssim": 0.9},
        {"dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 8, "ssim": 0.8},
    ]
    paths = {}
    for regime, regime_rows in (
        ("full", rows),
        ("no_contract", [rows[0], {**rows[1], "source_sample_index": 9}]),
    ):
        path = tmp_path / regime / "metrics.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(regime_rows)
        paths[regime] = path

    jobs = [
        {"regime": "full", "seed": 2027, "metrics": str(paths["full"])},
        {"regime": "no_contract", "seed": 2027, "metrics": str(paths["no_contract"])},
    ]
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_paired_metric_identities(jobs, expected_samples=2)


def test_role_separation_panel_keeps_metric_labels_out_of_the_title_area():
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    _draw_role_panel(
        axis,
        [
            {"regime": "full", "seed": 2027, "eta_b": 0.75, "eta_s": 0.66, "chi_bs": 0.18},
            {"regime": "no_contract", "seed": 2027, "eta_b": 0.61, "eta_s": 0.53, "chi_bs": 0.31},
        ],
    )

    assert all(text.get_position()[1] <= 1.0 for text in axis.texts)
    plt.close(figure)


def test_condition_contract_visualization_writes_publication_assets(tmp_path):
    config = OmegaConf.create(
        {
            "data": {"name": "synthetic", "test_length": 8, "train_length": 8, "val_length": 8},
            "model": {
                "hidden_channels": 8,
                "base_channels": 16,
                "feature_channels": 8,
                "cond_channels": 16,
                "background_hidden_channels": 16,
                "latent_channels": 4,
                "latent_hw": [18, 18],
                "codec_type": "simple",
                "codec_checkpoint": None,
                "lowpass_kernel": 5,
                "residual_backend": "simple_fm",
                "residual_num_inference_steps": 2,
            },
            "quality_gate": {"mode": "none"},
            "contrastive": {
                "embed_dim": 16,
                "temperature": 0.1,
                "symile_structural_weight": 1.0,
                "symile_numerical_weight": 1.0,
                "anchor_weight": 0.5,
                "reliability_prior_weight": 0.05,
                "sn_ortho_weight": 0.05,
                "unique_ortho_weight": 0.05,
                "min_symile_group_size": 2,
                "wavelet": {"level": 1, "boundary": "edge", "resize_mode": "nearest", "detail_fusion": "l2"},
            },
            "training": {"stage": "joint_full", "lr": 1e-4, "load_stage_checkpoint": None, "save_stage_outputs": False},
        }
    )
    config_path = tmp_path / "synthetic.yaml"
    OmegaConf.save(config, config_path)
    checkpoint = tmp_path / "joint_full.ckpt"
    torch.save({"state_dict": BGPDRFMLightning(config).state_dict()}, checkpoint)

    result = run_condition_contract_visualization(
        config=config_path,
        full_checkpoints=[(2027, checkpoint)],
        no_contract_checkpoints=[(2027, checkpoint)],
        output_dir=tmp_path / "figure3",
        device="cpu",
        batch_size=4,
        gallery_size=2,
        galleries_per_subset=1,
    )

    assert result["chance_recall_at_1"] == 0.5
    assert result["full_checkpoints"][0]["num_samples"] == 8
    assert "embeddings" not in result["full_checkpoints"][0]
    for name in (
        "figure3_condition_learning.pdf",
        "figure3_condition_learning.svg",
        "figure3_condition_learning.png",
        "retrieval_matrices.csv",
        "anchor_calibration.csv",
        "role_separation.csv",
        "metadata.json",
    ):
        assert (tmp_path / "figure3" / name).is_file()

    metadata = json.loads((tmp_path / "figure3" / "metadata.json").read_text(encoding="utf-8"))
    checkpoint_sha256 = metadata["full_checkpoints"][0]["checkpoint_info"]["sha256"]
    assert len(checkpoint_sha256) == 64

    from PIL import Image

    with Image.open(tmp_path / "figure3" / "figure3_condition_learning.png") as image:
        # The source figure is exported at AAAI two-column width so LaTeX does
        # not halve all labels when it includes the asset at \textwidth.
        assert image.width <= 2500


def test_spatial_condition_panel_exports_compact_publication_asset(tmp_path):
    output_dir = tmp_path / "spatial"
    output_dir.mkdir()
    coordinates = np.indices((24, 24)).sum(axis=0).astype(np.float32)
    retrieval = np.array(
        [
            [np.nan, 0.42, 0.35, 0.31],
            [0.42, np.nan, 0.28, 0.26],
            [0.35, 0.28, np.nan, 0.37],
            [0.31, 0.26, 0.37, np.nan],
        ]
    )
    examples = [
        {
            "dataset_name": dataset_name,
            "target": coordinates,
            "anchor_b": coordinates * 0.8,
            "probe_b": coordinates * 0.7,
            "anchor_s": np.sin(coordinates),
            "probe_s": np.cos(coordinates),
        }
        for dataset_name in ("FlatFaultB", "CurveFaultB")
    ]

    outputs = plot_figure3(
        output_dir=output_dir,
        retrieval_matrices={"background": retrieval, "structural": retrieval * 0.9},
        anchor_values={
            "background": (np.linspace(0.55, 0.8, 16), np.linspace(-0.25, 0.7, 16)),
            "structural": (np.linspace(0.75, 0.95, 16), np.linspace(-0.35, 0.85, 16)),
        },
        role_rows=[
            {"regime": "full", "seed": 2027, "eta_b": 0.76, "eta_s": 0.66, "chi_bs": 0.18},
            {"regime": "no_contract", "seed": 2027, "eta_b": 0.62, "eta_s": 0.48, "chi_bs": 0.31},
        ],
        spatial_examples=examples,
    )

    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    from PIL import Image

    with Image.open(output_dir / "figure3_condition_learning.png") as image:
        assert image.width <= 2500
        assert image.height <= 1800
