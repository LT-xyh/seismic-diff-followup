from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from bg_pdr_fm.data.datasets import OpenFWIBGDataset
from bg_pdr_fm.evaluation import dispatch_parallel_eval as dispatch
from bg_pdr_fm.evaluation import evaluate_bg_pdr_fm as evaluate


def test_default_openfwi_shards_are_stable_for_three_workers():
    datasets = [
        "FlatVelA",
        "FlatVelB",
        "CurveVelA",
        "CurveVelB",
        "FlatFaultA",
        "FlatFaultB",
        "CurveFaultA",
        "CurveFaultB",
    ]

    shards = dispatch.make_dataset_shards(datasets, workers=3)

    assert shards == [
        ["FlatVelA", "FlatVelB", "FlatFaultA"],
        ["CurveVelA", "CurveFaultA", "FlatFaultB"],
        ["CurveVelB", "CurveFaultB"],
    ]


def test_prepare_worker_config_forces_single_device_and_metrics_only(tmp_path):
    global_datasets = ["FlatVelA", "FlatVelB", "CurveVelA", "CurveVelB"]
    conf = OmegaConf.create(
        {
            "data": {"openfwi_datasets": global_datasets},
            "training": {"batch_size": 16, "devices": 4, "strategy": "ddp"},
            "evaluation": {"output_dir": "old", "max_batches": 10, "save_arrays": True, "save_panels": True},
        }
    )

    worker = dispatch.prepare_worker_config(
        conf,
        output_dir=tmp_path / "worker_00",
        datasets=["CurveVelB"],
        batch_size=200,
        max_batches=None,
        save_arrays=False,
        save_panels=False,
    )

    assert OmegaConf.select(conf, "training.batch_size") == 16
    assert OmegaConf.select(worker, "data.openfwi_datasets") == global_datasets
    assert OmegaConf.select(worker, "data.post_split_datasets") == ["CurveVelB"]
    assert OmegaConf.select(worker, "training.batch_size") == 200
    assert OmegaConf.select(worker, "training.devices") == 1
    assert OmegaConf.select(worker, "training.strategy") is None
    assert OmegaConf.select(worker, "evaluation.output_dir") == str(tmp_path / "worker_00")
    assert OmegaConf.select(worker, "evaluation.save_arrays") is False
    assert OmegaConf.select(worker, "evaluation.save_panels") is False


def test_post_split_filter_preserves_global_held_out_membership():
    records = [
        {"dataset_name": dataset_name, "sample_index": sample_index}
        for dataset_name in ("DatasetA", "DatasetB")
        for sample_index in range(100)
    ]
    fractions = (0.7, 0.2, 0.1)

    global_train = OpenFWIBGDataset._select_split_then_filter(records, "train", fractions, 42, None)
    global_val = OpenFWIBGDataset._select_split_then_filter(records, "val", fractions, 42, None)
    global_test = OpenFWIBGDataset._select_split_then_filter(records, "test", fractions, 42, None)
    worker_test = OpenFWIBGDataset._select_split_then_filter(records, "test", fractions, 42, ["DatasetB"])

    record_id = lambda record: (record["dataset_name"], record["sample_index"])
    train_ids = {record_id(record) for record in global_train}
    val_ids = {record_id(record) for record in global_val}
    global_test_ids = {record_id(record) for record in global_test}
    worker_ids = {record_id(record) for record in worker_test}

    assert worker_ids == {record_id(record) for record in global_test if record["dataset_name"] == "DatasetB"}
    assert worker_ids.isdisjoint(train_ids)
    assert worker_ids.isdisjoint(val_ids)
    assert worker_ids <= global_test_ids


def test_evaluation_record_identity_uses_global_dataset_ids():
    batch = SimpleNamespace(
        metadata={
            "dataset_id": torch.tensor([3.0]),
            "sample_index": torch.tensor([1729.0]),
        }
    )

    identity = evaluate.evaluation_record_identity(
        batch,
        item_idx=0,
        dataset_names=["FlatVelA", "FlatVelB", "CurveVelA", "CurveVelB"],
    )

    assert identity == {
        "dataset_id": 3,
        "dataset_name": "CurveVelB",
        "source_sample_index": 1729,
    }


def test_evaluation_record_identity_allows_non_openfwi_metadata():
    batch = SimpleNamespace(
        metadata={
            "dataset_id": torch.tensor([0.0]),
            "sample_index": torch.tensor([31.0]),
        }
    )

    identity = evaluate.evaluation_record_identity(batch, item_idx=0, dataset_names=[])

    assert identity == {
        "dataset_id": 0,
        "dataset_name": "",
        "source_sample_index": 31,
    }


def test_merge_worker_metrics_writes_summary_and_group_csv(tmp_path):
    root = tmp_path / "eval"
    root.mkdir()
    fieldnames = [
        "sample_index",
        "dataset_id",
        "dataset_name",
        "source_sample_index",
        "ssim",
        "mae",
        "rmse",
        "mse",
        "mae_l",
        "mae_h",
        "bg_mae",
        "epsilon_H_B",
        "rho_B",
        "rho_hat_B",
        "residual_energy_ratio_l2",
        "transport_target_ratio",
        "alpha_hat",
    ]
    workers = []
    for idx, rows in enumerate(
        [
            [{"sample_index": "0", "dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "101", "ssim": "0.9", "mae": "0.1", "rmse": "0.2", "mse": "0.04", "mae_l": "0.07", "mae_h": "0.03", "bg_mae": "0.05", "epsilon_H_B": "0.01", "rho_B": "1.0", "rho_hat_B": "0.5", "residual_energy_ratio_l2": "0.25", "transport_target_ratio": "0.5", "alpha_hat": "0.5"}],
            [{"sample_index": "0", "dataset_id": "3", "dataset_name": "CurveVelB", "source_sample_index": "202", "ssim": "0.7", "mae": "0.3", "rmse": "0.4", "mse": "0.16", "mae_l": "0.2", "mae_h": "0.1", "bg_mae": "0.15", "epsilon_H_B": "0.03", "rho_B": "1.2", "rho_hat_B": "0.6", "residual_energy_ratio_l2": "0.5", "transport_target_ratio": "0.6", "alpha_hat": "0.4"}],
        ]
    ):
        worker_dir = root / f"worker_{idx:02d}"
        worker_dir.mkdir()
        metrics_path = worker_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (worker_dir / "summary.json").write_text(json.dumps({"num_samples": len(rows), "checkpoints": [{"path": "ckpt"}]}))
        workers.append(dispatch.WorkerResult(name=f"worker_{idx:02d}", output_dir=worker_dir, datasets=["D"], gpu=idx, returncode=0, start_time=0.0, end_time=1.0))

    summary = dispatch.merge_worker_outputs(
        output_dir=root,
        worker_results=workers,
        config_summary={"stage": "residual", "data_name": "openfwi"},
        expected_samples=2,
    )

    assert summary["status"] == "completed"
    assert summary["num_samples"] == 2
    assert summary["means"]["ssim"] == 0.8
    assert summary["means"]["mae_l"] == 0.135
    assert summary["means"]["residual_energy_ratio_l2"] == 0.375
    assert summary["means"]["transport_target_ratio"] == 0.55
    assert (root / "merged_metrics.csv").is_file()
    assert (root / "dataset_group_summary.csv").is_file()
    assert (root / "dataset_summary.csv").is_file()
    assert (root / "held_out_manifest.csv").is_file()
    assert (root / "summary.json").is_file()

    with (root / "dataset_summary.csv").open(newline="", encoding="utf-8") as handle:
        dataset_rows = list(csv.DictReader(handle))
    assert [row["dataset"] for row in dataset_rows] == ["CurveVelB", "FlatVelA"]
    assert [int(row["num_samples"]) for row in dataset_rows] == [1, 1]


def test_classify_worker_failure_detects_silent_no_output(tmp_path):
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()

    reason = dispatch.classify_worker_failure(worker_dir, returncode=0)

    assert reason == "silent_no_output"


def test_worker_state_marks_unfinished_process_as_running(tmp_path):
    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()

    state = dispatch.worker_state(
        dispatch.WorkerResult(
            name="worker_00",
            output_dir=worker_dir,
            datasets=["FlatVelA"],
            gpu=4,
            returncode=None,
            start_time=10.0,
            end_time=None,
            pid=123,
        )
    )

    assert state["status"] == "running"
    assert state["failure_reason"] == ""
    assert state["elapsed_seconds"] is None


def test_best_successful_batch_picks_largest_success():
    results = [
        {"batch_size": 100, "status": "success"},
        {"batch_size": 200, "status": "failed"},
        {"batch_size": 150, "status": "success"},
    ]

    assert dispatch.best_successful_batch(results) == 150
