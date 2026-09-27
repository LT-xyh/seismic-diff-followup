"""Tests for the audited Table 2 qualitative-selection helpers."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from bg_pdr_fm.evaluation import visualize_table2_qualitative as qualitative


class Table2QualitativeSelectionTest(unittest.TestCase):
    def test_variant_state_loader_ignores_unrelated_wrapper_modules_but_requires_variant_keys(self):
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapted_auto_linear_original = torch.nn.Linear(2, 1)
                self.unrelated = torch.nn.Linear(2, 1)

        model = Wrapper()
        checkpoint = {
            "adapted_auto_linear_original.weight": torch.full((1, 2), 3.0),
            "adapted_auto_linear_original.bias": torch.full((1,), 4.0),
            "obsolete_module.weight": torch.ones((1, 1)),
        }

        result = qualitative.load_variant_state_dict(
            model,
            checkpoint,
            prefix="adapted_auto_linear_original.",
            method_name="Auto-Linear",
        )

        self.assertEqual(result["missing_keys"], 0)
        self.assertEqual(result["unexpected_keys"], 0)
        self.assertTrue(torch.equal(model.adapted_auto_linear_original.weight, torch.full((1, 2), 3.0)))
        with self.assertRaisesRegex(RuntimeError, "missing critical"):
            qualitative.load_variant_state_dict(
                Wrapper(),
                {"adapted_auto_linear_original.bias": torch.full((1,), 4.0)},
                prefix="adapted_auto_linear_original.",
                method_name="Auto-Linear",
            )

    def test_data_protocol_signature_accepts_scalar_and_sequence_config_values(self):
        conf = OmegaConf.create(
            {
                "data": {
                    "name": "openfwi",
                    "use_normalize": "-1_1",
                    "storage_backend": "lmdb",
                    "lmdb_root": "/tmp/openfwi",
                    "normalization_profile": "auto",
                    "well_seed": 1234,
                    "split_seed": 42,
                    "split_fractions": [0.7, 0.2, 0.1],
                    "openfwi_datasets": ["FlatVelA", "CurveVelB"],
                }
            }
        )

        signature = qualitative._data_protocol_signature(conf)

        self.assertEqual(signature["data.name"], "openfwi")
        self.assertEqual(signature["data.split_fractions"], [0.7, 0.2, 0.1])

    def test_profile_resolution_and_velocity_conversion_preserve_physical_target(self):
        baseline_conf = OmegaConf.create({"data": {"name": "openfwi", "normalization_profile": "auto", "use_normalize": "-1_1"}, "model": {"codec_type": "autoencoder"}})
        pdr_conf = OmegaConf.create({"data": {"name": "openfwi", "normalization_profile": "auto", "use_normalize": "-1_1"}, "model": {"codec_type": "identity"}})

        self.assertEqual(qualitative.resolved_normalization_profile(baseline_conf), "seismic_global")
        self.assertEqual(qualitative.resolved_normalization_profile(pdr_conf), "openfwi")
        seismic_global = qualitative.velocity_to_physical(np.array([[-0.4]], dtype=np.float32), "seismic_global", "-1_1")
        openfwi = qualitative.velocity_to_physical(np.array([[0.0]], dtype=np.float32), "openfwi", "-1_1")
        self.assertAlmostEqual(float(seismic_global[0, 0]), 2500.0)
        self.assertAlmostEqual(float(openfwi[0, 0]), 3000.0)
        canonical = qualitative.physical_to_openfwi_normalized(np.array([[3000.0]], dtype=np.float32))
        self.assertAlmostEqual(float(canonical[0, 0]), 0.0)

    def test_metric_csv_loader_filters_full_mode_and_rejects_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["dataset_name", "dataset_sample_index", "missing_mode", "mae", "mae_l", "mae_h", "ssim"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"dataset_name": "FlatVelA", "dataset_sample_index": 3, "missing_mode": "full",
                         "mae": 0.1, "mae_l": 0.1, "mae_h": 0.1, "ssim": 0.9},
                        {"dataset_name": "FlatVelA", "dataset_sample_index": 3, "missing_mode": "PSTM only",
                         "mae": 0.2, "mae_l": 0.2, "mae_h": 0.2, "ssim": 0.8},
                    ]
                )

            loaded = qualitative.load_metric_csv(path, "dataset_sample_index", "InversionNet")
            self.assertEqual(set(loaded), {("FlatVelA", 3)})
            self.assertAlmostEqual(float(loaded[("FlatVelA", 3)]["ssim"]), 0.9)

            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset_name", "dataset_sample_index", "missing_mode", "mae", "mae_l", "mae_h", "ssim"])
                writer.writerow({"dataset_name": "FlatVelA", "dataset_sample_index": 3, "missing_mode": "full",
                                 "mae": 0.3, "mae_l": 0.3, "mae_h": 0.3, "ssim": 0.7})
            with self.assertRaisesRegex(ValueError, "InversionNet.*duplicate"):
                qualitative.load_metric_csv(path, "dataset_sample_index", "InversionNet")

    def test_write_csv_rows_uses_lf_line_endings_for_tracked_audit_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            qualitative.write_csv_rows(path, [{"dataset_name": "FlatVelA", "value": 1.0}])

            self.assertNotIn(b"\r\n", path.read_bytes())

    def test_alignment_requires_each_method_to_match_manifest_identity(self):
        manifest = {("FlatVelA", 3), ("CurveVelB", 9)}
        baselines = {
            "InversionNet": {("FlatVelA", 3): {"mae_l": 0.1}, ("CurveVelB", 9): {"mae_l": 0.2}},
            "VelocityGAN": {("FlatVelA", 3): {"mae_l": 0.1}, ("CurveVelB", 9): {"mae_l": 0.2}},
        }
        pd_metrics = {("FlatVelA", 3): {"mae_l": 0.05}, ("CurveVelB", 9): {"mae_l": 0.1}}

        qualitative.assert_exact_held_out_alignment(manifest, baselines, pd_metrics)

        broken = dict(baselines)
        broken["VelocityGAN"] = {("FlatVelA", 3): {"mae_l": 0.1}, ("CurveVelB", 10): {"mae_l": 0.2}}
        with self.assertRaisesRegex(ValueError, "VelocityGAN.*extra=.*1.*missing=.*1"):
            qualitative.assert_exact_held_out_alignment(manifest, broken, pd_metrics)

    def test_alignment_rejects_dataset_id_mismatch_for_an_otherwise_matching_record(self):
        key = ("FlatVelA", 3)
        manifest = {key}
        manifest_dataset_ids = {key: 0}
        baselines = {"InversionNet": {key: {"dataset_id": "9", "mae_l": 0.1}}}
        pd_metrics = {key: {"dataset_id": "0", "mae_l": 0.05}}

        with self.assertRaisesRegex(ValueError, "InversionNet.*dataset_id mismatch"):
            qualitative.assert_exact_held_out_alignment(
                manifest,
                baselines,
                pd_metrics,
                manifest_dataset_ids=manifest_dataset_ids,
            )

    def test_manifest_metadata_enforces_expected_cardinality(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset_id", "dataset_name", "source_sample_index"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 1},
                        {"dataset_id": 1, "dataset_name": "CurveVelA", "source_sample_index": 2},
                    ]
                )

            metadata = qualitative.load_manifest_metadata(path, expected_count=2)
            self.assertEqual(metadata[("FlatVelA", 1)], 0)
            with self.assertRaisesRegex(ValueError, "expected 3"):
                qualitative.load_manifest_metadata(path, expected_count=3)

    def test_dataset_record_index_rejects_dataset_id_mismatch(self):
        key = ("FlatVelA", 3)
        manifest_dataset_ids = {key: 0}

        indices = qualitative.index_verified_dataset_records(
            [{"dataset_id": 0, "dataset_name": "FlatVelA", "sample_index": 3}],
            manifest_dataset_ids,
            "InversionNet",
        )

        self.assertEqual(indices, {key: 0})
        with self.assertRaisesRegex(ValueError, "InversionNet.*dataset_id mismatch"):
            qualitative.index_verified_dataset_records(
                [{"dataset_id": 9, "dataset_name": "FlatVelA", "sample_index": 3}],
                manifest_dataset_ids,
                "InversionNet",
            )

    def test_selection_manifest_uses_canonical_dataset_id(self):
        key = ("FlatVelA", 3)
        metric_values = {"mae": 0.1, "rmse": 0.1, "ssim": 0.9, "mae_l": 0.1, "mae_h": 0.1}
        rows = qualitative.selection_manifest_rows(
            selections={
                "advantage": [{
                    "status": "selected",
                    "criterion": "margin_l",
                    "dataset_name": key[0],
                    "source_sample_index": key[1],
                }],
            },
            formal_rows=[{
                "dataset_name": key[0],
                "source_sample_index": key[1],
                "margin_l": 0.01,
                "margin_h": 0.02,
                "margin_s": 0.03,
            }],
            baselines={"InversionNet": {key: metric_values}},
            pd_metrics={key: metric_values},
            checkpoints={"InversionNet": {"path": "inversion.ckpt"}, "PD-BG-RFM": {"path": "pdr.ckpt"}},
            manifest_dataset_ids={key: 7},
            base_seed=2027,
        )

        self.assertEqual(rows[0]["dataset_id"], 7)

    def test_margin_rows_use_best_baseline_per_metric(self):
        key = ("FlatVelA", 3)
        rows = qualitative.build_margin_rows(
            baselines={
                "InversionNet": {key: {"mae_l": 0.10, "mae_h": 0.20, "ssim": 0.80}},
                "VelocityGAN": {key: {"mae_l": 0.12, "mae_h": 0.15, "ssim": 0.91}},
            },
            pd_metrics={key: {"mae": 0.07, "mae_l": 0.08, "mae_h": 0.10, "ssim": 0.93}},
        )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["margin_l"], 0.02)
        self.assertAlmostEqual(rows[0]["margin_h"], 0.05)
        self.assertAlmostEqual(rows[0]["margin_s"], 0.02)

    def test_case_selection_uses_positive_margins_and_distinct_records(self):
        rows = [
            {"dataset_name": "FlatVelA", "source_sample_index": 1, "pd_mae": 0.10,
             "margin_l": 0.90, "margin_h": 0.10, "margin_s": 0.10},
            {"dataset_name": "CurveVelA", "source_sample_index": 2, "pd_mae": 0.20,
             "margin_l": 0.20, "margin_h": 0.80, "margin_s": 0.30},
            {"dataset_name": "FlatFaultA", "source_sample_index": 3, "pd_mae": 0.30,
             "margin_l": 0.10, "margin_h": 0.20, "margin_s": 0.95},
            {"dataset_name": "CurveFaultB", "source_sample_index": 4, "pd_mae": 0.40,
             "margin_l": -0.70, "margin_h": -0.80, "margin_s": -0.90},
            {"dataset_name": "FlatVelA", "source_sample_index": 5, "pd_mae": 0.50,
             "margin_l": -0.30, "margin_h": -0.20, "margin_s": -0.10},
        ]

        selected = qualitative.select_case_records(rows)

        advantage = selected["advantage"]
        self.assertEqual([case["criterion"] for case in advantage], ["margin_l", "margin_h", "margin_s"])
        self.assertEqual({case["source_sample_index"] for case in advantage}, {1, 2, 3})
        self.assertTrue(all(case[case["criterion"]] > 0 for case in advantage))
        failure = selected["failure"]
        self.assertEqual({case["source_sample_index"] for case in failure[:2]}, {4, 5})
        self.assertEqual(failure[2]["status"], "no_negative_candidate")
        self.assertIsNone(failure[2]["source_sample_index"])

    def test_case_selection_reports_missing_positive_candidate_without_substitution(self):
        rows = [
            {"dataset_name": "FlatVelA", "source_sample_index": 1, "pd_mae": 0.10,
             "margin_l": -0.10, "margin_h": 0.20, "margin_s": 0.30},
            {"dataset_name": "CurveVelA", "source_sample_index": 2, "pd_mae": 0.20,
             "margin_l": -0.20, "margin_h": 0.10, "margin_s": 0.10},
            {"dataset_name": "FlatFaultA", "source_sample_index": 3, "pd_mae": 0.30,
             "margin_l": -0.30, "margin_h": -0.10, "margin_s": -0.10},
        ]

        selected = qualitative.select_case_records(rows)

        self.assertEqual(selected["advantage"][0]["criterion"], "margin_l")
        self.assertEqual(selected["advantage"][0]["status"], "no_positive_candidate")
        self.assertIsNone(selected["advantage"][0]["source_sample_index"])

    def test_complete_panel_selection_rejects_missing_required_case(self):
        selections = {
            "advantage": [{"status": "selected"}, {"status": "selected"}, {"status": "no_positive_candidate"}],
            "representative": [{"status": "selected"}] * 8,
            "failure": [{"status": "selected"}] * 3,
        }

        with self.assertRaisesRegex(RuntimeError, "advantage"):
            qualitative.assert_complete_panel_selections(selections)

    def test_rendered_advantage_selection_preserves_formal_margin_and_records_fixed_seed_margin(self):
        rows = [
            {"dataset_name": "FlatVelA", "source_sample_index": 1, "pd_mae": 0.1,
             "margin_l": 0.9, "margin_h": 0.1, "margin_s": 0.1},
            {"dataset_name": "CurveVelA", "source_sample_index": 2, "pd_mae": 0.2,
             "margin_l": 0.2, "margin_h": 0.8, "margin_s": 0.3},
            {"dataset_name": "FlatFaultA", "source_sample_index": 3, "pd_mae": 0.3,
             "margin_l": 0.1, "margin_h": 0.2, "margin_s": 0.95},
        ]
        artifacts = {
            ("FlatVelA", 1): {"metrics": {"Baseline": {"mae_l": 0.3, "mae_h": 0.3, "ssim": 0.8}, "PD-BG-RFM": {"mae_l": 0.2, "mae_h": 0.2, "ssim": 0.9}}},
            ("CurveVelA", 2): {"metrics": {"Baseline": {"mae_l": 0.3, "mae_h": 0.3, "ssim": 0.8}, "PD-BG-RFM": {"mae_l": 0.2, "mae_h": 0.1, "ssim": 0.9}}},
            ("FlatFaultA", 3): {"metrics": {"Baseline": {"mae_l": 0.3, "mae_h": 0.3, "ssim": 0.8}, "PD-BG-RFM": {"mae_l": 0.2, "mae_h": 0.2, "ssim": 0.95}}},
        }

        selected, _ = qualitative.select_rendered_advantage_cases(
            formal_rows=rows,
            reconstruct=lambda key: artifacts[key],
        )

        self.assertAlmostEqual(selected[0]["margin_l"], 0.9)
        self.assertAlmostEqual(selected[0]["rendered_margin_l"], 0.1)
        self.assertAlmostEqual(selected[1]["margin_h"], 0.8)
        self.assertAlmostEqual(selected[1]["rendered_margin_h"], 0.2)
        self.assertAlmostEqual(selected[2]["margin_s"], 0.95)
        self.assertAlmostEqual(selected[2]["rendered_margin_s"], 0.15)

    def test_per_subset_cases_choose_maximum_ssim_margin_with_stable_tiebreak(self):
        rows = [
            {"dataset_name": "FlatVelA", "source_sample_index": 1, "pd_mae": 0.10,
             "margin_l": 0.0, "margin_h": 0.0, "margin_s": 0.12},
            {"dataset_name": "FlatVelA", "source_sample_index": 2, "pd_mae": 0.20,
             "margin_l": 0.0, "margin_h": 0.0, "margin_s": 0.35},
            {"dataset_name": "FlatVelA", "source_sample_index": 3, "pd_mae": 0.90,
             "margin_l": 0.0, "margin_h": 0.0, "margin_s": 0.35},
            {"dataset_name": "CurveVelB", "source_sample_index": 8, "pd_mae": 0.40,
             "margin_l": 0.0, "margin_h": 0.0, "margin_s": -0.10},
            {"dataset_name": "CurveVelB", "source_sample_index": 9, "pd_mae": 0.30,
             "margin_l": 0.0, "margin_h": 0.0, "margin_s": -0.25},
        ]

        selected = qualitative.select_case_records(rows)

        representatives = {case["dataset_name"]: case for case in selected["representative"]}
        self.assertEqual(representatives["FlatVelA"]["source_sample_index"], 2)
        self.assertEqual(representatives["CurveVelB"]["source_sample_index"], 8)
        self.assertEqual(representatives["FlatVelA"]["criterion"], "margin_s_max")
        self.assertEqual(representatives["CurveVelB"]["criterion"], "margin_s_max")
        self.assertAlmostEqual(representatives["CurveVelB"]["margin_s"], -0.10)

    def test_chunk_cases_keeps_order_and_balances_appendix_panels(self):
        cases = [{"dataset_name": "FlatVelA", "source_sample_index": index} for index in range(8)]

        chunks = qualitative.chunk_cases(cases, max_cases_per_panel=4)

        self.assertEqual([len(chunk) for chunk in chunks], [4, 4])
        self.assertEqual(chunks[0][0]["source_sample_index"], 0)
        self.assertEqual(chunks[1][-1]["source_sample_index"], 7)

    def test_standalone_velocity_export_writes_pixel_only_images_and_manifest_rows(self):
        key = ("FlatVelA", 3)
        case = {
            "dataset_name": key[0],
            "source_sample_index": key[1],
            "status": "selected",
        }
        target = np.linspace(1500.0, 4500.0, 16, dtype=np.float32).reshape(4, 4)
        artifacts = {
            key: {
                "target": target,
                "predictions": {"InversionNet": target + 10.0},
                "metrics": {"InversionNet": {"ssim": 0.98765}},
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            rows = qualitative.render_standalone_velocity_images(
                cases=[case],
                artifacts=artifacts,
                method_names=("InversionNet",),
                output_dir=Path(directory),
                velocity_limits=(1500.0, 4500.0),
            )

            self.assertEqual(len(rows), 2)
            target_row, prediction_row = rows
            self.assertEqual(target_row["method"], "Ground Truth")
            self.assertEqual(target_row["ssim"], 1.0)
            self.assertEqual(prediction_row["method"], "InversionNet")
            self.assertAlmostEqual(prediction_row["ssim"], 0.98765)
            self.assertEqual(prediction_row["dataset_name"], "FlatVelA")
            self.assertEqual(prediction_row["source_sample_index"], 3)
            self.assertIn("flatvela", prediction_row["filename"])
            self.assertIn("inversionnet", prediction_row["filename"])
            self.assertTrue((Path(directory) / target_row["filename"]).is_file())
            self.assertTrue((Path(directory) / prediction_row["filename"]).is_file())
            from PIL import Image

            with Image.open(Path(directory) / prediction_row["filename"]) as image:
                self.assertEqual(image.size, (4, 4))

    def test_standalone_error_export_writes_pixel_only_method_errors_and_manifest_rows(self):
        key = ("FlatVelA", 3)
        case = {
            "dataset_name": key[0],
            "source_sample_index": key[1],
            "status": "selected",
        }
        target = np.zeros((4, 4), dtype=np.float32)
        prediction = np.full((4, 4), 10.0, dtype=np.float32)
        artifacts = {
            key: {
                "target": target,
                "predictions": {"InversionNet": prediction},
                "metrics": {"InversionNet": {"mae": 10.0, "rmse": 10.0, "ssim": 0.8}},
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            rows = qualitative.render_standalone_error_images(
                cases=[case],
                artifacts=artifacts,
                method_names=("InversionNet",),
                output_dir=Path(directory),
                error_limits=(0.0, 100.0),
            )

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["method"], "InversionNet")
            self.assertEqual(row["reference"], "Ground Truth")
            self.assertAlmostEqual(row["mae"], 10.0)
            self.assertIn("error", row["filename"])
            self.assertIn("mae_10.0000", row["filename"])
            from PIL import Image

            with Image.open(Path(directory) / row["filename"]) as image:
                self.assertEqual(image.size, (4, 4))

    def test_record_seed_is_stable_and_identity_specific(self):
        first = qualitative.record_seed(2027, "FlatVelA", 17)
        self.assertEqual(first, qualitative.record_seed(2027, "FlatVelA", 17))
        self.assertNotEqual(first, qualitative.record_seed(2027, "FlatVelA", 18))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**31)

    def test_merge_isolated_method_outputs_requires_matching_record_targets(self):
        key = ("FlatVelA", 17)
        target = np.zeros((1, 1, 70, 70), dtype=np.float32)
        native_metrics = {
            "mae": 0.1,
            "rmse": 0.1,
            "ssim": 0.9,
            "mae_l": 0.1,
            "mae_h": 0.1,
        }
        worker_outputs = {
            "InversionNet": {
                "checkpoint": {"path": "inversion.ckpt", "missing_keys": 0, "unexpected_keys": 0},
                "records": {key: {"target": target, "prediction": target + 0.10, "native_metrics": native_metrics}},
            },
            "PD-BG-RFM": {
                "checkpoint": {"path": "pdr.ckpt", "missing_keys": 0, "unexpected_keys": 0},
                "records": {key: {"target": target.copy(), "prediction": target + 0.05, "native_metrics": native_metrics}},
            },
        }

        artifacts, checkpoints = qualitative.merge_isolated_method_outputs(
            keys=[key],
            worker_outputs=worker_outputs,
            normalization_profiles={"InversionNet": "seismic_global", "PD-BG-RFM": "openfwi"},
            base_seed=2027,
        )

        self.assertEqual(set(artifacts), {key})
        self.assertEqual(artifacts[key]["shape"], [1, 1, 70, 70])
        self.assertEqual(checkpoints["InversionNet"]["path"], "inversion.ckpt")
        broken_outputs = dict(worker_outputs)
        broken_outputs["PD-BG-RFM"] = dict(worker_outputs["PD-BG-RFM"])
        broken_outputs["PD-BG-RFM"]["records"] = {
            key: {"target": target + 1.0, "prediction": target, "native_metrics": native_metrics}
        }
        with self.assertRaisesRegex(ValueError, "Physical target mismatch"):
            qualitative.merge_isolated_method_outputs(
                keys=[key],
                worker_outputs=broken_outputs,
                normalization_profiles={"InversionNet": "seismic_global", "PD-BG-RFM": "openfwi"},
                base_seed=2027,
            )

    def test_isolated_worker_output_round_trip_preserves_identity_arrays_and_metrics(self):
        key = ("CurveVelB", 9)
        target = np.full((1, 1, 70, 70), 2100.0, dtype=np.float32)
        prediction = target + 25.0
        native_metrics = {
            "mae": 0.1,
            "rmse": 0.2,
            "ssim": 0.9,
            "mae_l": 0.08,
            "mae_h": 0.12,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.npz"
            qualitative.write_isolated_worker_output(
                path,
                method_name="InversionNet",
                checkpoint={"path": "model.ckpt", "missing_keys": 0, "unexpected_keys": 0},
                records={key: {"target": target, "prediction": prediction, "native_metrics": native_metrics}},
            )
            loaded = qualitative.read_isolated_worker_output(path)

        self.assertEqual(loaded["method_name"], "InversionNet")
        self.assertEqual(set(loaded["records"]), {key})
        self.assertTrue(np.array_equal(loaded["records"][key]["target"], target))
        self.assertTrue(np.array_equal(loaded["records"][key]["prediction"], prediction))
        self.assertEqual(loaded["records"][key]["native_metrics"], native_metrics)

    def test_isolated_worker_command_keeps_device_seed_and_explicit_artifact_paths(self):
        command = qualitative.build_isolated_worker_command(
            method_name="Auto-Linear",
            records_path=Path("/tmp/records.json"),
            output_path=Path("/tmp/auto_linear.npz"),
            device="cuda:0",
            base_seed=2027,
        )

        self.assertEqual(command[:3], [sys.executable, "-m", "bg_pdr_fm.evaluation.visualize_table2_qualitative"])
        self.assertIn("--isolated-worker", command)
        self.assertEqual(command[command.index("--worker-method") + 1], "Auto-Linear")
        self.assertEqual(command[command.index("--records-json") + 1], "/tmp/records.json")
        self.assertEqual(command[command.index("--worker-output") + 1], "/tmp/auto_linear.npz")
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(command[command.index("--base-seed") + 1], "2027")

    def test_rendered_margin_values_compare_pd_to_every_baseline(self):
        metrics = {
            "InversionNet": {"mae_l": 0.10, "mae_h": 0.20, "ssim": 0.80},
            "VelocityGAN": {"mae_l": 0.12, "mae_h": 0.15, "ssim": 0.91},
            "PD-BG-RFM": {"mae_l": 0.08, "mae_h": 0.10, "ssim": 0.93},
        }

        margins = qualitative.rendered_margin_values(metrics, "PD-BG-RFM")

        self.assertAlmostEqual(margins["margin_l"], 0.02)
        self.assertAlmostEqual(margins["margin_h"], 0.05)
        self.assertAlmostEqual(margins["margin_s"], 0.02)

    def test_rendered_case_figure_uses_shared_scales_and_writes_pdf_png(self):
        key = ("FlatVelA", 17)
        target = np.zeros((1, 70, 70), dtype=np.float32)
        predictions = {
            "InversionNet": np.full((1, 70, 70), 0.10, dtype=np.float32),
            "VelocityGAN": np.full((1, 70, 70), -0.20, dtype=np.float32),
            "UPFWI": np.full((1, 70, 70), 0.30, dtype=np.float32),
            "Auto-Linear": np.full((1, 70, 70), -0.40, dtype=np.float32),
            "Latent U-Net (Large)": np.full((1, 70, 70), 0.15, dtype=np.float32),
            "PD-BG-RFM": np.full((1, 70, 70), 0.05, dtype=np.float32),
        }
        cases = [{
            "dataset_name": key[0],
            "source_sample_index": key[1],
            "criterion": "margin_l",
            "margin_l": 0.01,
            "selection_group": "advantage",
            "status": "selected",
        }]
        metrics = {
            method_name: {"ssim": 0.90 + 0.01 * index}
            for index, method_name in enumerate(predictions)
        }
        artifacts = {key: {"target": target, "predictions": predictions, "metrics": metrics}}

        with tempfile.TemporaryDirectory() as directory:
            result = qualitative.render_case_figure(
                cases=cases,
                artifacts=artifacts,
                method_names=tuple(predictions),
                output_base=Path(directory) / "qualitative",
                title="criterion-selected advantage cases",
            )

            self.assertEqual(result["velocity_limits"], (-1.0, 1.0))
            self.assertEqual(result["error_limits"], (0.0, 0.4))
            self.assertEqual(result["velocity_cmap"], "jet")
            self.assertEqual(result["error_cmap"], "inferno")
            self.assertEqual(qualitative.TABLE2_ERROR_LIMITS, (0.0, 750.0))
            self.assertEqual(result["coordinate_system"], "x/z grid index")
            self.assertEqual(result["coordinate_ticks"], (0, 35, 69))
            self.assertTrue(Path(result["pdf"]).is_file())
            self.assertTrue(Path(result["png"]).is_file())
            self.assertGreater(Path(result["pdf"]).stat().st_size, 0)
            self.assertGreater(Path(result["png"]).stat().st_size, 0)

    def test_ssim_annotation_formats_finite_values_to_four_decimals(self):
        self.assertEqual(qualitative.format_ssim_annotation(0.98765), "SSIM=0.9877")
        with self.assertRaisesRegex(ValueError, "finite"):
            qualitative.format_ssim_annotation(float("nan"))


if __name__ == "__main__":
    unittest.main()
