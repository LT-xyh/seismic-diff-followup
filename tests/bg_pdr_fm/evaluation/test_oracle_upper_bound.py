import math
import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "bg_pdr_fm"
    / "evaluation"
    / "run_oracle_upper_bound.py"
)
SPEC = importlib.util.spec_from_file_location("run_oracle_upper_bound", MODULE_PATH)
oracle = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(oracle)


class OracleUpperBoundHelpersTest(unittest.TestCase):
    def test_mean_dict_ignores_nan_values(self):
        rows = [
            {"ssim": 0.9, "mae_l": 0.1},
            {"ssim": float("nan"), "mae_l": 0.2},
            {"ssim": 0.8, "mae_l": float("nan")},
        ]

        means = oracle.mean_dict(rows, ["ssim", "mae_l"])

        self.assertAlmostEqual(means["ssim"], 0.85)
        self.assertAlmostEqual(means["mae_l"], 0.15)

    def test_mean_dict_returns_nan_for_empty_metric(self):
        means = oracle.mean_dict([{"ssim": float("nan")}], ["ssim"])

        self.assertTrue(math.isnan(means["ssim"]))

    def test_build_diagnosis_identifies_stage1_background_bottleneck(self):
        diagnosis = oracle.build_diagnosis(
            {
                "baseline": {"ssim": 0.878, "mae_l": 0.058},
                "oracle_background_current_residual": {"ssim": 0.936, "mae_l": 0.024},
                "oracle_latent_residual": {"ssim": 0.972, "mae_l": 0.010},
                "oracle_background_oracle_latent": {"ssim": 0.973, "mae_l": 0.009},
            }
        )

        self.assertEqual(diagnosis["primary_bottleneck"], "stage1_background")
        self.assertIn("Stage 1", diagnosis["recommendation"])

    def test_build_diagnosis_identifies_latent_ceiling(self):
        diagnosis = oracle.build_diagnosis(
            {
                "baseline": {"ssim": 0.878, "mae_l": 0.058},
                "oracle_background_current_residual": {"ssim": 0.889, "mae_l": 0.053},
                "oracle_latent_residual": {"ssim": 0.918, "mae_l": 0.030},
                "oracle_background_oracle_latent": {"ssim": 0.921, "mae_l": 0.029},
            }
        )

        self.assertEqual(diagnosis["primary_bottleneck"], "latent_codec")
        self.assertIn("VAE", diagnosis["recommendation"])

    def test_apply_cli_overrides_updates_output_dir_and_max_batches(self):
        conf = {}

        oracle.apply_cli_overrides(
            fake_update,
            conf,
            output_dir="/tmp/oracle",
            max_batches=1,
        )

        self.assertEqual(conf["evaluation.output_dir"], "/tmp/oracle")
        self.assertEqual(conf["evaluation.max_batches"], 1)

    def test_batch_seed_offsets_base_seed_by_batch_index(self):
        self.assertEqual(oracle.batch_seed(1234, 0), 1234)
        self.assertEqual(oracle.batch_seed(1234, 7), 1241)
        self.assertIsNone(oracle.batch_seed(None, 7))

    def test_rows_for_path_filters_to_single_path(self):
        rows = [
            {"path": "baseline", "ssim": 0.8},
            {"path": "oracle_latent_residual", "ssim": 0.9},
            {"path": "baseline", "ssim": 0.7},
        ]

        filtered = oracle.rows_for_path(rows, "baseline")

        self.assertEqual(filtered, [{"path": "baseline", "ssim": 0.8}, {"path": "baseline", "ssim": 0.7}])


def fake_update(conf, path, value, merge=False):
    del merge
    conf[path] = value


if __name__ == "__main__":
    unittest.main()
