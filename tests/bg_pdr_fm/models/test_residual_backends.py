import unittest

import torch

from bg_pdr_fm.models.generators import ResidualFlowGenerator
from bg_pdr_fm.models.residual_backends import build_residual_backend


class DirectPredictorBackendTest(unittest.TestCase):
    def test_training_loss_predicts_latent_residual_shape(self):
        torch.manual_seed(7)
        backend = build_residual_backend(
            backend="direct_predictor",
            latent_channels=4,
            cond_channels=8,
            latent_hw=(6, 6),
            hidden_channels=16,
        )
        z_res = torch.randn(2, 4, 6, 6)
        cond = torch.randn(2, 8, 6, 6)

        output = backend.training_loss(z_res, cond)

        self.assertEqual(tuple(output["z_res_pred"].shape), tuple(z_res.shape))
        self.assertEqual(tuple(output["velocity_pred"].shape), tuple(z_res.shape))
        self.assertTrue(torch.isfinite(output["loss"]).item())
        self.assertGreater(float(output["loss"]), 0.0)

    def test_sample_is_deterministic_and_ignores_steps(self):
        torch.manual_seed(11)
        backend = build_residual_backend(
            backend="direct_predictor",
            latent_channels=4,
            cond_channels=8,
            latent_hw=(6, 6),
            hidden_channels=16,
        )
        cond = torch.randn(2, 8, 6, 6)

        sample_a = backend.sample(cond, x_size=(2, 4, 6, 6), steps=1)
        sample_b = backend.sample(cond, x_size=(2, 4, 6, 6), steps=100)

        self.assertTrue(torch.equal(sample_a, sample_b))
        self.assertEqual(tuple(sample_a.shape), (2, 4, 6, 6))


class UNetFMFiLMBackendTest(unittest.TestCase):
    def test_training_loss_predicts_latent_residual_shape(self):
        torch.manual_seed(17)
        backend = build_residual_backend(
            backend="unet_fm_film",
            latent_channels=4,
            cond_channels=8,
            latent_hw=(8, 8),
            hidden_channels=16,
        )
        z_res = torch.randn(2, 4, 8, 8)
        cond = torch.randn(2, 8, 8, 8)

        output = backend.training_loss(z_res, cond)

        self.assertEqual(tuple(output["z_res_pred"].shape), tuple(z_res.shape))
        self.assertEqual(tuple(output["velocity_pred"].shape), tuple(z_res.shape))
        self.assertEqual(tuple(output["t"].shape), (2,))
        self.assertTrue(torch.isfinite(output["loss"]).item())
        self.assertGreater(float(output["loss"]), 0.0)

    def test_sample_returns_requested_latent_shape(self):
        torch.manual_seed(19)
        backend = build_residual_backend(
            backend="unet_fm_film",
            latent_channels=4,
            cond_channels=8,
            latent_hw=(8, 8),
            hidden_channels=16,
        )
        cond = torch.randn(2, 8, 8, 8)

        sample = backend.sample(cond, x_size=(2, 4, 8, 8), steps=3)

        self.assertEqual(tuple(sample.shape), (2, 4, 8, 8))
        self.assertTrue(torch.isfinite(sample).all().item())


class ResidualFlowGeneratorConditionMergeTest(unittest.TestCase):
    def test_concat_condition_merge_doubles_backend_condition_channels(self):
        generator = ResidualFlowGenerator(
            cond_channels=8,
            latent_channels=4,
            bg_cond_channels=3,
            latent_hw=(8, 8),
            backend="direct_predictor",
            backend_hidden_channels=16,
            condition_merge="concat",
        )
        cond_s = torch.randn(2, 8, 8, 8)
        bg_hat = torch.randn(2, 1, 70, 70)

        merged = generator.merged_condition(cond_s, bg_hat)

        self.assertEqual(tuple(merged.shape), (2, 16, 8, 8))
        self.assertEqual(generator.backend.field.net[0].in_channels, 20)


if __name__ == "__main__":
    unittest.main()
