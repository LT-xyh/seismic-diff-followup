import torch

from bg_pdr_fm.evaluation.visualize_residual_flow_trajectory import select_requested_record
from bg_pdr_fm.models.generators import ResidualFlowGenerator
from bg_pdr_fm.models.residual_backends import UNetFiLMResidualFMBackend


def test_unet_film_trace_replays_sample_with_fixed_noise():
    backend = UNetFiLMResidualFMBackend(latent_channels=1, cond_channels=4, hidden_channels=8)
    backend.eval()
    condition = torch.randn(1, 4, 8, 8)
    noise = torch.randn(1, 1, 8, 8)

    trace = backend.sample_with_trace(condition, x_size=tuple(noise.shape), steps=5, noise=noise)
    sampled = backend.sample(condition, x_size=tuple(noise.shape), steps=5, noise=noise)

    assert set(trace) >= {"states", "velocities", "deltas", "times"}
    assert trace["states"].shape == (6, 1, 1, 8, 8)
    assert trace["velocities"].shape == (5, 1, 1, 8, 8)
    assert trace["deltas"].shape == (5, 1, 1, 8, 8)
    assert trace["times"].shape == (5, 1)
    torch.testing.assert_close(trace["states"][0], noise)
    torch.testing.assert_close(trace["states"][-1], sampled)
    torch.testing.assert_close(trace["deltas"], trace["velocities"] / 5.0)
    torch.testing.assert_close(trace["states"][1:], trace["states"][:-1] + trace["deltas"])
    torch.testing.assert_close(trace["times"].flatten(), torch.arange(5, dtype=noise.dtype) / 5.0)


def test_residual_generator_trace_keeps_concat_condition_and_final_state():
    generator = ResidualFlowGenerator(
        cond_channels=2,
        latent_channels=1,
        latent_hw=(8, 8),
        backend="unet_fm_film",
        backend_hidden_channels=8,
        condition_merge="concat",
        background_context_source="codec_latent_raw",
    )
    generator.eval()
    structural = torch.randn(1, 2, 8, 8)
    background = torch.randn(1, 1, 8, 8)
    noise = torch.randn(1, 1, 8, 8)

    trace = generator.sample_with_trace(
        structural,
        background,
        x_size=tuple(noise.shape),
        steps=4,
        z_bg=background,
        noise=noise,
    )
    sampled = generator.sample(
        structural,
        background,
        x_size=tuple(noise.shape),
        steps=4,
        z_bg=background,
        noise=noise,
    )

    assert trace["condition"].shape == (1, 3, 8, 8)
    torch.testing.assert_close(trace["condition"], torch.cat([structural, background], dim=1))
    torch.testing.assert_close(trace["states"][-1], sampled)


def test_default_sample_retains_the_original_euler_update():
    backend = UNetFiLMResidualFMBackend(latent_channels=1, cond_channels=2, hidden_channels=8)
    backend.eval()
    condition = torch.randn(1, 2, 8, 8)
    steps = 3

    torch.manual_seed(3407)
    actual = backend.sample(condition, x_size=(1, 1, 8, 8), steps=steps)
    torch.manual_seed(3407)
    expected = torch.randn(1, 1, 8, 8)
    dt = 1.0 / steps
    for idx in range(steps):
        time = expected.new_full((expected.shape[0],), float(idx) / float(steps))
        expected = expected + dt * backend.field(expected, time, condition)

    torch.testing.assert_close(actual, expected)


def test_requested_non_heldout_asset_is_marked_without_official_metrics(tmp_path):
    manifest = tmp_path / "held_out_manifest.csv"
    metrics = tmp_path / "merged_metrics.csv"
    manifest.write_text(
        "dataset_id,dataset_name,source_sample_index,worker\n"
        "5,FlatFaultB,123,worker_00\n",
        encoding="utf-8",
    )
    metrics.write_text(
        "dataset_id,dataset_name,source_sample_index,ssim\n"
        "5,FlatFaultB,123,0.9\n",
        encoding="utf-8",
    )

    selection = select_requested_record(
        manifest,
        metrics,
        dataset_name="FlatFaultB",
        source_sample_index=27006,
        dataset_id=5,
    )

    assert selection["selected_manifest"]["source_sample_index"] == "27006"
    assert selection["provenance_scope"] == "not_in_official_global_heldout"
    assert selection["dataset_split"] == "all"
    assert selection["selected_metrics"]["ssim"] == ""
    assert selection["median_ssim"] is None
