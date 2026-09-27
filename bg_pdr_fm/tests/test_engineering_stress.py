from __future__ import annotations

import torch
from omegaconf import OmegaConf

from bg_pdr_fm.data import BGSampleBatch
from bg_pdr_fm.evaluation.engineering_stress import (
    EngineeringStressScenario,
    apply_engineering_stress,
    get_engineering_stress_scenario,
)
from bg_pdr_fm.evaluation.run_engineering_stress_benchmark import (
    ENGINEERING_JOBS,
    _selected_jobs,
    apply_openfwi_engineering_overrides,
)


def _batch(batch_size: int = 4) -> BGSampleBatch:
    depth = torch.arange(batch_size * 70 * 70, dtype=torch.float32).view(batch_size, 1, 70, 70) / 1000.0
    migrated = torch.ones(batch_size, 1, 1000, 70, dtype=torch.float32)
    rms = torch.full((batch_size, 1, 1000, 70), 2.0, dtype=torch.float32)
    horizon = torch.full((batch_size, 1, 70, 70), 3.0, dtype=torch.float32)
    well_log = torch.full((batch_size, 1, 70, 70), 4.0, dtype=torch.float32)
    well_mask = torch.ones(batch_size, 1, 70, 70, dtype=torch.float32)
    return BGSampleBatch(
        depth_vel=depth,
        migrated_image=migrated,
        horizon=horizon,
        rms_vel=rms,
        well_log=well_log,
        well_mask=well_mask,
        modality_mask=torch.ones(batch_size, 4, dtype=torch.float32),
        modality_quality=torch.ones(batch_size, 4, dtype=torch.float32),
        metadata={"sample_index": torch.arange(batch_size, dtype=torch.float32).view(batch_size, 1)},
    )


def test_engineering_stress_is_deterministic_for_same_seed_and_batch_index():
    scenario = EngineeringStressScenario(
        name="test",
        pstm_snr_db=20.0,
        rms_scale_max=0.02,
        rms_smooth_noise_std_frac=0.01,
        rms_lowfreq_bias_std_frac=0.01,
        horizon_missing_prob=0.25,
        well_missing_prob=0.5,
        seed=123,
    )

    a = apply_engineering_stress(_batch(), scenario, batch_index=7)
    b = apply_engineering_stress(_batch(), scenario, batch_index=7)

    assert torch.equal(a.migrated_image, b.migrated_image)
    assert torch.equal(a.rms_vel, b.rms_vel)
    assert torch.equal(a.horizon, b.horizon)
    assert torch.equal(a.well_log, b.well_log)
    assert torch.equal(a.modality_mask, b.modality_mask)
    assert torch.equal(a.depth_vel, b.depth_vel)


def test_pstm_snr20_adds_finite_noise_without_changing_target():
    scenario = EngineeringStressScenario(
        name="pstm_only",
        pstm_snr_db=20.0,
        rms_scale_max=0.0,
        rms_smooth_noise_std_frac=0.0,
        rms_lowfreq_bias_std_frac=0.0,
        horizon_missing_prob=0.0,
        well_missing_prob=0.0,
        seed=5,
    )
    batch = _batch()

    stressed = apply_engineering_stress(batch, scenario, batch_index=0)

    assert torch.isfinite(stressed.migrated_image).all()
    assert not torch.equal(stressed.migrated_image, batch.migrated_image)
    assert torch.equal(stressed.depth_vel, batch.depth_vel)
    assert torch.equal(stressed.horizon, batch.horizon)
    assert torch.equal(stressed.well_log, batch.well_log)


def test_horizon_and_well_missing_update_inputs_masks_and_quality():
    scenario = EngineeringStressScenario(
        name="all_missing",
        pstm_snr_db=None,
        rms_scale_max=0.0,
        rms_smooth_noise_std_frac=0.0,
        rms_lowfreq_bias_std_frac=0.0,
        horizon_missing_prob=1.0,
        well_missing_prob=1.0,
        seed=9,
    )

    stressed = apply_engineering_stress(_batch(), scenario, batch_index=0)

    assert torch.count_nonzero(stressed.horizon).item() == 0
    assert torch.count_nonzero(stressed.well_log).item() == 0
    assert torch.count_nonzero(stressed.well_mask).item() == 0
    assert torch.all(stressed.modality_mask[:, 1] == 0)
    assert torch.all(stressed.modality_mask[:, 3] == 0)
    assert torch.all(stressed.modality_quality[:, 1] == 0)
    assert torch.all(stressed.modality_quality[:, 3] == 0)
    assert torch.all(stressed.modality_mask[:, [0, 2]] == 1)


def test_zero_missing_prob_keeps_horizon_and_well_channels():
    scenario = EngineeringStressScenario(
        name="no_missing",
        pstm_snr_db=None,
        rms_scale_max=0.0,
        rms_smooth_noise_std_frac=0.0,
        rms_lowfreq_bias_std_frac=0.0,
        horizon_missing_prob=0.0,
        well_missing_prob=0.0,
        seed=11,
    )
    batch = _batch()

    stressed = apply_engineering_stress(batch, scenario, batch_index=0)

    assert torch.equal(stressed.horizon, batch.horizon)
    assert torch.equal(stressed.well_log, batch.well_log)
    assert torch.equal(stressed.well_mask, batch.well_mask)
    assert torch.equal(stressed.modality_mask, batch.modality_mask)
    assert torch.equal(stressed.modality_quality, batch.modality_quality)


def test_apply_openfwi_engineering_overrides_enables_full_mode_stress():
    conf = OmegaConf.create(
        {
            "data": {"name": "placeholder"},
            "evaluation": {"missing_modes": ["w/o horizon"], "output_dir": "old", "checkpoint": "old.ckpt"},
            "training": {"devices": 8, "num_workers": 0},
        }
    )

    updated = apply_openfwi_engineering_overrides(
        conf,
        checkpoint=None,
        output_dir="out/job",
        openfwi_root="/data/openfwi",
        lmdb_root="/data/openfwi_lmdb",
        max_batches=2,
        num_workers=3,
        scenario="pstm20_rms_mild_h025_w050",
        seed=2027,
    )

    assert updated.data.name == "openfwi"
    assert updated.data.storage_backend == "lmdb"
    assert list(updated.evaluation.missing_modes) == ["full"]
    assert updated.evaluation.output_dir == "out/job"
    assert updated.evaluation.max_batches == 2
    assert updated.evaluation.engineering_stress.enabled is True
    assert updated.evaluation.engineering_stress.scenario == "pstm20_rms_mild_h025_w050"
    assert updated.evaluation.engineering_stress.seed == 2027
    assert updated.training.devices == 1
    assert updated.training.num_workers == 3


def test_moderate_engineering_scenario_matches_protocol_values():
    scenario = get_engineering_stress_scenario("pstm15_rms_moderate_h050_w075", seed=3031)

    assert scenario.name == "pstm15_rms_moderate_h050_w075"
    assert scenario.pstm_snr_db == 15.0
    assert scenario.rms_scale_max == 0.05
    assert scenario.rms_smooth_noise_std_frac == 0.025
    assert scenario.rms_lowfreq_bias_std_frac == 0.025
    assert scenario.horizon_missing_prob == 0.50
    assert scenario.well_missing_prob == 0.75
    assert scenario.seed == 3031


def test_engineering_runner_job_registry_includes_completed_table_models():
    names = {job.name for job in ENGINEERING_JOBS}

    assert {
        "smooth_dix",
        "mm_invnet",
        "sv_inv_net",
        "velocity_gan",
        "adapted_gfi",
        "adapted_auto_linear",
    }.issubset(names)


def test_engineering_runner_selects_all_jobs_by_default():
    assert _selected_jobs(None) == list(ENGINEERING_JOBS)
