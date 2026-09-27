from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from bg_pdr_fm.training import dispatch_bg_pdr_fm_parallel_upgrade_queue as queue


ROCM_SMI_SAMPLE = """
HCU[0]\t\t: HCU use (%): 0.0
HCU[1]\t\t: HCU use (%): 0.0
HCU[2]\t\t: HCU use (%): 0.0
HCU[3]\t\t: HCU use (%): 15.0
HCU[4]\t\t: HCU use (%): 0.0
HCU[5]\t\t: HCU use (%): 0.0
HCU[6]\t\t: HCU use (%): 0.0
HCU[7]\t\t: HCU use (%): 0.0
HCU[0]\t\t: vram Total Used Memory (MiB): 3
HCU[1]\t\t: vram Total Used Memory (MiB): 3
HCU[2]\t\t: vram Total Used Memory (MiB): 3
HCU[3]\t\t: vram Total Used Memory (MiB): 12058
HCU[4]\t\t: vram Total Used Memory (MiB): 3
HCU[5]\t\t: vram Total Used Memory (MiB): 3
HCU[6]\t\t: vram Total Used Memory (MiB): 3
HCU[7]\t\t: vram Total Used Memory (MiB): 5042
"""


def test_rocm_smi_parser_and_stable_group_allocation():
    stats = queue.parse_rocm_smi_text(ROCM_SMI_SAMPLE)
    idle = queue.idle_gpus_from_stats(
        stats,
        allowed_gpus=[7, 6, 5, 4, 3, 2, 1, 0],
        max_util=5.0,
        max_mem_mib=1024.0,
    )
    groups = queue.allocate_gpu_groups(idle, group_size=3, max_groups=2)

    assert idle == [6, 5, 4, 2, 1, 0]
    assert groups == [(6, 5, 4), (2, 1, 0)]


def test_default_jobs_keep_stage1_and_diffusers_branches_sequenced():
    jobs = {job.name: job for job in queue.DEFAULT_JOBS}

    assert jobs["background_h128"].base_config.name == "openfwi_lmdb_background_unet_direct_h128_l1l2_e100.yaml"
    assert jobs["background_h192"].depends_on == "background_h128"
    assert jobs["diffusers_crossattn_medium"].base_config.name.endswith("medium_e100.yaml")
    assert jobs["diffusers_crossattn_large"].depends_on == "diffusers_crossattn_medium"
    assert queue.DEFAULT_BATCH_CANDIDATES == (256, 512, 768, 1000)


def test_prepare_config_overrides_training_paths_without_mutating_base(tmp_path):
    base = OmegaConf.create(
        {
            "training": {
                "batch_size": 100,
                "devices": 4,
                "max_epochs": 100,
                "stage_checkpoint_path": "logs/original/stage_checkpoints",
                "logging": {"log_dir": "logs/original/lightning", "log_version": "original"},
                "checkpoint": {
                    "dirpath": "logs/original/lightning/checkpoints",
                    "filename": "original-epoch_{epoch}",
                },
            },
            "diagnostics": {"output_dir": "logs/original/diagnostics"},
            "evaluation": {"output_dir": "logs/original/evaluation"},
        }
    )

    prepared = queue.prepare_training_config(
        base,
        job_name="background_h128",
        run_dir=tmp_path / "run",
        batch_size=512,
        devices=3,
        max_epochs=1,
        limit_train_batches=20,
        limit_val_batches=1,
    )

    assert OmegaConf.select(base, "training.batch_size") == 100
    assert OmegaConf.select(prepared, "training.batch_size") == 512
    assert OmegaConf.select(prepared, "training.devices") == 3
    assert OmegaConf.select(prepared, "training.limit_train_batches") == 20
    assert OmegaConf.select(prepared, "training.stage_checkpoint_path") == str(tmp_path / "run" / "stage_checkpoints")
    assert OmegaConf.select(prepared, "training.checkpoint.dirpath") == str(tmp_path / "run" / "lightning" / "checkpoints")


def test_dependent_job_is_not_ready_after_dependency_failure():
    job = queue.UpgradeJob(
        name="diffusers_crossattn_large",
        base_config=Path("large.yaml"),
        run_dir=Path("large"),
        checkpoint_name="residual_last.ckpt",
        depends_on="diffusers_crossattn_medium",
    )

    assert not queue.dependency_ready(
        job,
        completed={},
        skipped={},
        failed={"diffusers_crossattn_medium": {"reason": "failed"}},
    )


def test_ramp_requires_at_least_one_success():
    results = [{"batch_size": 256, "status": "failed"}]

    assert queue.best_successful_batch(results) is None


def test_skip_failed_dependency_jobs_removes_blocked_pending_job():
    blocked = queue.UpgradeJob(
        name="background_h192",
        base_config=Path("h192.yaml"),
        run_dir=Path("h192"),
        checkpoint_name="background_last.ckpt",
        depends_on="background_h128",
    )
    pending = [blocked]
    skipped: dict[str, dict[str, object]] = {}

    queue.skip_failed_dependency_jobs(
        pending,
        skipped=skipped,
        failed={"background_h128": {"reason": "batch_ramp_failed"}},
    )

    assert pending == []
    assert skipped["background_h192"]["reason"] == "dependency_failed"


def test_worker_command_preserves_gpu_group_and_job_name(tmp_path):
    job = queue.UpgradeJob(
        name="background_h128",
        base_config=Path("h128.yaml"),
        run_dir=tmp_path / "run",
        checkpoint_name="background_last.ckpt",
    )

    command = queue.build_worker_command(
        job,
        python="/python",
        queue_dir=tmp_path / "queue",
        gpus=(6, 5, 4),
        batch_candidates=(256, 512),
        ramp_steps=20,
    )

    assert command[:3] == ["/python", "-m", "bg_pdr_fm.training.dispatch_bg_pdr_fm_parallel_upgrade_queue"]
    assert "--worker-job" in command
    assert "background_h128" in command
    assert "--worker-gpus" in command
    assert "6,5,4" in command
