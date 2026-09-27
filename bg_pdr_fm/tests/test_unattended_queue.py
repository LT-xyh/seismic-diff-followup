from __future__ import annotations

from pathlib import Path

from bg_pdr_fm.training.unattended_3day_queue import (
    QueueJob,
    build_plan,
    format_summary_markdown,
    next_pixel_followup,
    next_auto_linear_followup,
)
from bg_pdr_fm.training import dispatch_aaai27_baseline_gpu_queue as baseline_queue


def test_build_plan_preserves_current_jobs_and_next_steps():
    pixel_job = QueueJob(
        name="pixelfm_e100",
        run_dir=Path("logs/bg_pdr_fm/residual_train_smooth_hc64_nogate_pixelfm_film_predbg_e100"),
        train_pid=247860,
        kind="residual",
        train_config=Path("bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_pixelfm_film_predbg_e100.yaml"),
        eval_config=Path("bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_pixelfm_film_predbg_e100_eval_mb10.yaml"),
        eval_gpu="7",
        next_train_config=None,
    )
    auto_job = QueueJob(
        name="adapted_auto_linear_ae_pretrain",
        run_dir=Path("logs/bg_pdr_fm/aaai27/formal/adapted_auto_linear_ae_pretrain"),
        train_pid=1197522,
        kind="benchmark",
        train_config=Path("bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_ae_pretrain.yaml"),
        eval_config=Path("bg_pdr_fm/configs/experiments/aaai27/formal_adapted_auto_linear_multimodal.yaml"),
        eval_gpu="6",
        next_train_config=Path("bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_multimodal.yaml"),
    )

    plan = build_plan([pixel_job, auto_job])

    assert plan["jobs"][0]["name"] == "pixelfm_e100"
    assert plan["jobs"][0]["current"]["train_pid"] == 247860
    assert plan["jobs"][0]["current"]["status"] in {"running", "finished_or_unknown"}
    assert plan["jobs"][0]["next"]["train_config"] is None
    assert plan["jobs"][1]["next"]["eval_config"].name == "formal_adapted_auto_linear_multimodal.yaml"


def test_followup_selection_is_method_specific():
    pixel_next = next_pixel_followup(
        Path("logs/bg_pdr_fm/residual_train_smooth_hc64_nogate_pixelfm_film_predbg_e100"),
        ssim=0.905,
        mae_l=0.044,
    )
    auto_next = next_auto_linear_followup(
        Path("logs/bg_pdr_fm/aaai27/formal/adapted_auto_linear_ae_pretrain"),
    )

    assert pixel_next["action"] == "stop_no_ae_route"
    assert pixel_next["train_config"] is None
    assert auto_next["action"] == "launch_inverse_training"
    assert auto_next["train_config"].name == "adapted_auto_linear_multimodal.yaml"


def test_summary_markdown_mentions_gpu_assignment_and_status():
    markdown = format_summary_markdown(
        {
            "jobs": [
                {
                    "name": "pixelfm_e100",
                    "current": {"status": "running", "gpu_group": "7,0,1,2"},
                    "next": {"action": "mb10_eval"},
                }
            ]
        }
    )

    assert "pixelfm_e100" in markdown
    assert "7,0,1,2" in markdown
    assert "mb10_eval" in markdown


def test_baseline_queue_tracks_live_job_even_when_checkpoint_exists(monkeypatch, tmp_path):
    job = baseline_queue.QueueJob(
        name="velocity_gan",
        config=tmp_path / "velocity_gan.yaml",
        run_dir=tmp_path / "velocity_gan",
    )

    monkeypatch.setattr(baseline_queue, "JOBS", (job,))
    monkeypatch.setattr(baseline_queue, "_validate_config", lambda item: {"variant": item.name, "batch_size": 1, "devices": 1})
    monkeypatch.setattr(baseline_queue, "_existing_live_pid", lambda item: 12345)
    monkeypatch.setattr(baseline_queue, "_run_finished", lambda item: True)
    monkeypatch.setattr(baseline_queue, "_pid_alive", lambda pid: True)

    state = baseline_queue.run_queue(
        queue_dir=tmp_path / "queue",
        allowed_gpus=[7],
        python="/bin/false",
        poll_seconds=1,
        max_util=5.0,
        max_mem_mib=1024.0,
        max_concurrent=8,
        mail_to="",
        smtp_config=None,
        dry_run=True,
    )

    assert "velocity_gan" in state["running"]
    assert "velocity_gan" not in state["skipped"]


def test_baseline_queue_can_filter_jobs(monkeypatch, tmp_path):
    first = baseline_queue.QueueJob(
        name="first",
        config=tmp_path / "first.yaml",
        run_dir=tmp_path / "first",
    )
    second = baseline_queue.QueueJob(
        name="second",
        config=tmp_path / "second.yaml",
        run_dir=tmp_path / "second",
    )

    monkeypatch.setattr(baseline_queue, "JOBS", (first, second))
    monkeypatch.setattr(baseline_queue, "_validate_config", lambda item: {"variant": item.name, "batch_size": 1, "devices": 1})
    monkeypatch.setattr(baseline_queue, "_existing_live_pid", lambda item: None)
    monkeypatch.setattr(baseline_queue, "_run_finished", lambda item: False)

    state = baseline_queue.run_queue(
        queue_dir=tmp_path / "queue",
        allowed_gpus=[7],
        python="/bin/false",
        poll_seconds=1,
        max_util=5.0,
        max_mem_mib=1024.0,
        max_concurrent=8,
        mail_to="",
        smtp_config=None,
        dry_run=True,
        selected_jobs={"second"},
    )

    assert state["pending"] == ["second"]
    assert "first" not in state["validated"]


def test_baseline_queue_includes_mae_auto_linear_jobs():
    jobs = {job.name: job for job in baseline_queue.JOBS}

    assert jobs["adapted_auto_linear_mae_ae_pretrain"].config.name == "adapted_auto_linear_mae_ae_pretrain.yaml"
    assert jobs["adapted_auto_linear_mae_ae_pretrain"].run_dir.name == "adapted_auto_linear_mae_ae_pretrain"
    assert jobs["adapted_auto_linear_mae_multimodal"].config.name == "adapted_auto_linear_mae_multimodal.yaml"
    assert jobs["adapted_auto_linear_mae_multimodal"].run_dir.name == "adapted_auto_linear_mae_multimodal"
    assert jobs["adapted_auto_linear_mae_multimodal"].depends_on == "adapted_auto_linear_mae_ae_pretrain"


def test_baseline_queue_does_not_run_dependent_job_after_dependency_failure():
    job = baseline_queue.QueueJob(
        name="inverse",
        config=Path("inverse.yaml"),
        run_dir=Path("inverse"),
        depends_on="ae",
    )

    assert not baseline_queue._ready(job, completed={}, skipped={}, failed={"ae": {"reason": "failed"}})


def test_baseline_queue_selects_by_free_vram_even_when_gpu_is_busy():
    stats = {
        4: {"util": 95.0, "total_mib": 65_520.0, "used_mib": 33_000.0},
        5: {"util": 0.0, "total_mib": 65_520.0, "used_mib": 50_000.0},
    }

    selected = baseline_queue._gpus_with_free_vram(
        stats,
        allowed=[4, 5],
        reserved=set(),
        required_free_mib=30_000.0,
    )

    assert selected == [4]


def test_baseline_queue_prefers_more_free_vram_then_higher_gpu_id():
    stats = {
        4: {"util": 0.0, "total_mib": 65_520.0, "used_mib": 20_000.0},
        6: {"util": 100.0, "total_mib": 65_520.0, "used_mib": 15_000.0},
        7: {"util": 100.0, "total_mib": 65_520.0, "used_mib": 15_000.0},
    }

    selected = baseline_queue._gpus_with_free_vram(
        stats,
        allowed=[4, 6, 7],
        reserved=set(),
        required_free_mib=20_000.0,
    )

    assert selected == [7, 6, 4]


def test_memory_requirement_adds_fractional_and_fixed_headroom():
    assert baseline_queue.required_free_mib(16_000.0) == 22_048


def test_official_baseline_jobs_use_isolated_training_and_eval_artifacts():
    jobs = {job.name: job for job in baseline_queue.JOBS}
    expected = {
        "adapted_inversion_net": "adapted_inversion_net_official_e100",
        "velocity_gan": "velocity_gan_official_e100",
        "adapted_upfwi": "adapted_upfwi_official_e100",
    }

    for name, run_name in expected.items():
        job = jobs[name]
        assert job.run_dir.name == run_name
        assert job.eval_config is not None
        assert job.eval_dir is not None
        assert "official_e100" in job.eval_dir.name


def test_official_pipeline_job_requires_checkpoint_and_evaluation(tmp_path):
    run_dir = tmp_path / "formal" / "adapted_upfwi_official_e100"
    eval_dir = tmp_path / "eval" / "adapted_upfwi_official_e100"
    job = baseline_queue.QueueJob(
        name="adapted_upfwi",
        config=tmp_path / "adapted_upfwi.yaml",
        run_dir=run_dir,
        eval_config=tmp_path / "formal_adapted_upfwi.yaml",
        eval_dir=eval_dir,
    )
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").touch()

    assert not baseline_queue._run_finished(job)
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text("{}", encoding="utf-8")
    assert baseline_queue._run_finished(job)
