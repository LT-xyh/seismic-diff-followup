from __future__ import annotations

import os
import tempfile

from bg_pdr_fm import runtime


def test_configure_checkpoint_temp_dir_routes_temp_and_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("BG_PDR_FM_CHECKPOINT_TMPDIR", raising=False)

    expected = runtime.resolve_checkpoint_temp_dir(tmp_path / "scratch")
    value = runtime.configure_checkpoint_temp_dir(tmp_path / "scratch")

    assert value == str(expected)
    assert tempfile.gettempdir() == value
    assert os.environ["TMPDIR"] == value
    assert os.environ["TMP"] == value
    assert os.environ["TEMP"] == value
    assert os.environ["BG_PDR_FM_CHECKPOINT_TMPDIR"] == value
    assert os.environ["XDG_CACHE_HOME"].startswith(value)
    assert os.environ["TORCH_HOME"].startswith(value)

    with tempfile.NamedTemporaryFile(dir=value) as handle:
        handle.write(b"checkpoint smoke")
        handle.flush()
        assert handle.name.startswith(value)


def test_worker_environment_isolated_to_requested_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/tmp")
    expected = runtime.resolve_checkpoint_temp_dir(tmp_path / "worker-scratch")
    env = runtime.configure_checkpoint_worker_environment(
        {"CUDA_VISIBLE_DEVICES": "7"}, tmp_path / "worker-scratch"
    )

    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert env["TMPDIR"] == str(expected)
    assert env["TMP"] == str(expected)
    assert env["TEMP"] == str(expected)
    assert env["MPLCONFIGDIR"].startswith(str(expected))
    assert (expected / "cache").is_dir()


def test_long_scratch_path_is_shortened_for_multiprocessing_socket():
    long_path = "/public/home/xuyinghao/workspace/seismic-diff/logs/" + ("nested/" * 16) + "worker/tmp"

    resolved = runtime.resolve_checkpoint_temp_dir(long_path)

    assert len(str(resolved)) < 80
    assert str(resolved).startswith(str(runtime.default_checkpoint_temp_dir()))
