"""Runtime helpers shared by BG-PDR-FM entry points."""

from __future__ import annotations

import os
from pathlib import Path
import hashlib
import re
import shutil
import tempfile
from typing import Any
import warnings

import torch


def _public_home() -> Path | None:
    """Return the matching home directory on the large public mount."""
    home = Path.home().resolve()
    public_root = Path("/public")
    try:
        relative_home = home.relative_to(public_root)
    except ValueError:
        relative_home = home.relative_to("/")
    candidate = public_root / relative_home
    if candidate.parent.is_dir() or candidate.is_dir():
        return candidate
    return None


def default_checkpoint_temp_dir() -> Path:
    """Choose a scratch directory that is not the small root filesystem."""
    public_home = _public_home()
    if public_home is not None:
        return public_home / "tmp"
    return Path.home() / "tmp"


def short_checkpoint_temp_dir(identifier: str | os.PathLike[str]) -> Path:
    """Return a short, deterministic scratch path for a long run directory.

    ``multiprocessing.resource_sharer`` places a Unix socket below the
    process temporary directory. Linux limits that socket path to roughly 108
    bytes, so using a deeply nested experiment directory as ``TMPDIR`` is
    unsafe even when the directory is on the correct filesystem.
    """
    text = str(identifier)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(text).name or "run")[:24].strip("-") or "run"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return default_checkpoint_temp_dir() / "bg_pdr_fm_runs" / f"{label}-{digest}"


def _existing_device(path: Path) -> int | None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        return os.stat(candidate).st_dev
    except OSError:
        return None


def resolve_checkpoint_temp_dir(preferred: str | os.PathLike[str] | None = None) -> Path:
    """Resolve and validate the checkpoint scratch directory.

    An explicit path is honored unless it resolves to the root filesystem while
    a matching writable path exists on ``/public``. This prevents an old YAML
    or shell environment containing ``/tmp`` from silently reintroducing the
    failure mode this helper is intended to prevent.
    """
    requested = preferred or os.environ.get("BG_PDR_FM_CHECKPOINT_TMPDIR")
    path = Path(requested).expanduser() if requested else default_checkpoint_temp_dir()
    if len(str(path)) > 80:
        warnings.warn(
            f"Shortening checkpoint scratch path {path} for Unix socket compatibility.",
            RuntimeWarning,
            stacklevel=2,
        )
        path = short_checkpoint_temp_dir(path)
    public_default = default_checkpoint_temp_dir()
    if not public_default.exists():
        public_default.mkdir(parents=True, exist_ok=True)
    try:
        root_device = os.stat("/").st_dev
        path_device = _existing_device(path)
        public_device = os.stat(public_default).st_dev
    except OSError:
        root_device = path_device = public_device = None
    if path_device == root_device and public_device != root_device and public_default != path:
        warnings.warn(
            f"Redirecting checkpoint scratch from root filesystem {path} to {public_default}.",
            RuntimeWarning,
            stacklevel=2,
        )
        path = public_default
    path.mkdir(parents=True, exist_ok=True)

    if not os.access(path, os.W_OK):
        raise PermissionError(f"Checkpoint temporary directory is not writable: {path}")
    free_bytes = shutil.disk_usage(path).free
    minimum_free_bytes = int(
        os.environ.get("BG_PDR_FM_MIN_SCRATCH_BYTES", str(512 * 1024 * 1024))
    )
    if free_bytes < minimum_free_bytes:
        raise OSError(
            f"Checkpoint scratch has insufficient free space: {path} has {free_bytes} bytes, "
            f"requires at least {minimum_free_bytes}."
        )

    # Exercise the same directory used by tempfile before a long run starts.
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".bg-pdr-fm-probe-", dir=path, delete=False) as probe:
            probe.write(b"bg-pdr-fm scratch probe\n")
            probe.flush()
            os.fsync(probe.fileno())
            probe_path = Path(probe.name)
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
    return path


def _apply_scratch_environment(path: Path, env: dict[str, str]) -> dict[str, str]:
    """Populate all common Python/ML cache variables for a worker process."""
    cache_root = path / "cache"
    cache_dirs = {
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
        "HF_HOME": cache_root / "huggingface",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch_extensions",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "PYTHONPYCACHEPREFIX": cache_root / "pycache",
    }
    path.mkdir(parents=True, exist_ok=True)
    for cache_dir in cache_dirs.values():
        cache_dir.mkdir(parents=True, exist_ok=True)
    value = str(path)
    env.update(
        {
            "TMPDIR": value,
            "TMP": value,
            "TEMP": value,
            "BG_PDR_FM_CHECKPOINT_TMPDIR": value,
            **{key: str(cache_dir) for key, cache_dir in cache_dirs.items()},
        }
    )
    return env


def configure_checkpoint_worker_environment(
    env: dict[str, str] | None = None,
    scratch_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Return a child environment whose temporary writes use public scratch."""
    child_env = dict(os.environ if env is None else env)
    path = resolve_checkpoint_temp_dir(scratch_dir)
    return _apply_scratch_environment(path, child_env)


def configure_checkpoint_temp_dir(preferred: str | os.PathLike[str] | None = None) -> str:
    """Route atomic checkpoint temporary files to the data volume.

    Lightning/fsspec writes a complete temporary checkpoint before moving it
    into the final checkpoint directory. On this host, ``/tmp`` is on a small
    root filesystem while experiment outputs live on ``/public``.
    """
    path = resolve_checkpoint_temp_dir(preferred)
    _apply_scratch_environment(path, os.environ)
    value = str(path)
    tempfile.tempdir = value
    return value


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_torch_runtime(matmul_precision: str | None = None) -> None:
    """Apply project-wide PyTorch runtime settings.

    K100_AI is exposed through PyTorch's CUDA API but uses a ROCm/HIP backend.
    On this stack, the default scaled-dot-product attention dispatcher can try
    to load CUDA flash-attention libraries. Prefer the math backend on ROCm so
    training and evaluation stay portable.
    """
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))

    if not torch.cuda.is_available():
        return

    is_rocm = getattr(torch.version, "hip", None) is not None
    allow_tf32 = not is_rocm or _env_enabled("HIPBLASLT_ALLOW_TF32")
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn") and not is_rocm:
        torch.backends.cudnn.allow_tf32 = True

    if not is_rocm:
        return

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def configured_torch_device(accelerator: Any = "auto", devices: Any = "auto") -> torch.device:
    """Resolve a single torch device from Lightning-style config values."""
    accelerator_name = str(accelerator).lower()
    if accelerator_name not in {"auto", "gpu", "cuda"}:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution was requested, but torch.cuda.is_available() is false.")
    index = _first_device_index(devices)
    device_count = torch.cuda.device_count()
    if index >= device_count:
        index = 0
    return torch.device(f"cuda:{index}")


def _first_device_index(devices: Any) -> int:
    if devices is None:
        return 0
    if isinstance(devices, int):
        return 0
    if isinstance(devices, str):
        text = devices.strip().lower()
        if text in {"", "auto"}:
            return 0
        if "," in text:
            text = text.split(",", 1)[0].strip()
        return int(text) if text.isdigit() else 0
    try:
        values = list(devices)
    except TypeError:
        values = []
    if values:
        try:
            return int(values[0])
        except (TypeError, ValueError):
            return 0
    return 0
