from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_README = REPO_ROOT / "reproducibility" / "README.md"
FORBIDDEN_SOURCE_PARTS = {"_build_full", "_build_submission", "checkpoints", "logs"}


def load_release_manifest() -> dict[str, object]:
    assert RELEASE_README.is_file(), f"missing release README: {RELEASE_README}"
    readme = RELEASE_README.read_text(encoding="utf-8")
    match = re.search(r"`(reproducibility/release_manifest\.yaml)`", readme)
    assert match, "reproducibility/README.md must declare the release manifest path"

    manifest_path = REPO_ROOT / match.group(1)
    assert manifest_path.is_file(), f"missing release manifest: {manifest_path}"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict), "release manifest must be a YAML mapping"
    return manifest


def test_release_manifest_contains_only_source_inputs() -> None:
    manifest = load_release_manifest()

    assert manifest["version"] == 1
    assert manifest["entrypoints"]
    assert manifest["files"]
    for path in manifest["files"]:
        assert isinstance(path, str)
        source_path = REPO_ROOT / path
        assert source_path.is_file(), path
        assert not FORBIDDEN_SOURCE_PARTS.intersection(Path(path).parts)
        assert not path.endswith(".local.yaml")


def test_release_branch_does_not_track_generated_release_artifacts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    forbidden = {
        "_build_full",
        "_build_submission",
        "logs",
        "checkpoints",
        "release_outputs",
    }
    generated = [
        path
        for path in tracked
        if forbidden.intersection(Path(path).parts)
        or path.endswith(".local.yaml")
        or "__pycache__" in Path(path).parts
        or Path(path).name.endswith(".frame.png")
    ]
    assert generated == [], "generated or machine-local files remain tracked: " + ", ".join(generated[:20])


def test_release_requirements_include_tensorboard_backend() -> None:
    requirements_path = REPO_ROOT / "reproducibility" / "requirements.txt"
    requirements = {
        line.split("==", 1)[0].split(">=", 1)[0].strip().lower()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "tensorboard" in requirements
