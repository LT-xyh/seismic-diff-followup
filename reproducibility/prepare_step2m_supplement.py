#!/usr/bin/env python3
"""Prepare the project-owned BG-RFM source-code ZIP for journal peer review.

No dataset, model checkpoint, credential, or third-party baseline is redistributed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

ROOT_PREFIX = "BG-RFM-Supplementary-Code/"
SECRET_RE = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{30,}\b"),
]
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".json"}
REQUIRED = {
    "bg_pdr_fm/models/encoder.py",
    "bg_pdr_fm/models/generators.py",
    "bg_pdr_fm/lightning/bg_pdr_fm_module.py",
    "bg_pdr_fm/training/train_bg_pdr_fm.py",
    "bg_pdr_fm/evaluation/evaluate_bg_pdr_fm.py",
    "bg_pdr_fm/evaluation/benchmark_metrics.py",
    "scripts/aaai27/prepare_openfwi.py",
    "scripts/aaai27/train.py",
    "scripts/aaai27/evaluate.py",
    "scripts/aaai27/smoke.py",
    "reproducibility/requirements.txt",
    "reproducibility/release_manifest.yaml",
    "reproducibility/THIRD_PARTY_NOTICES.md",
}

def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])

def package_allowed(path: str, manifest_files: set[str]) -> bool:
    p = Path(path)
    if not path or path.startswith("bg_pdr_fm/external_baselines/"):
        return False
    if any(x in ("__pycache__", "generated", "outputs", "checkpoints", "cache") or x.startswith(".") for x in p.parts):
        return False
    if any(x in p.name.lower() for x in ("credentials", "auth.json", "private", "secret", "password")):
        return False
    if path in manifest_files:
        return True
    if path in ("README.md", "reproducibility/prepare_step2m_supplement.py"):
        return True
    if p.suffix.lower() not in SOURCE_SUFFIXES:
        return False
    if path.startswith(("bg_pdr_fm/", "scripts/aaai27/", "reproducibility/", "tests/")):
        return True
    if path in ("tools/build_openfwi_lmdb.py", "tools/benchmark_dataset_io.py"):
        return True
    return False

def entry(name: str) -> zipfile.ZipInfo:
    obj = zipfile.ZipInfo(ROOT_PREFIX + name, (2026, 10, 16, 0, 0, 0))
    obj.compress_type = zipfile.ZIP_DEFLATED
    obj.external_attr = 0o644 << 16
    return obj

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo.resolve()
    manifest = (root / "reproducibility/release_manifest.yaml").read_text()
    manifest_files: set[str] = set()
    inside = False
    for line in manifest.splitlines():
        if line == "files:":
            inside = True
            continue
        if inside and line and not line.startswith(" "):
            inside = False
        if inside and line.startswith("  - "):
            manifest_files.add(line[4:].strip())
    tracked = sorted(x.decode() for x in git(root, "ls-files", "-z").split(b"\0") if x)
    names = [x for x in tracked if package_allowed(x, manifest_files)]
    assert REQUIRED.issubset(names), "Essential source files missing from package allowlist"
    files: dict[str, bytes] = {}
    for name in names:
        payload = (root / name).read_bytes()
        if b"\0" in payload or any(q.search(payload) for q in SECRET_RE):
            raise RuntimeError("Binary or possible secret in package: " + name)
        if name.endswith(".py"):
            compile(payload.decode("utf-8"), name, "exec")
        files[name] = payload
    size = sum(map(len, files.values()))
    assert 0 < size < 20 * 1024 * 1024, f"Unexpected source bytes: {size}"
    sha = git(root, "rev-parse", "HEAD").decode().strip()
    notes = (
        "# BG-RFM Supplementary Source Code (Prepared for Journal Peer Review)\n\n"
        f"Snapshot: {sha}\n"
        "This archive has NOT been submitted to MDPI.\n\n"
        "## Reproduction entrypoints\n"
        "Python 3.10, compatible PyTorch and dependencies: reproducibility/requirements.txt\n"
        "Preparation: python scripts/aaai27/prepare_openfwi.py --data-root data/openfwi --output data/openfwi_lmdb\n"
        "Train: python scripts/aaai27/train.py --config bg_pdr_fm/configs/release/aaai27/train.yaml\n"
        "Evaluate: python scripts/aaai27/evaluate.py --config bg_pdr_fm/configs/release/aaai27/evaluate.yaml\n"
        "CPU smoke: python scripts/aaai27/smoke.py\n"
        "Use --dry-run with relevant wrappers before executing expensive commands.\n"
        "See scripts/aaai27/README.md and reproducibility/README.md.\n\n"
        "## Evidence and dependencies\n"
        "No raw/OpenFWI-derived 336,000-record materialization, LMDB files, model checkpoints, "
        "run outputs or generated images are included. Full experiments require separately "
        "obtained data and the formal reported checkpoint.\n"
        "SeisMIC-Init is a related public dataset, not verified as an exact copy of these records.\n"
        "Third-party baseline implementations under bg_pdr_fm/external_baselines/ are excluded "
        "because public redistribution permissions have not been established. The original "
        "reproducibility/release_manifest.yaml includes these external dependencies; its "
        "full-checkout validator is not guaranteed to pass on this legally reduced archive. "
        "See reproducibility/THIRD_PARTY_NOTICES.md for upstream details.\n"
        "The repository has no general project license; no license is invented here. "
        "A later publication-specific GitHub release requires author-approved licensing.\n"
    ).encode()
    checks = "".join(hashlib.sha256(v).hexdigest() + "  " + n + "\n" for n, v in files.items()).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as arc:
        for name, payload in sorted(files.items()):
            arc.writestr(entry(name), payload)
        arc.writestr(entry("SUPPLEMENTARY_README.md"), notes)
        arc.writestr(entry("SOURCE_SHA256SUMS.txt"), checks)
    with zipfile.ZipFile(args.output) as arc:
        assert arc.testzip() is None
        assert len(arc.namelist()) == len(files) + 2
    result = {
        "source_commit": sha,
        "archive": args.output.name,
        "number_of_project_owned_files": len(files),
        "source_bytes": size,
        "zip_bytes": args.output.stat().st_size,
        "project_license_present": False,
        "external_baseline_code_excluded": True,
        "dataset_and_checkpoints_excluded": True,
        "submitted_to_MDPI": False,
        "full_reproduction_requires_separate_data_checkpoints_and_licensed_upstream_baselines": True,
    }
    (args.output.parent / "STEP2M_CODE_AUDIT.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
