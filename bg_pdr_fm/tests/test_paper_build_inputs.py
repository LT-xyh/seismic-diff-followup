from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "docs" / "paper" / "AAAI2027"


def _copy_build_script(tmp_path: Path) -> Path:
    copied_dir = tmp_path / "paper"
    copied_dir.mkdir()
    copied = copied_dir / "build.sh"
    copied.write_text((PAPER_DIR / "build.sh").read_text())
    copied.chmod(0o755)
    return copied


def _fake_tex_toolchain(tmp_path: Path) -> tuple[Path, Path]:
    tools_dir = tmp_path / "fake-tex"
    tools_dir.mkdir()
    marker = tmp_path / "working-directories.txt"
    script = "#!/usr/bin/env bash\nprintf '%s\\n' \"$PWD\" >> \"$BUILD_TEST_LOG\"\n"
    for name in ("pdflatex", "bibtex"):
        binary = tools_dir / name
        binary.write_text(script)
        binary.chmod(0o755)
    return tools_dir, marker


def test_submission_wrapper_defines_flag_before_inputting_paper():
    paper = PAPER_DIR / "paper.tex"
    submission = PAPER_DIR / "paper_submission.tex"

    assert paper.is_file()
    assert submission.is_file()
    wrapper = submission.read_text()
    assert wrapper.index(r"\newif\ifaaaiwithoutappendix") < wrapper.index(r"\aaaiwithoutappendixtrue")
    assert wrapper.index(r"\aaaiwithoutappendixtrue") < wrapper.index(r"\input{paper.tex}")


def test_build_script_is_directly_executable():
    assert os.access(PAPER_DIR / "build.sh", os.X_OK)


def test_build_script_uses_its_own_directory_for_tex_commands(tmp_path: Path):
    build_script = _copy_build_script(tmp_path)
    tools_dir, marker = _fake_tex_toolchain(tmp_path)
    env = os.environ.copy()
    env.update({"TL": str(tools_dir), "BUILD_TEST_LOG": str(marker)})

    result = subprocess.run(
        ["bash", str(build_script), "submission"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text().splitlines() == [str(build_script.parent)] * 4


def test_build_script_has_explicit_missing_binary_errors(tmp_path: Path):
    build_script = _copy_build_script(tmp_path)
    env = os.environ.copy()
    env["TL"] = str(tmp_path / "missing-tex")

    result = subprocess.run(
        ["bash", str(build_script), "submission"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Configured pdflatex binary is missing" in result.stderr


def test_build_script_rejects_invalid_mode_before_checking_toolchain(tmp_path: Path):
    build_script = _copy_build_script(tmp_path)
    env = os.environ.copy()
    env["TL"] = str(tmp_path / "missing-tex")

    result = subprocess.run(
        ["bash", str(build_script), "invalid"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_build_script_declares_self_directory_and_both_binary_guards():
    script = (PAPER_DIR / "build.sh").read_text()

    assert "SCRIPT_DIR=" in script
    assert 'cd "$SCRIPT_DIR"' in script
    assert "Configured pdflatex binary is missing" in script
    assert "Configured bibtex binary is missing" in script
