#!/usr/bin/env python3
"""Build a minimal, auditable MDPI LaTeX source ZIP from the active manuscript.

Only main.tex, recursively referenced LaTeX inputs, references.bib and actually
referenced figures are copied. Historical archives, standalone supplement,
development documentation and TeX build outputs are intentionally excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

INPUT_PATTERN = re.compile(r"\\input\s*\{([^}]+)\}")
GRAPHICS_PATTERN = re.compile(
    r"\\(?:includegraphics(?:\[[^\]]*\])?|qpanel)\s*\{([^}]+)\}"
)
BIB_PATTERN = re.compile(r"\\bibliography\s*\{([^}]+)\}")


def scan_active_sources(root: Path) -> tuple[set[Path], set[Path]]:
    papers = set()
    images = set()
    pending = [root / "main.tex"]
    while pending:
        path = pending.pop().resolve()
        if path in papers:
            continue
        if not path.is_file() or not path.is_relative_to(root):
            raise RuntimeError(f"Missing or disallowed LaTeX file: {path}")
        papers.add(path)
        raw = path.read_text(encoding="utf-8")
        for child in INPUT_PATTERN.findall(raw):
            pending.append(root / (child if child.endswith(".tex") else child + ".tex"))
        for bib_list in BIB_PATTERN.findall(raw):
            for item in bib_list.split(","):
                pending.append(root / (item.strip() + ".bib"))
        for source_name in GRAPHICS_PATTERN.findall(raw):
            if "#" in source_name:
                continue  # Template macro argument such as \includegraphics{#1}
            image_path = (root / source_name).resolve()
            if not image_path.is_file():
                raise RuntimeError(f"Missing image: {image_path}")
            if not image_path.is_relative_to(root.parent):
                raise RuntimeError(f"Figure outside docs/paper: {image_path}")
            images.add(image_path)
    return papers, images


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/paper/RemoteSensing")
    ap.add_argument("--output-dir", default="build/remote-sensing/isolated-source")
    ap.add_argument("--zip", default="build/remote-sensing/BG-RFM_RemoteSensing_PREVIEW_source.zip")
    ap.add_argument("--strict", action="store_true", help="Reject administrative placeholders")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    destination = Path(args.output_dir).resolve()
    zip_path = Path(args.zip).resolve()

    if args.strict:
        requirements = ("submission/author_metadata.tex", "submission/admin_statements.tex",
                        "submission/ai_methods_disclosure.tex")
        for rel in requirements:
            txt = (root / rel).read_text(encoding="utf-8")
            if any(token.lower() in txt.lower() for token in
                   ("pending confirmation", "author confirmation required", "[authors to confirm",
                    "[confirmed", "not yet confirmed", "[version(s)", "preview")):
                raise RuntimeError(f"Author confirmation incomplete: {rel}")

    papers, images = scan_active_sources(root)
    # No source may be supplied via an unrelated historical archive.
    for item in papers:
        if "archive" in item.parts or item.name == "supplement.tex":
            raise RuntimeError(f"Historical source in active dependency graph: {item}")
    paths = papers | images
    # Preserve editable/provenance sources of the restored vector figure,
    # alongside the PDF assets needed by pdflatex. These are not historical
    # unrelated submission files.
    extras = [
        root / "figures/figure1_bg_rfm_restored.svg",
        root / "figures/render_restored_artwork.py",
        root.parent / "AAAI2027/figures/method_overview.svg",
        root.parent / "AAAI2027/figures/figure3_condition_learning/figure3_condition_learning.svg",
        root.parent / "AAAI2027/figures/figure4_residual_transport/figure4_residual_transport.svg",
    ]
    for asset in extras:
        if not asset.exists():
            raise RuntimeError(f"Required editable figure source missing: {asset}")
        paths.add(asset.resolve())
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for path in sorted(paths):
        rel = path.relative_to(root.parent)
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)

    readme = destination / "README_SOURCE.txt"
    readme.write_text(
        "BG-RFM Remote Sensing single-manuscript LaTeX source.\n"
        "Compile: cd RemoteSensing && latexmk -pdf main.tex\n"
        "Includes all active Appendices A-C in main.tex; no standalone supplement.\n"
        "Restored figures include embedded vector PDFs and editable AAAI-derived SVGs.\n"
        "PREVIEW ONLY: author metadata, funding, COI, GenAI and related-manuscript "
        "declarations require confirmation before submission.\n",
        encoding="utf-8",
    )
    records = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(f"{sha}  {path.relative_to(destination).as_posix()}")
    (destination / "SHA256SUMS.txt").write_text("\n".join(records) + "\n", encoding="utf-8")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=8) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(destination).as_posix())

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        banned = (".log", ".aux", ".bbl", ".blg", ".fls", ".fdb_latexmk")
        assert not any(name.endswith(banned) for name in names)
        assert not any(name.endswith("/main.pdf") or name.endswith("/supplement.pdf") for name in names)
        assert "RemoteSensing/figures/figure1_bg_rfm_restored.pdf" in names
        assert "RemoteSensing/figures/figure1_bg_rfm_restored.svg" in names
        assert not any("/archive/" in name or name.endswith("supplement.tex")
                       for name in names)
        assert "RemoteSensing/main.tex" in names
        assert "RemoteSensing/references.bib" in names
    print(f"ACTIVE_TEX_AND_BIB={len(papers)} FIGURES={len(images)} ZIP_FILES={len(names)}")
    print(f"ZIP_BYTES={zip_path.stat().st_size} ZIP={zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
