#!/usr/bin/env python3
"""Restore byte-identical AAAI manuscript; separately label recreated figure exports."""
import hashlib
import json
from pathlib import Path
import re
import shutil

import cairosvg
import img2pdf

ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / "docs/paper/aaaii"
ART = ROOT / "docs/paper/AAAI2027/figures"
DST = ROOT / "docs/paper/RemoteSensing_AAAI_Rebuild"
EXPECTED = {
    "pd-bg-rfm.tex": "51d57b5954516e0040114aee9a3863a4ea5869b78500cde7d4112d6edf4d7e83",
    "paper0726.tex": "5c933cb974ba83b47b3394e11a3b518818832aa8f2315e2210a70d9f721710e1",
}
FIGURES = {
    "figures/aaai/motivation.pdf": "figure1_motivation/motivation_statistics_combined_v2.png",
    "figures/aaai/method_overview.pdf": "method_overview.svg",
    "figures/aaai/qualitative.pdf": "table2_qualitative/table2_representative_cases.png",
    "figures/aaai/condition_learning.pdf": "figure3_condition_learning/figure3_condition_learning.svg",
    "figures/aaai/residual_transport.pdf": "figure4_residual_transport/figure4_residual_transport.svg",
    "figures/aaai/table2_representative_cases_1.pdf": "table2_qualitative/table2_representative_cases_1.png",
    "figures/aaai/table2_representative_cases_2.pdf": "table2_qualitative/table2_representative_cases_2.png",
    "figures/aaai/table2_failure_cases.pdf": "table2_qualitative/table2_failure_cases.png",
    "figures/aaai/appendix_full_condition_diagnostics.png": "aaai/appendix_full_condition_diagnostics.png",
}
EDITABLE = [
    "method_overview.svg", "method_overview_editable.pptx", "method_overview_preview.png",
    "figure1_motivation/motivation_statistics_combined_v2.png",
    "figure1_motivation/motivation_statistics_combined_v2_editable.pptx",
    "figure3_condition_learning/figure3_condition_learning.svg",
    "figure3_condition_learning/figure3_condition_learning_editable.pptx",
    "figure4_residual_transport/figure4_residual_transport.svg",
    "figure4_residual_transport/figure4_transport_energy_editable.pptx",
    "table2_qualitative/table2_representative_cases.png",
    "table2_qualitative/table2_representative_cases_1.png",
    "table2_qualitative/table2_representative_cases_2.png",
    "table2_qualitative/table2_failure_cases.png",
    "aaai/appendix_full_condition_diagnostics.png",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_copy(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert source.read_bytes() == target.read_bytes()


def main():
    for name, expected in EXPECTED.items():
        if sha(HIST / name) != expected:
            raise RuntimeError("Historic manuscript differs from verified archive: " + name)
    if DST.exists():
        raise RuntimeError("Restoration directory exists; refuse to overwrite.")
    DST.mkdir(parents=True)
    for name in ["pd-bg-rfm.tex", "paper0726.tex", "references.bib",
                 "aaai2027.sty", "aaai2027.bst", "ReproducibilityChecklist.tex"]:
        exact_copy(HIST / name, DST / name)
    for name in EDITABLE:
        exact_copy(ART / name, DST / "original_artwork" / name)
    content = (DST / "pd-bg-rfm.tex").read_text(encoding="utf-8")
    required = sorted(set(re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content)))
    if required != sorted(FIGURES):
        raise RuntimeError("Unexpected figure references: " + str(required))
    records = []
    for rel in required:
        archived_export = HIST / rel
        target = DST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if archived_export.is_file():
            exact_copy(archived_export, target)
            original = archived_export
            status = "ORIGINAL_EXPORTED_FILE"
        else:
            original = ART / FIGURES[rel]
            if not original.is_file():
                raise FileNotFoundError(original)
            if target.suffix == ".png":
                exact_copy(original, target)
                status = "ORIGINAL_PNG_COPY_AT_RESTORED_PATH"
            elif original.suffix == ".svg":
                cairosvg.svg2pdf(url=str(original), write_to=str(target))
                status = "NEW_SVG_TO_PDF_EXPORT_NOT_VERIFIED_SUBMITTED"
            elif original.suffix == ".png":
                target.write_bytes(img2pdf.convert(str(original)))
                status = "NEW_PNG_TO_PDF_EXPORT_NOT_VERIFIED_SUBMITTED"
            else:
                raise RuntimeError(original)
        records.append({
            "latex_reference": rel,
            "archived_artwork": str(original.relative_to(ROOT)),
            "source_sha256": sha(original),
            "restored_sha256": sha(target),
            "status": status,
        })

    candidate_pdfs = [
        str(p.relative_to(ROOT)) for p in (ROOT/"docs/paper").rglob("*.pdf")
        if "RemoteSensing_AAAI_Rebuild" not in str(p)
    ]
    status = {
        "classification": "RECONSTRUCTED FROM ARCHIVED SOURCE",
        "verified_original_submission_pdf": False,
        "original_pdf_candidates": sorted(candidate_pdfs),
        "source_tex_sha256": EXPECTED,
        "copied_sources_byte_identical": True,
        "figure_exports": records,
    }
    (DST / "PROVENANCE.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme = [
        "# Original AAAI-27 manuscript restoration — STEP 0",
        "",
        "CLASSIFICATION: RECONSTRUCTED FROM ARCHIVED SOURCE.",
        "The exact submitted AAAI PDF was not found in the bounded repository",
        "or currently accessible Project attachments. PDF equality is not verified.",
        "",
        "The following were copied byte-for-byte from docs/paper/aaaii:",
        "pd-bg-rfm.tex, paper0726.tex, references.bib, aaai2027.sty,",
        "aaai2027.bst, ReproducibilityChecklist.tex.",
        "Original claims, text, equations, tables, numbers and citations were",
        "retained without edits. The copied source SHA256 values are:",
        "",
        "pd-bg-rfm.tex: " + EXPECTED["pd-bg-rfm.tex"],
        "paper0726.tex: " + EXPECTED["paper0726.tex"],
        "",
        "The historic LaTeX references eight PDF figure exports and one PNG.",
        "The eight originally named PDF exports are ABSENT from the archive.",
        "They have been newly exported from existing original AAAI2027 SVG/PNG",
        "artwork for compilation; these are derived rendering artifacts, NOT",
        "authenticated submitted PDF figure files. Editable originals are",
        "preserved under original_artwork. The appendix PNG is an exact art copy.",
        "",
        "Figure export provenance (also in PROVENANCE.json):",
    ]
    readme += [
        "- " + r["latex_reference"] + " <- " + r["archived_artwork"] + " : " + r["status"]
        for r in records
    ]
    readme += [
        "", "Rebuild with: latexmk -pdf pd-bg-rfm.tex",
        "Compiled output, if successful, is RECONSTRUCTED FROM ARCHIVED SOURCE.",
        "No Step 1 scientific editing or MDPI conversion was performed.",
    ]
    (DST / "README_RESTORATION.md").write_text(
        "\n".join(readme) + "\n", encoding="utf-8")
    checks = []
    for p in sorted(DST.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            checks.append(sha(p) + "  " + p.relative_to(DST).as_posix())
    (DST / "SHA256SUMS.txt").write_text(
        "\n".join(checks) + "\n", encoding="utf-8")
    for name in EXPECTED:
        assert sha(HIST / name) == sha(DST / name)
    print("RESTORATION_PASS")
    print("PRIMARY_SHA256=" + EXPECTED["pd-bg-rfm.tex"])
    print("RESTORED_FILES=" + str(len(checks)))
    print("RECONSTRUCTED_PDF_EXPORTS=8")
    print("ORIGINAL_SUBMISSION_PDF=NOT_FOUND")


if __name__ == "__main__":
    main()
