#!/usr/bin/env python3
"""Stage ONLY byte-identical scientific material for the current MDPI template.

Does not select an unofficial or outdated class. The official September 2026
ACS template is a required separate input before main.tex can be finalized.
No scientific prose, equations, captions, tables, citations, figures or results
may be edited in this stage.
"""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parents[1]
ORIGINAL = HERE / "original" / "pd-bg-rfm.tex"
EARLY = HERE / "original" / "paper0726.tex"
BIB = HERE / "references.bib"
MDPI_BODY = HERE / "CONTENT_EXACT_FROM_AAAI.tex"
ABSTRACT = HERE / "ABSTRACT_EXACT_FROM_AAAI.tex"
AUDIT = HERE / "CONTENT_PRESERVATION.json"
SOURCE_SHA = "51d57b5954516e0040114aee9a3863a4ea5869b78500cde7d4112d6edf4d7e83"
EARLY_SHA = "5c933cb974ba83b47b3394e11a3b518818832aa8f2315e2210a70d9f721710e1"
BIB_SHA = "9fa6af7bcf73bc3c1ced76e98eae3a235c564ce802101c94bafec435cfbac356"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha(ORIGINAL) == SOURCE_SHA, "AAAI primary source altered"
    assert sha(EARLY) == EARLY_SHA, "AAAI early manuscript altered"
    assert sha(BIB) == BIB_SHA, "AAAI bibliography altered"
    # Historical Git blob identity is checked separately in CI for the bib.
    original = ORIGINAL.read_text(encoding="utf-8")
    assert original.count("\\begin{abstract}") == original.count("\\end{abstract}") == 1
    abstart = original.index("\\begin{abstract}") + len("\\begin{abstract}")
    abend = original.index("\\end{abstract}", abstart)
    abstract_text = original[abstart:abend]
    end_abstract = abend + len("\\end{abstract}")
    assert original[end_abstract:].endswith("\\end{document}\n")
    scientific_body = original[end_abstract:original.rfind("\\end{document}")]

    # Byte-exact substring, including all historical captions and appendices.
    assert abstract_text in original
    assert scientific_body in original
    ABSTRACT.write_text(abstract_text, encoding="utf-8")
    MDPI_BODY.write_text(scientific_body, encoding="utf-8")

    def capture_environment(s, env):
        pattern = rf"\\begin\{{{re.escape(env)}\}}[\s\S]*?\\end\{{{re.escape(env)}\}}"
        return re.findall(pattern, s)
    citations = re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}",original)
    labels = re.findall(r"\\label\{([^}]+)\}",original)
    refkeys = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}",original)
    includegraphics = sorted(set(re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",original)))
    bib_entries = re.findall(r"@\w+\s*\{\s*([^,\n]+)",BIB.read_text(encoding="utf-8"))
    needed = set(k.strip() for v in citations for k in v.split(","))
    missing = sorted(needed-set(bib_entries))
    if missing:raise RuntimeError("Missing original citation entries: "+str(missing))
    figures_missing = [f for f in includegraphics if not (HERE/f).is_file()]
    if figures_missing:raise RuntimeError("Missing original figures: "+str(figures_missing))
    scientific_checks = {
        "status":"STEP1 SCIENTIFIC SOURCE STAGED, OFFICIAL 2026-09-11 MDPI TEMPLATE PENDING",
        "primary_source":str(ORIGINAL.relative_to(HERE)),
        "primary_source_sha256":sha(ORIGINAL),
        "earlier_source_sha256":sha(EARLY),
        "bibliography_sha256":sha(BIB),
        "historical_title":"Predicting the Background, Transporting the Structure: Physics-Decoupled Multimodal Subsurface Velocity Inversion",
        "abstract_text_extract_exact": True,
        "scientific_body_substring_exact": True,
        "scientific_body_sha256":hashlib.sha256(scientific_body.encode()).hexdigest(),
        "abstract_sha256":hashlib.sha256(abstract_text.encode()).hexdigest(),
        "original_content_lines":len(scientific_body.splitlines()),
        "equation_env_count":sum(len(capture_environment(scientific_body,e)) for e in ("equation","align","align*","multline","gather","split")),
        "tabular_env_count":len(capture_environment(scientific_body,"tabular"))+len(capture_environment(scientific_body,"tabular*")),
        "figure_references":includegraphics,
        "figure_count":len(includegraphics),
        "citation_groups":len(citations),
        "citation_keys":sorted(needed),
        "label_count":len(labels),
        "cross_reference_count":len(refkeys),
        "original_bib_entry_count":len(bib_entries),
        "missing_citations":missing,
        "missing_figures":figures_missing,
        "not_yet_done":[
            "Install actual official MDPI ACS LaTeX ZIP updated 2026-09-11",
            "Read its real template.tex and Definitions/mdpi.cls before choosing commands",
            "Wrap exact original title, exact abstract, exact content with MDPI official commands",
            "Preserve bibliography entries while adapting presentation to official MDPI numeric ACS style",
            "Compile MDPI final and perform full semantic/visual comparison",
            "Create final MDPI ZIP and report page count",
        ]
    }
    AUDIT.write_text(json.dumps(scientific_checks,indent=2,ensure_ascii=False)+"\n",
                     encoding="utf-8")
    print("STAGING_PASS", len(scientific_body),"scientific characters",
          len(includegraphics),"original figures",len(needed),"citation keys")
    print("MDPI_CURRENT_OFFICIAL_TEMPLATE=NOT_DOWNLOADED (HTTP 403)")
    print("NOT_A_COMPILED_MDPI_MANUSCRIPT")


if __name__ == "__main__":
    main()
