#!/usr/bin/env python3
"""Read-only Step-1.1 verification of a clean-checkout official MDPI build."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys
from pypdf import PdfReader

ROOT = Path("docs/paper/RemoteSensing_Stepwise")
WORK = Path("build/step1-mdpi")
PDF = ROOT / "main.pdf"
LOG = ROOT / "main.log"
BBL = ROOT / "main.bbl"
REPORT = WORK / "pdf-verification.json"
APPENDICES = (
    "Theoretical Derivations and Proofs",
    "Supplementary Results and Diagnostics",
    "Complete Condition-Learning Objective",
    "Implementation and Evaluation Protocols",
)
EXPECTED_VALUES = ("336,000", "33,600", "0.772", "0.0270", "0.9913")
EXPECTED_CLASS_HASH = "658dbb5b2db2f6560bf3de3ecff7efac310a5817721eebd7539c1990b5345f01"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    state = {"status": "FAIL", "scientific_content_changed": False}
    try:
        if not PDF.is_file() or PDF.stat().st_size < 10000:
            raise RuntimeError("MDPI compiled PDF missing")
        if not LOG.is_file() or not BBL.is_file() or BBL.stat().st_size < 1000:
            raise RuntimeError("LaTeX / BibTeX outputs missing")
        if sha(ROOT / "Definitions/mdpi.cls") != EXPECTED_CLASS_HASH:
            raise RuntimeError("Official class SHA mismatch")
        template = (ROOT / "OFFICIAL_TEMPLATE_REFERENCE.tex").read_text(encoding="utf-8")
        maintex = (ROOT / "main.tex").read_text(encoding="utf-8")
        for marker in (r"\addhighlights{yes}", r"\renewcommand{\addhighlights}"):
            if marker not in template or marker not in maintex:
                raise RuntimeError("Highlights implementation deviates from official template: "+marker)
        for marker in ("What are the main findings?",
                       "What are the implications of the main findings?",
                       "[Pending author-supplied findings]", "[Pending author-supplied implications]"):
            if marker not in maintex:
                raise RuntimeError("Missing author-unconfirmed Highlights marker: "+marker)
        body = (ROOT / "CONTENT_MDPI_LAYOUT.tex").read_text(encoding="utf-8")
        if r"\appendixstart" not in body or r"\appendixtitles{yes}" not in body:
            raise RuntimeError("Official appendix formatting missing")
        figures = sorted(set(re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)))
        if len(figures) != 9:
            raise RuntimeError("Expected nine original figure paths")
        original = Path("docs/paper/RemoteSensing_AAAI_Rebuild")
        for f in figures:
            if (ROOT / f).read_bytes() != (original / f).read_bytes():
                raise RuntimeError("Original figure altered: "+f)
        warnings = []
        critical = ("There were undefined references", "There were undefined citations",
                    "Citation", "Reference", "destination with the same identifier",
                    "multiply-defined labels", "Fatal error occurred", "! LaTeX Error:")
        for line in LOG.read_text(errors="replace").splitlines():
            if ("Warning" in line or "warning" in line or
                    "Overfull " in line or "Underfull " in line or "! LaTeX Error:" in line):
                warnings.append(line.strip()[:300])
        fatal = [x for x in warnings if (
            any(y.lower() in x.lower() for y in critical) and
            ("undefined" in x.lower() or "multiply" in x.lower() or
             "destination" in x.lower() or "! latex error" in x.lower()
             or "fatal" in x.lower())
        )]
        if fatal:
            raise RuntimeError("Unresolved cross-references or duplicate destinations: "+str(fatal[:6]))
        reader = PdfReader(PDF)
        pages = len(reader.pages)
        if not 20 <= pages <= 50:
            raise RuntimeError("Unexpected MDPI PDF page count: "+str(pages))
        page_text = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page_text)
        normalized = re.sub(r"\s+", " ", text).lower()
        for term in (*APPENDICES, "What are the main findings?",
                     "What are the implications of the main findings?"):
            if term.lower() not in normalized:
                raise RuntimeError("Not rendered in PDF: "+term)
        for val in EXPECTED_VALUES:
            if val not in text:
                raise RuntimeError("Historical numerical token missing: "+val)
        if any(len(t) < 10 for t in page_text):
            raise RuntimeError("Unexpected blank PDF page")
        if r"\bibitem" not in BBL.read_text(errors="replace"):
            raise RuntimeError("BibTeX result has no bibliography items")
        state.update(status="PASS", pages=pages, pdf_sha256=sha(PDF),
                     pdf_bytes=PDF.stat().st_size,
                     original_figures_byte_identical=len(figures),
                     appendix_sections_present=list(APPENDICES),
                     highlights="Official mdpi.cls macro with original template redefinition and unfilled placeholders",
                     references_and_citations_resolved=True,
                     warnings=warnings[:60], warning_count=len(warnings))
        print("STEP1_OFFICIAL_MDPI_BUILD_PASS pages",pages,"original_figures",len(figures))
        return 0
    except Exception as ex:
        state["error"] = str(ex)
        print("STEP1_OFFICIAL_MDPI_BUILD_FAILURE",str(ex))
        return 1
    finally:
        REPORT.write_text(json.dumps(state,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
