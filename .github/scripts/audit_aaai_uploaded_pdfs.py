#!/usr/bin/env python3
"""Read-only audit of user-supplied AAAI PDF exports and byte-exact Step-0 source."""
import hashlib
import json
from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "docs/paper/RemoteSensing_AAAI_Rebuild"
HIST = ROOT / "docs/paper/aaaii"
AAAI = ROOT / "docs/paper/AAAI2027"
OUT = ROOT / "build/aaai-original-restoration"
BLOB_SHAS = {
    "condition_learning.pdf": "589056ce25a036661fae0f90dcca892643f84bb1",
    "method_overview.pdf": "d2f3ee10c3bfa72fbf605209abb30809b36ef170",
    "motivation.pdf": "e979407337000c9b44f73fb2f4d2cf6b2894ea30",
    "qualitative.pdf": "44998d35bd675f6306cc1491b10639069071cc55",
    "residual_transport.pdf": "31ad8cd7badd4cae005401433193486370c4b6f7",
    "table2_failure_cases.pdf": "33fd940c63c3fda49425380a4d787032274ffe24",
    "table2_representative_cases_1.pdf": "801b4416a6b2241d49445a9334f3e002cf7a86c1",
    "table2_representative_cases_2.pdf": "2cc25e65b07d94a4b1cde1df0f00eeeca93bc9c2",
}
SOURCE_SHA256 = {
    "pd-bg-rfm.tex": "51d57b5954516e0040114aee9a3863a4ea5869b78500cde7d4112d6edf4d7e83",
    "paper0726.tex": "5c933cb974ba83b47b3394e11a3b518818832aa8f2315e2210a70d9f721710e1",
}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def blob_sha(b):
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_SHA256.items():
        b1, b2 = (HIST/name).read_bytes(), (WORK/name).read_bytes()
        assert b1 == b2 and sha256(b2) == expected, name

    records = []
    for name, expected in BLOB_SHAS.items():
        locs = [
            HIST / "figures/aaai" / name,
            AAAI / "figures/aaai" / name,
            WORK / "figures/aaai" / name,
        ]
        blobs = [x.read_bytes() for x in locs]
        assert all(x == blobs[0] for x in blobs), "Cross-directory PDF mismatch: " + name
        digest = blob_sha(blobs[0])
        assert digest == expected, "Uploaded Git blob not as verified: " + name
        assert blobs[0].startswith(b"%PDF-"), "Invalid PDF header: " + name
        reader = PdfReader(str(locs[2]), strict=True)
        pages = len(reader.pages)
        assert pages > 0, "PDF has no pages: " + name
        for page in reader.pages:
            _ = page.mediabox
        records.append({
            "filename": name,
            "git_blob_sha": digest,
            "sha256": sha256(blobs[0]),
            "bytes": len(blobs[0]),
            "pages": pages,
            "copies": [str(x.relative_to(ROOT)) for x in locs],
            "three_copies_identical": True,
        })

    # Rebuild a manifest for immutable source/artwork only. Historical metadata
    # and LaTeX-generated files are intentionally not included.
    immutable = [
        p for p in WORK.rglob("*")
        if p.is_file()
        and p.name not in ("PROVENANCE.json", "README_RESTORATION.md", "SHA256SUMS.txt")
        and p.suffix.lower() not in (".aux",".log",".toc",".out",".fls",".fdb_latexmk",
                                     ".bbl",".blg",".synctex.gz")
        and p.name != "pd-bg-rfm.pdf"
    ]
    manifest = "".join(
        sha256(p.read_bytes())+"  "+p.relative_to(WORK).as_posix()+"\n"
        for p in sorted(immutable)
    )
    (OUT/"SHA256SUMS.updated.txt").write_text(manifest, encoding="utf-8")
    report = {
        "status": "UPLOADED PDF FIGURES VERIFIED (NOT THE SUBMITTED MANUSCRIPT PDF)",
        "supplied_branch": "xyh/upload-aaai-pdf-figures",
        "supplied_commit": "dc1e70dab42ad01f043961a01711ae8f4e950a48",
        "original_manuscript_pdf": "NOT UPLOADED; historic submission equivalence unverified",
        "source_sha256": SOURCE_SHA256,
        "figures": records,
        "manifest_entries": len(immutable),
    }
    (OUT/"uploaded_figure_integrity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print("AAAI_PDF_ASSET_AUDIT_PASS",len(records),"figure PDFs",
          "three identical copies each",len(immutable),"manifest files")
    for x in records:
        print(x["filename"],"git",x["git_blob_sha"],"sha256",x["sha256"],"pages",x["pages"])


if __name__ == "__main__":
    main()
