#!/usr/bin/env python3
"""Install only the verified user-supplied current official MDPI ACS LaTeX ZIP.

Usage: python docs/paper/RemoteSensing_Stepwise/tools/install_uploaded_mdpi_acs.py /path/to/MDPI_template_ACS.zip
Does not edit scientific text or experimental artifacts.
"""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ZIP_SHA256 = "62744425fbcec9cd3e58147cbee65bdec2e2ff0440f29792c26edc97a11b6c70"
CLASS_SHA256 = "658dbb5b2db2f6560bf3de3ecff7efac310a5817721eebd7539c1990b5345f01"

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Please supply the verified 2026-09-11 official ACS ZIP path")
    p = Path(sys.argv[1]).resolve()
    archive = p.read_bytes()
    if hashlib.sha256(archive).hexdigest() != ZIP_SHA256:
        raise SystemExit("Official archive SHA-256 mismatch; no substitute accepted")
    with zipfile.ZipFile(p) as z:
        for required in ("template.tex", "Definitions/mdpi.cls",
                         "Definitions/mdpi.bst", "Definitions/journalnames.tex"):
            if required not in z.namelist():
                raise RuntimeError("Official MDPI template is incomplete: " + required)
        cls = z.read("Definitions/mdpi.cls")
        if (hashlib.sha256(cls).hexdigest() != CLASS_SHA256 or
            b"\\def\\mdpidate{2026-09-11}" not in cls or
            b"\\def\\mdpiversion{v6.5a}" not in cls):
            raise RuntimeError("MDPI class does not match the audited official v6.5a")
        for name in z.namelist():
            if name.startswith("Definitions/") and not name.endswith("/"):
                out = ROOT / name
                out.parent.mkdir(parents=True, exist_ok=True)
                data = z.read(name)
                if out.exists() and out.read_bytes() != data:
                    raise RuntimeError("Refusing to overwrite differing official file: " + name)
                out.write_bytes(data)
        (ROOT / "OFFICIAL_TEMPLATE_REFERENCE.tex").write_bytes(z.read("template.tex"))
    status = {"verified_zip_sha256": ZIP_SHA256, "mdpi_class": "v6.5a",
              "mdpi_class_date": "2026-09-11",
              "official_source": "https://www.mdpi.com/authors/latex",
              "scientific_content_changed": False,
              "status": "format-only MDPI template installed; author metadata pending"}
    (ROOT / "OFFICIAL_MDPI_TEMPLATE_PROVENANCE.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print("MDPI_OFFICIAL_TEMPLATE_INSTALL_PASS", "v6.5a")

if __name__ == "__main__":
    main()
