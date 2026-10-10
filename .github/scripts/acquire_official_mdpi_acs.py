#!/usr/bin/env python3
"""Acquire the exact official 2026-09-11 MDPI ACS ZIP for Step 1.1.

Order: 1) publisher mdpi.com endpoint(s), 2) optional author-controlled
PRIVATE GitHub release asset accessed by a read-only fine-grained token.
No third-party templates and no public distribution of Definitions files.

GitHub Actions environment for private fallback (configured by owner):
  MDPI_ACS_PRIVATE_REPO      = OWNER/PRIVATE-REPOSITORY
  MDPI_ACS_PRIVATE_TAG       = mdpi-acs-2026-09-11
  GH_TOKEN                   = secret with Contents:Read on that private repo

The private release asset MUST be named MDPI_template_ACS.zip and be the
unmodified user-verified publisher archive. Both exact digests are mandatory.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile

ZIP_SHA256 = "62744425fbcec9cd3e58147cbee65bdec2e2ff0440f29792c26edc97a11b6c70"
CLASS_SHA256 = "658dbb5b2db2f6560bf3de3ecff7efac310a5817721eebd7539c1990b5345f01"
CLASS_VERSION = b"\\def\\mdpiversion{v6.5a}"
CLASS_DATE = b"\\def\\mdpidate{2026-09-11}"
WORK = Path("build/step1-mdpi")
OUT = WORK / "MDPI_template_ACS.zip"
REPORT = WORK / "template-acquisition.json"
PUBLISHER_URLS = (
    "https://www.mdpi.com/data/MDPI_template.zip",
    "https://www.mdpi.com/data/MDPI_template.zip?v=20260911",
)


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def validate(blob: bytes) -> dict[str, object]:
    actual = sha256(blob)
    if actual != ZIP_SHA256:
        raise ValueError(f"SHA256_MISMATCH for ZIP: expected {ZIP_SHA256}, got {actual}")
    import io
    with zipfile.ZipFile(io.BytesIO(blob), "r") as z:
        needed = ("Definitions/mdpi.cls", "Definitions/mdpi.bst",
                  "Definitions/journalnames.tex", "template.tex")
        if not set(needed).issubset(z.namelist()):
            raise ValueError("Official archive missing required MDPI ACS files")
        cls = z.read("Definitions/mdpi.cls")
        cls_sha = sha256(cls)
        if cls_sha != CLASS_SHA256:
            raise ValueError(f"CLASS_SHA256_MISMATCH: {cls_sha}")
        if CLASS_VERSION not in cls or CLASS_DATE not in cls:
            raise ValueError("Official mdpi.cls version/date mismatch")
        template = z.read("template.tex")
        if b"\\addhighlights{yes}" not in template or b"\\renewcommand{\\addhighlights}" not in template:
            raise ValueError("Publisher template missing native Highlights example")
        if b"remotesensing" not in z.read("Definitions/journalnames.tex"):
            raise ValueError("Publisher journal list does not include remotesensing")
    return dict(zip_sha256=actual, mdpi_class_sha256=cls_sha,
                mdpi_class_version="v6.5a", mdpi_class_date="2026-09-11",
                zip_bytes=len(blob))


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "official_source": "https://www.mdpi.com/authors/latex",
        "expected_zip_sha256": ZIP_SHA256,
        "expected_class_sha256": CLASS_SHA256,
        "attempts": [],
        "status": "BLOCKED",
    }
    try:
        for url in PUBLISHER_URLS:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0",
                                  "Accept": "application/zip,application/octet-stream;q=0.8,*/*;q=0.1"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    blob = response.read()
                verified = validate(blob)
                OUT.write_bytes(blob)
                report.update({"status": "VERIFIED", "source_type": "publisher",
                               "verified_url": url, **verified})
                print("OFFICIAL_MDPI_ACS_VERIFIED_FROM_PUBLISHER", verified)
                return 0
            except urllib.error.HTTPError as e:
                report["attempts"].append({"source": "publisher", "url": url,
                                           "http_status": e.code, "error": e.reason})
                print(f"PUBLISHER_DOWNLOAD_FAILED url={url} HTTP {e.code}")
            except Exception as e:
                report["attempts"].append({"source": "publisher", "url": url,
                                           "error": str(e)[:400]})
                print(f"PUBLISHER_DOWNLOAD_FAILED url={url} error={type(e).__name__}")

        private_repo = os.environ.get("MDPI_ACS_PRIVATE_REPO", "").strip()
        private_tag = os.environ.get("MDPI_ACS_PRIVATE_TAG", "mdpi-acs-2026-09-11").strip()
        token = os.environ.get("GH_TOKEN", "")
        if private_repo and private_tag and token:
            # Never print or persist the private token or release asset.
            print("OFFICIAL_PUBLISHER_DOWNLOAD_UNAVAILABLE: trying authenticated private cache")
            result = subprocess.run(
                ["gh", "release", "download", private_tag, "--repo", private_repo,
                 "--pattern", "MDPI_template_ACS.zip", "--dir", str(WORK),
                 "--clobber"], capture_output=True, text=True, check=False, timeout=120,
            )
            if result.returncode:
                report["attempts"].append({"source": "private-authenticated-cache",
                                           "error": ("gh release download failed; verify token, "
                                                     "release tag and asset name"),
                                           "exit_code": result.returncode})
                print("PRIVATE_MDPI_CACHE_DOWNLOAD_FAILED")
            else:
                try:
                    verified = validate(OUT.read_bytes())
                    report.update({"status": "VERIFIED", "source_type": "private-author-controlled-cache",
                                   "verified_release_tag": private_tag, **verified})
                    print("OFFICIAL_MDPI_ACS_VERIFIED_FROM_PRIVATE_CACHE", verified)
                    return 0
                except Exception as e:
                    OUT.unlink(missing_ok=True)
                    report["attempts"].append(
                        {"source": "private-authenticated-cache", "error": str(e)[:400]})
                    print("PRIVATE_MDPI_CACHE_INVALID:", type(e).__name__)
        else:
            missing = []
            if not private_repo: missing.append("MDPI_ACS_PRIVATE_REPO")
            if not private_tag: missing.append("MDPI_ACS_PRIVATE_TAG")
            if not token: missing.append("MDPI_ACS_PRIVATE_READ_TOKEN (mapped to GH_TOKEN)")
            report["missing_private_cache_configuration"] = missing
            print("PUBLISHER_UNAVAILABLE_AND_PRIVATE_DEPENDENCY_NOT_CONFIGURED")
        OUT.unlink(missing_ok=True)
        print("BLOCKED: official publisher HTTP 403 or unavailable; "
              "set up private read-only MDPI template release (no public ZIP/class)")
        return 2
    finally:
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                          encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
