#!/usr/bin/env python3
"""Compatibility entry point for Step-1 official MDPI template installation.

The format-only manuscript main.tex and CONTENT_MDPI_LAYOUT.tex are already
tracked on remote-sensing/step1-mdpi-format. Use the verified installer to
supply the official publisher class files; do not regenerate scientific text.

Usage:
  python docs/paper/RemoteSensing_Stepwise/tools/assemble_only_from_official_mdpi_zip.py --official-zip MDPI_template_ACS.zip
"""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-zip", required=True)
    a = p.parse_args()
    verified_installer = Path(__file__).with_name("install_uploaded_mdpi_acs.py")
    cmd = [sys.executable, str(verified_installer), a.official_zip]
    subprocess.run(cmd, check=True)
    print("Official class assets installed. Compile main.tex with pdfLaTeX and BibTeX.")

if __name__ == "__main__":
    main()
