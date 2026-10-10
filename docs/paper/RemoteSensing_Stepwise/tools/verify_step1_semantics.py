#!/usr/bin/env python3
"""Validate that MDPI-only LaTeX commands do not alter AAAI science."""
from pathlib import Path
import hashlib
import re

ROOT = Path(__file__).resolve().parents[1]
old = (ROOT / "CONTENT_EXACT_FROM_AAAI.tex").read_text(encoding="utf-8")
new = (ROOT / "CONTENT_MDPI_LAYOUT.tex").read_text(encoding="utf-8")
first = (ROOT / "original/pd-bg-rfm.tex").read_bytes()

assert hashlib.sha256(first).hexdigest() == (
    "51d57b5954516e0040114aee9a3863a4ea5869b78500cde7d4112d6edf4d7e83")
reversed_layout = new.replace("\\appendixtitles{yes}\n", "")
reversed_layout = reversed_layout.replace("\\appendixstart\n", "")
reversed_layout = reversed_layout.replace(
    "\\begin{adjustwidth}{-\\extralength}{0cm}\n", "")
reversed_layout = reversed_layout.replace("\n\\end{adjustwidth}", "")
reversed_layout = reversed_layout.replace(
    "\\begin{tabular*}{\\linewidth}", "\\begin{tabular*}{\\textwidth}")
assert re.sub(r"\s+", "", reversed_layout) == re.sub(r"\s+", "", old)

def envs(text, name):
    return re.findall(
        r"\\begin\{" + re.escape(name) +
        r"\}([\s\S]*?)\\end\{" + re.escape(name) + r"\}", text)

for name in ("equation", "align", "align*", "tabular", "tabular*",
             "algorithmic"):
    a = envs(old, name)
    b = envs(new, name)
    if name == "tabular*":
        b = [x.replace("\\linewidth", "\\textwidth") for x in b]
    assert [re.sub(r"\s+", "", x) for x in a] == [
        re.sub(r"\s+", "", x) for x in b], name
    print(name, len(a), "blocks unchanged")

for command in ("cite", "label", "ref", "eqref"):
    def occurrences(text):
        return re.findall(r"\\" + command + r"\{([^}]+)\}", text)
    assert occurrences(old) == occurrences(new), command

figs = sorted(set(re.findall(
    r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", old)))
assert len(figs) == 9
historical = ROOT.parent / "RemoteSensing_AAAI_Rebuild"
for f in figs:
    assert (ROOT / f).read_bytes() == (historical / f).read_bytes(), f
assert (ROOT / "references.bib").read_bytes() == (
    ROOT / "original/references.bib").read_bytes()
print("STEP1_SEMANTIC_PRESERVATION_PASS", len(figs), "original figures")
