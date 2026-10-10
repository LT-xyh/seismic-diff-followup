#!/usr/bin/env python3
"""Complete format-only MDPI wrapper ONLY after an authentic MDPI ACS ZIP is supplied.

Before calling this script, verify the ZIP was downloaded from the publisher's
current official https://www.mdpi.com/authors/latex ACS link. No fallback class.
Scientific body and abstract are existing byte-exact substrings of the original.
"""
import argparse, hashlib, json, re, shutil, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EXPECTED_ORIGINAL_SHA = "51d57b5954516e0040114aee9a3863a4ea5869b78500cde7d4112d6edf4d7e83"


def balanced_argument(s, prefix):
    at = s.find(prefix)
    if at < 0:raise RuntimeError("Original or template missing "+prefix)
    i=at+len(prefix)
    depth=1
    pos=i
    while pos<len(s):
        if s[pos]=="{" and (pos==0 or s[pos-1]!="\\"):depth+=1
        if s[pos]=="}" and (pos==0 or s[pos-1]!="\\"):
            depth-=1
            if depth==0:return s[i:pos]
        pos+=1
    raise RuntimeError("Unclosed LaTeX argument: "+prefix)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--official-zip",required=True)
    ap.add_argument("--publisher-url",required=True)
    ap.add_argument("--verified-last-updated",required=True)
    ap.add_argument("--acknowledge-official-source",action="store_true",required=True)
    args=ap.parse_args()
    if not args.publisher_url.startswith("https://www.mdpi.com/"):
        raise SystemExit("Refuse non-MDPI source for official template")
    if args.verified_last_updated != "2026-09-11":
        raise SystemExit("Official ACS template last-updated date must be reverified")
    src=HERE/"original/pd-bg-rfm.tex"
    source=src.read_text(encoding="utf-8")
    if hashlib.sha256(src.read_bytes()).hexdigest()!=EXPECTED_ORIGINAL_SHA:
        raise SystemExit("Historical source mismatch")
    full_abstract=(HERE/"ABSTRACT_EXACT_FROM_AAAI.tex").read_text(encoding="utf-8")
    full_body=(HERE/"CONTENT_EXACT_FROM_AAAI.tex").read_text(encoding="utf-8")
    assert full_abstract in source and full_body in source
    original_title=balanced_argument(source,r"\title{")

    supplied=Path(args.official_zip).resolve()
    archive_bytes=supplied.read_bytes()
    with zipfile.ZipFile(supplied) as z:
        paths=z.namelist()
        mdpi_cls=next((x for x in paths if x.endswith("Definitions/mdpi.cls")),None)
        template_tex=next((x for x in paths if x.endswith("template.tex")),None)
        mdpi_bst=next((x for x in paths if x.endswith("Definitions/mdpi.bst")),None)
        journalnames=next((x for x in paths if x.endswith("Definitions/journalnames.tex")),None)
        if not all((mdpi_cls,template_tex,mdpi_bst,journalnames)):
            raise SystemExit("Official ACS ZIP missing template/Definitions components")
        cls=z.read(mdpi_cls).decode("utf-8",errors="replace")
        template=z.read(template_tex).decode("utf-8",errors="replace")
        names=z.read(journalnames).decode("utf-8",errors="replace")
        if r"\ProvidesClass{Definitions/mdpi}" not in cls or "remotesensing" not in names:
            raise SystemExit("Official template does not substantiate MDPI Remote Sensing syntax")
        required=(r"\Title{",r"\Author{",r"\abstract{",r"\keyword{")
        for macro in required:
            if macro not in template:raise SystemExit("Official template missing macro "+macro)
        if r"\addhighlights" not in (template+cls):
            raise SystemExit("Official MDPI Highlights macro unavailable; inspect template version manually")
        # Only a clean Definitions subtree is imported; no example science.
        out=HERE/"Definitions"
        if out.exists():raise SystemExit("Definitions directory already exists; refuse overwrite")
        out.mkdir()
        prefix=mdpi_cls.rsplit("Definitions/",1)[0]+"Definitions/"
        for item in paths:
            if item.startswith(prefix) and not item.endswith("/"):
                suffix=item[len(prefix):]
                dest=out/suffix
                if ".." in dest.parts:raise SystemExit("Unsafe official ZIP path")
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(item))
    (HERE/"OFFICIAL_MDPI_TEMPLATE_REFERENCE.tex").write_text(template,encoding="utf-8")
    metadata=r"""
% STEP 1 ONLY. Author identifiers and ethics items remain unconfirmed.
% Source: authentic publisher MDPI ACS template archive; see provenance JSON.
\documentclass[remotesensing,article,submit,moreauthors,pdftex]{Definitions/mdpi}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\newif\ifaaaiwithappendix
\aaaiwithappendixtrue
\newif\ifconditioncontractfigure
\IfFileExists{figures/figure3_condition_learning/figure3_condition_learning.pdf}
  {\conditioncontractfiguretrue}{\conditioncontractfigurefalse}
\Title{""" + original_title + r"""}
\TitleCitation{""" + original_title + r"""}
\Author{Chunlei Wu, Yinghao Xu and Jing Lu*}
\AuthorNames{Chunlei Wu, Yinghao Xu and Jing Lu}
\address{[AFFILIATIONS NOT YET CONFIRMED]}
\corres{Correspondence: [CORRESPONDING EMAIL NOT YET CONFIRMED]}
\abstract{""" + full_abstract + r"""}
\keyword{[KEYWORDS PENDING — FORMAT ONLY]}
\addhighlights{
\textbf{What are the main findings?}
[Pending scientific authors' confirmation]
\textbf{What are the implications of the main findings?}
[Pending scientific authors' confirmation]
}
\bibliographystyle{Definitions/mdpi}
\begin{document}
\input{CONTENT_EXACT_FROM_AAAI}
\end{document}
"""
    (HERE/"main.tex").write_text(metadata,encoding="utf-8")
    props={
        "status":"COMPILED MDPI FORMAT NOT YET VERIFIED",
        "zip_source_url":args.publisher_url,
        "zip_sha256":hashlib.sha256(archive_bytes).hexdigest(),
        "zip_bytes":len(archive_bytes),
        "declared_official_zip_date":args.verified_last_updated,
        "mdpi_cls_header":cls[:700],
        "original_title_exact":original_title,
        "abstract_exact":full_abstract in source,
        "scientific_body_exact":full_body in source,
        "original_scientific_text_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
        "author_metadata_incomplete":True,
    }
    (HERE/"OFFICIAL_MDPI_TEMPLATE_PROVENANCE.json").write_text(
        json.dumps(props,indent=2,ensure_ascii=False)+"\n")
    print("STRUCTURAL_MIGRATION_CREATED. Must compile and audit; NOT automatically journal-ready.")


if __name__=="__main__":main()
