#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TL="${TL:-/public/home/xuyinghao/.local/texlive/2026/bin/x86_64-linux}"
PDFLATEX="$TL/pdflatex"
BIBTEX="$TL/bibtex"
MODE="${1:-submission}"

case "$MODE" in
  submission)
    MAIN="paper_submission"
    OUTDIR="_build_submission"
    ;;
  full)
    MAIN="paper"
    OUTDIR="_build_full"
    ;;
  *)
    echo "Usage: $0 [submission|full]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PDFLATEX" ]]; then
  echo "Configured pdflatex binary is missing or not executable: $PDFLATEX" >&2
  exit 127
fi
if [[ ! -x "$BIBTEX" ]]; then
  echo "Configured bibtex binary is missing or not executable: $BIBTEX" >&2
  exit 127
fi

mkdir -p "$OUTDIR"
"$PDFLATEX" -interaction=nonstopmode -halt-on-error -output-directory "$OUTDIR" "$MAIN.tex"
"$BIBTEX" "$OUTDIR/$MAIN"
"$PDFLATEX" -interaction=nonstopmode -halt-on-error -output-directory "$OUTDIR" "$MAIN.tex"
"$PDFLATEX" -interaction=nonstopmode -halt-on-error -output-directory "$OUTDIR" "$MAIN.tex"

echo "Built $OUTDIR/$MAIN.pdf"
