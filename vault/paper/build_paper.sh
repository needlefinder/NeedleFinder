#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

echo "[1/3] Generating vector PDF/PNG/SVG figures via generate_pmb_figures.py..."
/usr/bin/python3 generate_pmb_figures.py

echo "[2/3] Generating LaTeX manuscript (needlefinder_2026_pmb.tex) via Pandoc..."
pandoc manuscript.md \
  --from markdown+tex_math_dollars+raw_tex+fenced_divs \
  --to latex \
  --standalone \
  --number-sections \
  --citeproc \
  --bibliography=references.bib \
  -H style/preamble.tex \
  -V documentclass=article \
  -V classoption=10pt \
  -V geometry:a4paper,textwidth=170mm,textheight=242mm,centering \
  -o needlefinder_2026_pmb.tex

echo "[3/3] Compiling PDF manuscript (needlefinder_2026_pmb.pdf)..."
if command -v tectonic >/dev/null 2>&1; then
  PDF_ENGINE="tectonic"
elif command -v xelatex >/dev/null 2>&1; then
  PDF_ENGINE="xelatex"
elif command -v pdflatex >/dev/null 2>&1; then
  PDF_ENGINE="pdflatex"
else
  echo "Error: No LaTeX engine (tectonic, xelatex, pdflatex) found." >&2
  exit 1
fi

pandoc manuscript.md \
  --from markdown+tex_math_dollars+raw_tex+fenced_divs \
  --pdf-engine="$PDF_ENGINE" \
  --number-sections \
  --citeproc \
  --bibliography=references.bib \
  -H style/preamble.tex \
  -V documentclass=article \
  -V classoption=10pt \
  -V geometry:a4paper,textwidth=170mm,textheight=242mm,centering \
  -o needlefinder_2026_pmb.pdf

echo "[4/4] Building self-contained HTML companion report (needlefinder_2026_pmb.html)..."
/usr/bin/python3 build_html.py

echo "Build complete:"
ls -lh "$SCRIPT_DIR/needlefinder_2026_pmb.tex" "$SCRIPT_DIR/needlefinder_2026_pmb.pdf" "$SCRIPT_DIR/needlefinder_2026_pmb.html"
