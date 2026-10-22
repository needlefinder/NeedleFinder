#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"

# Shared pandoc -> LaTeX options (print edition design lives in style/preamble.tex).
# A4, 11 pt EB Garamond on a 120 mm measure (~68 characters per line);
# the 160 mm block is the float width; body text is inset 20 mm per side (\nfbreak).
LATEX_OPTS=(
  --number-sections
  --citeproc
  --bibliography=references.bib
  --csl=style/institute-of-physics-numeric.csl
  -H style/preamble.tex
  -V documentclass=article
  -V fontsize=11pt
  -V indent=true
  -V colorlinks=true
  -V linkcolor=nfaccent
  -V citecolor=nfink
  -V urlcolor=nfaccent
  -V "geometry:a4paper,textwidth=160mm,textheight=232mm,top=31mm,hcentering,headsep=9mm,footskip=13mm"
)
# The preamble uses fontspec/unicode-math (EB Garamond, Garamond-Math, IBM Plex
# Mono from the TeX distribution), so a Unicode engine is required.

echo "[1/3] Generating vector PDF/PNG/SVG figures via generate_pmb_figures.py..."
/usr/bin/python3 generate_pmb_figures.py

echo "[2/3] Generating LaTeX manuscript (needlefinder_2026_pmb.tex) via Pandoc..."
pandoc manuscript.md \
  --from markdown+tex_math_dollars+raw_tex+fenced_divs \
  --to latex \
  --standalone \
  "${LATEX_OPTS[@]}" \
  -o needlefinder_2026_pmb.tex

echo "[3/3] Compiling PDF manuscript (needlefinder_2026_pmb.pdf)..."
if command -v tectonic >/dev/null 2>&1; then
  PDF_ENGINE="tectonic"
elif command -v xelatex >/dev/null 2>&1; then
  PDF_ENGINE="xelatex"
else
  echo "Error: No Unicode LaTeX engine (tectonic, xelatex) found." >&2
  exit 1
fi

pandoc manuscript.md \
  --from markdown+tex_math_dollars+raw_tex+fenced_divs \
  --pdf-engine="$PDF_ENGINE" \
  "${LATEX_OPTS[@]}" \
  -o needlefinder_2026_pmb.pdf

echo "[4/4] Building self-contained HTML companion report (needlefinder_2026_pmb.html)..."
/usr/bin/python3 build_html.py

echo "Build complete:"
ls -lh "$SCRIPT_DIR/needlefinder_2026_pmb.tex" "$SCRIPT_DIR/needlefinder_2026_pmb.pdf" "$SCRIPT_DIR/needlefinder_2026_pmb.html"
