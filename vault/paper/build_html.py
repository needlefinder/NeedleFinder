#!/usr/bin/env python3
"""Build self-contained HTML companion edition of NeedleFinder 2026 PMB manuscript.

Design (screen edition, mirrors the print edition in style/preamble.tex):
  * EB Garamond text face with old-style proportional figures in prose and
    tabular lining figures in tables; Garamond-Math for MathML; IBM Plex Mono
    for code. All fonts are subset to the characters used and embedded as
    WOFF2 data URIs, so the page is one self-contained file.
  * A typographic grid: a ~66-character text measure with section numbers and
    figure/table labels hung in the left margin on wide screens; figures and
    tables break out of the measure. Collapses to one column on phones.
  * Booktabs-style tables (hairline rules, no zebra), one vermilion accent,
    warm paper tone, prefers-color-scheme dark mode, print stylesheet.
"""
import base64
import io
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
FONT_DIR = ROOT / "style" / "fonts"
TEMPLATE = ROOT / "style" / "nf_template.html"

HTML_CSS = """
<style>
/*@@FONTS@@*/
:root {
  --paper: #FBF9F5;
  --ink: #23201B;
  --ink-2: #625B51;
  --rule: #23201B;
  --hair: #CFC7BA;
  --accent: #B34E00;          /* Okabe-Ito vermilion, darkened for text */
  --mark: #D55E00;            /* Okabe-Ito vermilion, for marks */
  --ours-tint: #FAEFE4;
  --ct-tint: #F0EEE9;
  --mat: transparent;
  --measure: 29.5rem;
  --hang: 7.5rem;
  --serif: "EB Garamond NF", "EB Garamond", "Garamond", "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif;
  --mono: "Plex Mono NF", "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  --math: "Garamond Math NF", "Garamond-Math", "STIX Two Math", "Cambria Math", math;
  color-scheme: light dark;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #191714;
    --ink: #E9E3D8;
    --ink-2: #A69E91;
    --rule: #E9E3D8;
    --hair: #4A453D;
    --accent: #F0A15E;
    --mark: #E57A2E;
    --ours-tint: #33261B;
    --ct-tint: #24221E;
    --mat: #F3EFE7;
  }
}
html {
  font-size: 19.5px;
  background: var(--paper);
  -webkit-text-size-adjust: 100%;
  text-size-adjust: 100%;
}
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 1rem;
  line-height: 1.56;
  font-variant-numeric: oldstyle-nums proportional-nums;
  font-feature-settings: "kern", "liga", "onum", "pnum";
  font-kerning: normal;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  hanging-punctuation: first allow-end last;
}

/* ---------- Grid ---------------------------------------------------- */
.page {
  display: grid;
  grid-template-columns:
    [full-start] minmax(1.1rem, 1fr)
    [wide-start] minmax(0, var(--hang))
    [text-start] min(var(--measure), calc(100vw - 2.2rem))
    [text-end] minmax(0, var(--hang))
    [wide-end] minmax(1.1rem, 1fr)
    [full-end];
  padding: 4.5rem 0 6rem;
}
.page > * { grid-column: text; min-width: 0; }
.page > figure,
.page > .nf-table-wrap { grid-column: wide; }

/* ---------- Title block --------------------------------------------- */
.masthead { margin-bottom: 2.6rem; }
.kicker {
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: .3rem 1.5rem;
  margin: 0 0 3rem;
  padding-bottom: .55rem;
  border-bottom: 0.5px solid var(--hair);
  font-variant-caps: all-small-caps;
  font-variant-numeric: oldstyle-nums;
  letter-spacing: .09em;
  font-size: .82rem;
  color: var(--ink-2);
}
h1.title {
  font-weight: 400;
  font-size: 2.15rem;
  line-height: 1.17;
  letter-spacing: -0.003em;
  margin: 0 0 1.5rem;
  text-wrap: balance;
  font-feature-settings: "kern", "liga", "lnum";
  hyphens: manual;
  font-variant-numeric: lining-nums;
}
.nf-tablenotes p { margin: 0 0 .25em; text-indent: 0; }
.nf-table-caption p { display: inline; }
.masthead p { text-indent: 0; text-align: left; hyphens: manual; }
.authors { margin: 0; font-size: 1.1rem; line-height: 1.45; text-wrap: balance; }
.date { margin: .35rem 0 0; font-style: italic; color: var(--ink-2); }
.date em { font-style: normal; }

/* ---------- Abstract ------------------------------------------------ */
.abstract {
  font-size: .93rem;
  line-height: 1.5;
  padding-bottom: 1.6rem;
  margin-bottom: 1rem;
  border-bottom: 0.5px solid var(--hair);
}
.abstract-label {
  font-size: .82rem; font-weight: 400;
  font-variant-caps: all-small-caps; letter-spacing: .12em;
  margin: 0 0 .7rem;
  color: var(--ink-2);
}
.abstract p { margin: 0 0 .7em; text-indent: 0; }

/* ---------- Running text -------------------------------------------- */
p {
  margin: 0;
  text-align: left;           /* ragged right: browser hyphenation is too unreliable to justify */
  hyphens: auto;
  -webkit-hyphens: auto;
  hyphenate-limit-chars: 6 3 3;
  text-wrap: pretty;
  orphans: 2; widows: 2;
}
p + p { text-indent: 1.25em; }
strong, b { font-weight: 600; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; text-decoration-thickness: .06em; text-underline-offset: .18em; }
.citation a, .citation { color: inherit; }
.citation a:hover { color: var(--accent); }
code {
  font-family: var(--mono);
  font-size: .8em;
  font-variant-numeric: normal;
  letter-spacing: -0.01em;
  overflow-wrap: anywhere;
  hyphens: none;
}
sup, sub { line-height: 0; font-size: .7em; }

/* ---------- Headings: numbers hung in the margin --------------------- */
h1, h2, h3, h4 { font-weight: 400; color: var(--ink); text-wrap: balance; }
.page > h1:not(.title) {
  font-size: 1rem;
  font-variant-caps: all-small-caps;
  letter-spacing: .12em;
  margin: 3.3rem 0 1.1rem;
  font-feature-settings: "kern", "c2sc", "smcp", "onum";
}
.page > h2 {
  font-size: 1.04rem;
  font-style: italic;
  line-height: 1.35;
  margin: 2.1rem 0 .65rem;
}
.page > h3 {
  font-size: 1rem;
  font-style: italic;
  line-height: 1.4;
  margin: 1.6rem 0 .4rem;
}
.header-section-number {
  color: var(--mark);
  font-style: normal;
  font-variant-caps: normal;
  letter-spacing: 0;
  font-feature-settings: "onum", "pnum";
}
h1 + p, h2 + p, h3 + p { text-indent: 0; }
@media (min-width: 52rem) {
  .header-section-number {
    float: left;
    width: 5rem;
    margin-left: -6.25rem;
    text-align: right;
  }
}
@media (max-width: 51.99rem) {
  .header-section-number::after { content: "\\2002"; }
}

/* ---------- Lists --------------------------------------------------- */
ol, ul { padding-left: 1.5em; margin: .6rem 0 .9rem; }
li { margin: .3rem 0; text-align: left; hyphens: auto; -webkit-hyphens: auto; text-wrap: pretty; }
li::marker { color: var(--ink-2); font-feature-settings: "onum"; }
li > p { text-indent: 0; }

/* ---------- Mathematics (MathML set in Garamond-Math) ---------------- */
math { font-family: var(--math); font-size: 1.04em; }
math mn, math mtext {
  font-family: var(--serif);
  font-variant-numeric: oldstyle-nums proportional-nums;
}
math[display="block"] { margin: 1rem 0; overflow-x: auto; overflow-y: hidden; max-width: 100%; }
mtext { white-space: pre; }
.mglue, .nobr { white-space: nowrap; }

/* ---------- Figures -------------------------------------------------- */
.page { counter-reset: fig; }
figure {
  counter-increment: fig;
  margin: 2.4rem 0 2.6rem;
  padding: 0;
  break-inside: avoid;
}
figure img {
  display: block;
  width: 100%;
  height: auto;
  background: #FFFFFF;
  padding: .65rem;
  border-radius: 6px;
  border: 0.5px solid var(--hair);
  box-sizing: border-box;
  mix-blend-mode: normal;        /* preserve crisp scientific colors & MRI contrast */
}
@media (prefers-color-scheme: dark) {
  figure img { mix-blend-mode: normal; background: #FFFFFF; padding: .8rem; box-sizing: border-box; }
}
figcaption {
  max-width: min(52rem, 100%);
  margin: 1rem auto 0;
  font-size: .88rem;
  line-height: 1.5;
  text-align: left;
  hyphens: auto; -webkit-hyphens: auto;
  text-wrap: pretty;
  color: var(--ink);
}
figcaption::before,
.nf-table-caption .nf-label {
  content: "Figure\\00a0" counter(fig);
  font-variant-caps: all-small-caps;
  letter-spacing: .1em;
  font-feature-settings: "c2sc", "smcp", "lnum";
  color: var(--ink-2);
  font-weight: 400;
}
figcaption::before { margin-right: .6em; }

/* ---------- Tables (booktabs) ---------------------------------------- */
.nf-table-wrap { margin: 2.4rem 0 2.6rem; break-inside: avoid; }
.nf-table-caption {
  max-width: var(--measure);
  margin: 0 auto .85rem;
  font-size: .84rem;
  line-height: 1.45;
  text-align: left;
  hyphens: auto; -webkit-hyphens: auto;
  text-wrap: pretty;
}
.nf-table-caption .nf-label { margin-right: .6em; }
@media (min-width: 52rem) {
  figcaption, .nf-table-caption { position: relative; }
  figcaption::before, .nf-table-caption .nf-label {
    position: absolute;
    left: -7.5rem; width: 6.6rem;
    text-align: right;
    margin: 0;
  }
}
/* the 12-column Table 2 and Appendix B ablation table may use the whole window, up to ~74rem */
.page > [id="tbl:main"], .page > [id="tbl:app_ablation"] { grid-column: full; padding-inline: 1.1rem; }
[id="tbl:main"] .nf-scroll, [id="tbl:main"] .nf-tablenotes,
[id="tbl:app_ablation"] .nf-scroll, [id="tbl:app_ablation"] .nf-tablenotes { max-width: 74rem; margin-inline: auto; }
[id="tbl:main"] .nf-scroll, [id="tbl:app_ablation"] .nf-scroll { padding-left: .8rem; }
.nf-scroll {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  padding-left: .8rem;             /* room for the vermilion row mark */
  margin-left: -.8rem;
}
table.nf-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .83rem;
  line-height: 1.35;
  font-variant-numeric: lining-nums tabular-nums;
  font-feature-settings: "kern", "lnum", "tnum";
  border-top: 1.1px solid var(--rule);
  border-bottom: 1.1px solid var(--rule);
}
table.nf-table th {
  font-weight: 400;
  font-style: normal;
  vertical-align: bottom;
  padding: .55em .55em .4em;
  white-space: nowrap;
}
table.nf-table thead tr:last-child th { border-bottom: 0.6px solid var(--rule); }
/* booktabs cmidrule(lr): a trimmed hairline under a column group */
table.nf-table th.cmid {
  background: linear-gradient(var(--rule), var(--rule)) no-repeat center bottom / calc(100% - .9em) 0.6px;
  padding-bottom: .3em;
}
table.nf-table td {
  padding: .28em .55em;
  text-align: right;
  white-space: nowrap;
  vertical-align: baseline;
}
table.nf-table th:first-child, table.nf-table td:first-child { padding-left: 0; }
table.nf-table .al-l { text-align: left; }
table.nf-table .al-r { text-align: right; }
table.nf-table .al-c { text-align: center; }
table.nf-table td.indent-1 { padding-left: 1.1em; }
table.nf-table td.indent-2 { padding-left: 2.2em; }
table.nf-table tr.rule-above td { border-top: 0.6px solid var(--rule); }
table.nf-table tr.space-above td { padding-top: .6em; }
table.nf-table tr.subsection-row td { padding-left: 1.1em; padding-top: .45em; }
table.nf-table th:last-child, table.nf-table td:last-child { padding-right: 0; }
table.nf-table tbody tr:first-child td { padding-top: .5em; }
table.nf-table tbody tr:last-child td { padding-bottom: .5em; }
table.nf-table tr.section-row td {
  font-style: italic;
  text-align: left;
  white-space: normal;
  padding: .85em 0 .25em;
  color: var(--ink);
}
table.nf-table tr.ours-row td { background: var(--ours-tint); }
table.nf-table tr.ours-row td:first-child { position: relative; }
table.nf-table tr.ours-row td:first-child::before {
  content: "";
  position: absolute;
  left: -.6rem; top: .3em; bottom: .3em;
  width: 2px;
  background: var(--mark);
}
table.nf-table tr.ct-row td { background: var(--ct-tint); }
table.nf-table code { font-size: .86em; }
table.nf-table sup { font-size: .72em; }
.best { font-weight: 600; }
table.nf-table strong { font-weight: 600; }
table.nf-table math mn, .nf-tablenotes math mn { font-variant-numeric: lining-nums tabular-nums; }
table.nf-table u, .second { text-decoration: underline; text-decoration-thickness: .055em; text-underline-offset: .2em; }
.nf-tablenotes {
  font-size: .79rem;
  line-height: 1.48;
  color: var(--ink-2);
  margin-top: .6rem;
  font-variant-numeric: lining-nums;
}
.nf-tablenotes p { margin: 0 0 .25em; text-indent: 0; }
.nf-table-caption p { display: inline; }

/* ---------- Interactive 3D Viewer ----------------------------------- */
.nf-3d-widget {
  margin: .9rem 0 .4rem;
  border: 0.5px solid var(--hair);
  border-radius: 6px;
  background: #FFFFFF;
  padding: .65rem .85rem .75rem;
  font-family: var(--serif);
  color: #23201B;
}
.nf-3d-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: .45rem .8rem;
  padding-bottom: .5rem;
  margin-bottom: .5rem;
  border-bottom: 0.5px solid #E2DDD5;
  font-size: .78rem;
}
.nf-3d-title {
  font-weight: 600;
  color: #23201B;
  letter-spacing: .01em;
}
.nf-3d-title span {
  font-family: var(--mono);
  font-size: .72rem;
  font-weight: 400;
  color: #B34E00;
  background: #FAEFE4;
  padding: .1rem .4rem;
  border-radius: 3px;
  margin-left: .4rem;
}
.nf-3d-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: .35rem .65rem;
}
.nf-3d-controls select, .nf-3d-controls button {
  font-family: var(--serif);
  font-size: .76rem;
  padding: .18rem .5rem;
  border: 0.5px solid #CFC7BA;
  border-radius: 4px;
  background: #FBF9F5;
  color: #23201B;
  cursor: pointer;
}
.nf-3d-controls button:hover, .nf-3d-controls select:hover {
  border-color: #B34E00;
  color: #B34E00;
}
.nf-3d-toggles {
  display: flex;
  flex-wrap: wrap;
  gap: .35rem .9rem;
  padding-top: .45rem;
  font-size: .75rem;
  color: #57524A;
}
.nf-3d-toggles label {
  display: inline-flex;
  align-items: center;
  gap: .28rem;
  cursor: pointer;
  user-select: none;
}
.nf-3d-canvas-wrap {
  position: relative;
  width: 100%;
  height: 360px;
  background: radial-gradient(circle at 50% 42%, #FCFBF8 0%, #F4F1EA 100%);
  border: 0.5px solid #E6E1D8;
  border-radius: 4px;
  overflow: hidden;
}
.nf-3d-canvas-wrap canvas {
  display: block;
  width: 100%;
  height: 100%;
  cursor: grab;
}
.nf-3d-canvas-wrap canvas:active { cursor: grabbing; }
.nf-3d-hud {
  position: absolute;
  left: .65rem;
  bottom: .45rem;
  font-family: var(--mono);
  font-size: .68rem;
  color: #625B51;
  background: rgba(251,249,245,0.86);
  padding: .15rem .45rem;
  border-radius: 3px;
  pointer-events: none;
}
@media print {
  .nf-3d-widget { display: none !important; }
}

/* ---------- Graphical Abstract & Collapsible Plain-Language Summary --- */
.nf-ga-card {
  margin: 1.5rem 0 1.2rem;
  padding: .75rem .9rem .85rem;
  border: 0.5px solid var(--hair);
  border-radius: 6px;
  background: #FFFFFF;
}
.nf-ga-label {
  font-variant-caps: all-small-caps;
  letter-spacing: .1em;
  font-size: .78rem;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: .45rem;
}
.nf-ga-card img {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 4px;
}
details.nf-pls {
  margin: 0 0 1.8rem;
  padding: .65rem .95rem;
  border: 0.5px solid var(--hair);
  border-left: 3px solid var(--accent);
  border-radius: 4px;
  background: rgba(179, 78, 0, 0.03);
  font-size: .92rem;
}
details.nf-pls summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: .01em;
}
details.nf-pls p {
  margin: .6rem 0 0;
  line-height: 1.52;
}

/* ---------- References ----------------------------------------------- */
#refs { font-size: .86rem; line-height: 1.42; margin-top: .4rem; }
.csl-entry { padding-left: 1.5em; text-indent: -1.5em; margin: 0 0 .45rem; text-align: left; hyphens: manual; overflow-wrap: anywhere; }
.csl-entry a { overflow-wrap: anywhere; }

/* ---------- Small screens -------------------------------------------- */
@media (max-width: 40rem) {
  html { font-size: 17.5px; }
  .page { padding: 2rem 0 4rem; }
  .kicker { margin-bottom: 2rem; flex-direction: column; }
  h1.title { font-size: 1.65rem; }
  .page > figure, .page > .nf-table-wrap { grid-column: text; }
  p, li, figcaption, .nf-table-caption { text-align: left; }
  /* the 12-column Table 2 and Appendix B ablation table may use the whole window, up to ~74rem */
.page > [id="tbl:main"], .page > [id="tbl:app_ablation"] { grid-column: full; padding-inline: 1.1rem; }
[id="tbl:main"] .nf-scroll, [id="tbl:main"] .nf-tablenotes,
[id="tbl:app_ablation"] .nf-scroll, [id="tbl:app_ablation"] .nf-tablenotes { max-width: 74rem; margin-inline: auto; }
[id="tbl:main"] .nf-scroll, [id="tbl:app_ablation"] .nf-scroll { padding-left: .8rem; }
.nf-scroll { margin-left: 0; padding-left: .7rem; }
}

/* ---------- Print ---------------------------------------------------- */
@media print {
  @page { size: A4; margin: 22mm 20mm 24mm; }
  :root { --paper: #fff; --ink: #000; --ink-2: #444; --ours-tint: #FBF1E7; --ct-tint: #F1EFEB; --mat: transparent; }
  html { font-size: 10.5pt; }
  .page { padding: 0; grid-template-columns: [full-start wide-start] 0 [text-start] 1fr [text-end] 0 [wide-end full-end]; }
  figure img { mix-blend-mode: normal; background: none; padding: 0; }
  h1, h2, h3 { break-after: avoid; }
  figure, .nf-table-wrap { break-inside: avoid; }
  /* the 12-column Table 2 and Appendix B ablation table may use the whole window, up to ~74rem */
.page > [id="tbl:main"], .page > [id="tbl:app_ablation"] { grid-column: full; padding-inline: 1.1rem; }
[id="tbl:main"] .nf-scroll, [id="tbl:main"] .nf-tablenotes,
[id="tbl:app_ablation"] .nf-scroll, [id="tbl:app_ablation"] .nf-tablenotes { max-width: 74rem; margin-inline: auto; }
[id="tbl:main"] .nf-scroll, [id="tbl:app_ablation"] .nf-scroll { padding-left: .8rem; }
.nf-scroll { overflow: visible; }
  a { color: inherit; }
  .header-section-number { float: none; margin: 0; width: auto; }
  .header-section-number::after { content: "\\2002"; }
  figcaption::before, .nf-table-caption .nf-label { position: static; }
}
</style>
"""

# ---------------------------------------------------------------------------
# Tables: single source. The HTML tables are generated at build time from the
# same tables/*.tex files the PDF \input-s. The converter handles exactly the
# LaTeX subset those files use and raises on anything else, so the two
# editions cannot silently diverge.
# ---------------------------------------------------------------------------
class TexTableError(RuntimeError):
    pass


# Macros the converter understands, by where they may appear.
STRUCTURAL_MACROS = {
    "begin", "end", "centering", "caption", "label", "footnotesize", "scriptsize", "small",
    "setlength", "tabcolsep", "toprule", "midrule", "bottomrule", "cmidrule",
    "addlinespace", "multicolumn", "oursrow", "ctrow", "rowcolor", "item",
    "addtocounter",
}
PREAMBLE_MACROS = {"NF", "Eshaft", "Etip", "Davg", "HD", "rep", "best", "second", "tnote", "quad"}
TEXT_MACROS = {"textbf", "textit", "emph", "texttt", "underline", "textsuperscript", "text", "ref", "href"}
MATH_MACROS = {"mathrm", "text", "tau", "le", "ge", "times", "to", "pm", "Delta", "cdot", "sim", "approx", "gamma", "sigma", "rho", "alpha", "hat", "parallel", "perp", "sqrt", "min"}
KNOWN_ENVS = {"table", "threeparttable", "tabular", "tablenotes"}


def _strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def _braced(s, i):
    """s[i] == '{' -> (content, index after the matching '}')."""
    if i >= len(s) or s[i] != "{":
        raise TexTableError(f"expected '{{' at: {s[i:i + 40]!r}")
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "\\":
            continue
        if c == "{" and s[j - 1] != "\\":
            depth += 1
        elif c == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise TexTableError(f"unbalanced braces in: {s[i:i + 60]!r}")


def _command_arg(tex, name):
    m = re.search(r"\\" + name + r"\s*(?=\{)", tex)
    if not m:
        return None, None
    return _braced(tex, m.end())


def _split_top(s, sep):
    """Split on a separator that is not inside braces and not escaped."""
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            two = s[i:i + 2]
            if sep == r"\\" and two == r"\\" and depth == 0:
                out.append("".join(cur))
                cur = []
                i += 2
                continue
            cur.append(two)
            i += 2
            continue
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if sep == "&" and c == "&" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def _check_macros(tex, where):
    allowed = STRUCTURAL_MACROS | PREAMBLE_MACROS | TEXT_MACROS | MATH_MACROS
    for m in re.finditer(r"\\([A-Za-z]+)", tex):
        if m.group(1) not in allowed:
            raise TexTableError(f"{where}: unsupported LaTeX macro \\{m.group(1)} -- extend the converter")
    for env in re.findall(r"\\(?:begin|end)\{([^}]*)\}", tex):
        if env not in KNOWN_ENVS:
            raise TexTableError(f"{where}: unsupported environment {env!r}")


def _expand_preamble_macros(tex, refmap):
    """Expand the preamble macros of style/preamble.tex into plain LaTeX pandoc can read."""
    # a control word without {} swallows the following spaces, exactly as TeX does
    def word(name):
        return r"\\" + name + r"(?![A-Za-z])(?:\{\}|\s*)"

    tex = re.sub(word("NF"), lambda m: r"\textbf{NeedleFinder\,2026}", tex)
    tex = re.sub(word("Eshaft"), lambda m: r"$E_{\mathrm{shaft}}$", tex)
    tex = re.sub(word("Etip"), lambda m: r"$E_{\mathrm{tip}}$", tex)
    tex = re.sub(word("Davg"), lambda m: r"$D_{\mathrm{avg}}$", tex)
    tex = re.sub(word("HD"), lambda m: r"$\mathrm{HD}$", tex)
    # \rep is \textsuperscript{r}; inside math it becomes ^{\mathrm{r}}
    tex = tex.replace(r"$\rep$", r"\textsuperscript{r}")
    parts = re.split(r"(?<!\\)(\$[^$]*\$)", tex)
    for k, p in enumerate(parts):
        if p.startswith("$"):
            parts[k] = re.sub(word("rep"), lambda m: r"^{\mathrm{r}}", p)
        else:
            parts[k] = re.sub(word("rep"), lambda m: r"\textsuperscript{r}", p)
    tex = "".join(parts)
    tex = re.sub(r"\\tnote\{([^}]*)\}", lambda m: r"\textsuperscript{" + m.group(1) + "}", tex)
    tex = re.sub(r"\\best(?=\{)", lambda m: r"\textbf", tex)
    tex = re.sub(r"\\second(?=\{)", lambda m: r"\underline", tex)
    tex = re.sub(r"\\quad(?![A-Za-z])\s*", lambda m: "\u2003", tex)

    def ref(m):
        key = m.group(1)
        if key not in refmap:
            raise TexTableError(f"\\ref{{{key}}}: unknown label")
        return r"\href{#" + key + "}{" + str(refmap[key]) + "}"

    tex = re.sub(r"\\ref\{([^}]*)\}", ref, tex)
    # adjacent inline formulas produced by expansion ($\mathrm{HD}$$_{x}$) -> one formula
    tex = tex.replace("$$", "")
    return tex.strip()


def _col_aligns(spec):
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r">\{[^}]*\}", "", spec)
    aligns = []
    for ch in spec:
        if ch in "LlYX":
            aligns.append("left")
        elif ch in "Rr":
            aligns.append("right")
        elif ch in "Cc":
            aligns.append("center")
        elif ch.isspace() or ch == "|":
            continue
        else:
            raise TexTableError(f"unsupported column type {ch!r} in spec {spec!r}")
    return aligns


ROW_PREFIX = re.compile(
    r"^\s*(\\toprule|\\midrule|\\bottomrule|\\cmidrule(?:\([lr]*\))?\{(\d+)-(\d+)\}"
    r"|\\addlinespace(?:\[[^\]]*\])?|\\oursrow|\\ctrow|\\rowcolor\{(\w+)\})"
)


def parse_tex_table(path, refmap):
    """Parse one tables/*.tex file into a structure with LaTeX fragments still to convert."""
    raw = _strip_comments(path.read_text(encoding="utf-8"))
    _check_macros(raw, path.name)
    label = re.search(r"\\label\{([^}]*)\}", raw)
    if not label:
        raise TexTableError(f"{path.name}: no \\label")
    caption, _ = _command_arg(raw, "caption")
    if caption is None:
        raise TexTableError(f"{path.name}: no \\caption")

    m = re.search(r"\\begin\{tabular\}\s*(?=\{)", raw)
    spec, start = _braced(raw, m.end())
    end = raw.index(r"\end{tabular}", start)
    body = raw[start:end]
    # If the .tex file split a tall table into 2 floats for print pagination, concatenate the second tabular body in HTML
    m2 = re.search(r"\\begin\{tabular\}\s*(?=\{)", raw[end:])
    if m2:
        _, start2 = _braced(raw, end + m2.end())
        end2 = raw.index(r"\end{tabular}", start2)
        body2 = raw[start2:end2]
        # Strip the repeated header block (up to the first \midrule) from the continued tabular
        mid2 = re.search(r"\\midrule", body2)
        if mid2:
            body2 = body2[mid2.end():]
        body = body.rstrip() + "\n\\midrule\n" + body2.lstrip()
    aligns = _col_aligns(spec)
    ncols = len(aligns)

    rows, pending, in_head = [], {"rule": False, "space": False, "cls": set()}, True
    head_rows = []
    for seg in _split_top(body, r"\\"):
        s = seg
        flags = {"rule": False, "space": False, "cls": set(), "cmid": []}
        flags["rule"] = pending["rule"]
        flags["space"] = pending["space"]
        while True:
            mm = ROW_PREFIX.match(s)
            if not mm:
                break
            tok = mm.group(1)
            if tok.startswith(r"\midrule"):
                if in_head:
                    in_head = False
                else:
                    flags["rule"] = True
            elif tok.startswith(r"\cmidrule"):
                (head_rows[-1] if head_rows else rows[-1])["cmid"].append((int(mm.group(2)), int(mm.group(3))))
            elif tok.startswith(r"\addlinespace"):
                flags["space"] = True
            elif tok == r"\oursrow":
                flags["cls"].add("ours-row")
            elif tok == r"\ctrow" or (mm.group(4) in ("ctrow", "nfctrow")):
                flags["cls"].add("ct-row")
            elif tok.startswith(r"\rowcolor"):
                raise TexTableError(f"{path.name}: unsupported \\rowcolor{{{mm.group(4)}}}")
            s = s[mm.end():]
        if not s.strip():
            # a segment holding only rules: carry them to the next row
            pending = {"rule": flags["rule"], "space": flags["space"], "cls": set()}
            if flags["cls"]:
                raise TexTableError(f"{path.name}: row style without a row")
            continue
        pending = {"rule": False, "space": False, "cls": set()}
        cells = []
        col = 0
        for c in _split_top(s, "&"):
            c = c.strip()
            span, align = 1, aligns[col] if col < ncols else "left"
            mc = re.match(r"\\multicolumn\{(\d+)\}", c)
            if mc:
                span = int(mc.group(1))
                mspec, after = _braced(c, mc.end())
                a = _col_aligns(mspec)
                align = a[0] if a else "left"
                inner, after = _braced(c, after)
                if c[after:].strip():
                    raise TexTableError(f"{path.name}: text after \\multicolumn: {c[after:]!r}")
                c = inner.strip()
            indent = 0
            while c.startswith(r"\quad"):
                indent += 1
                c = c[len(r"\quad"):].lstrip()
            cells.append({"tex": c, "span": span, "align": align, "indent": indent})
            col += span
        if col != ncols:
            raise TexTableError(f"{path.name}: row has {col} columns, expected {ncols}: {s.strip()[:80]!r}")
        row = {"cells": cells, **flags}
        if len(cells) == 1 and cells[0]["span"] == ncols:
            row["cls"].add("section-row")
            if cells[0]["indent"]:
                row["cls"].add("subsection-row")
        (head_rows if in_head else rows).append(row)

    notes = []
    mn = re.search(r"\\begin\{tablenotes\}(?:\[[^\]]*\])?(.*?)\\end\{tablenotes\}", raw, re.S)
    if mn:
        txt = re.sub(r"^\s*\\(?:footnotesize|scriptsize)", "", mn.group(1).strip())
        for item in re.split(r"\\item(?![A-Za-z])", txt)[1:]:
            item = item.strip()
            mark = None
            mi = re.match(r"\[([^\]]*)\]\s*", item)
            if mi:
                mark, item = mi.group(1), item[mi.end():]
            notes.append({"mark": mark, "tex": item})

    return {"label": label.group(1), "caption": caption, "aligns": aligns,
            "head": head_rows, "rows": rows, "notes": notes, "file": path.name}


def latex_fragments_to_html(fragments):
    """Convert many LaTeX fragments in ONE pandoc call (MathML for math)."""
    if not fragments:
        return []
    doc = "\n\n".join(f"NFFRAG{i:05d}NF {frag}" for i, frag in enumerate(fragments))
    res = subprocess.run(["pandoc", "-f", "latex", "-t", "html5", "--mathml", "--wrap=none"],
                         input=doc, capture_output=True, text=True, check=True)
    if res.stderr.strip():
        raise TexTableError("pandoc warnings while converting table fragments:\n" + res.stderr)
    found = dict(
        (int(m.group(1)), m.group(2).strip())
        for m in re.finditer(r"<p>NFFRAG(\d{5})NF ?(.*?)</p>", res.stdout, re.S)
    )
    if len(found) != len(fragments):
        missing = sorted(set(range(len(fragments))) - set(found))
        raise TexTableError(f"fragments lost in conversion: {[fragments[i] for i in missing[:3]]}")
    # pandoc renders \, as U+2006 (1/6 em), almost a full word space in Garamond (0.2 em);
    # U+2009 (0.1 em) reads as the thin space TeX produces
    return [found[i].replace("\u2006", "\u2009") for i in range(len(fragments))]


def render_tables(table_paths, refmap):
    """Return {path: html} for every table, converting all fragments in one pandoc call."""
    parsed = [parse_tex_table(p, refmap) for p in table_paths]
    frags = []

    def add(tex, where):
        tex = _expand_preamble_macros(tex, refmap)
        _check_macros(tex, where)
        frags.append(tex)
        return len(frags) - 1

    for t in parsed:
        t["caption_i"] = add(t["caption"], t["file"] + " caption")
        for r in t["head"] + t["rows"]:
            for c in r["cells"]:
                c["i"] = add(c["tex"], t["file"]) if c["tex"] else None
        for n in t["notes"]:
            n["i"] = add(n["tex"], t["file"] + " notes")
    html_frags = latex_fragments_to_html(frags)

    def cell_html(c, tag, cmid=False):
        content = html_frags[c["i"]] if c["i"] is not None else ""
        cls = [f"al-{c['align'][0]}"]
        if c["indent"]:
            cls.append(f"indent-{min(c['indent'], 2)}")
        if cmid:
            cls.append("cmid")
        span = f' colspan="{c["span"]}"' if c["span"] > 1 else ""
        return f'<{tag}{span} class="{" ".join(cls)}">{content}</{tag}>'

    def row_html(r, tag):
        cls = sorted(r["cls"])
        if r["rule"]:
            cls.append("rule-above")
        if r["space"]:
            cls.append("space-above")
        out, col = [], 1
        for c in r["cells"]:
            cm = any(a <= col and col + c["span"] - 1 <= b for a, b in r["cmid"])
            out.append(cell_html(c, tag, cm))
            col += c["span"]
        attr = f' class="{" ".join(cls)}"' if cls else ""
        return f"<tr{attr}>{''.join(out)}</tr>"

    result = {}
    for t, path in zip(parsed, table_paths):
        num = refmap[t["label"]]
        note_ps = []
        for n in t["notes"]:
            mark = "<sup>%s</sup>&#x2009;" % n["mark"] if n["mark"] else ""
            note_ps.append("<p>%s%s</p>" % (mark, html_frags[n["i"]]))
        notes = "".join(note_ps)
        result[path] = (
            f'<div class="nf-table-wrap" id="{t["label"]}">'
            f'<div class="nf-table-caption"><span class="nf-label">Table&nbsp;{num}</span>{html_frags[t["caption_i"]]}</div>'
            f'<div class="nf-scroll"><table class="nf-table nf-cols-{len(t["aligns"])}">'
            f'<thead>{"".join(row_html(r, "th") for r in t["head"])}</thead>'
            f'<tbody>{"".join(row_html(r, "td") for r in t["rows"])}</tbody></table></div>'
            + (f'<div class="nf-tablenotes">{notes}</div>' if notes else "")
            + "</div>"
        )
    return result


def label_numbers(md_text):
    """Number tables and figures in source order, as LaTeX does: {label: n}."""
    refmap, ntab, nfig = {}, 0, 0
    for m in re.finditer(r"\\input\{(tables/[^}]+)\}|\{#(fig:[^\s}]+)", md_text):
        if m.group(1):
            lab = re.search(r"\\label\{([^}]*)\}", (ROOT / m.group(1)).read_text(encoding="utf-8"))
            if not lab:
                raise TexTableError(f"{m.group(1)}: no \\label")
            ntab += 1
            refmap[lab.group(1)] = ntab
        else:
            nfig += 1
            refmap[m.group(2)] = nfig
    return refmap



# ---------------------------------------------------------------------------
# Fonts: subset to what the document uses, embed as WOFF2 data URIs.
# ---------------------------------------------------------------------------
TEXT_FACES = [
    # (css family, file, weight, style)
    ("EB Garamond NF", "EBGaramond-Regular.ttf", 400, "normal"),
    ("EB Garamond NF", "EBGaramond-Italic.ttf", 400, "italic"),
    ("EB Garamond NF", "EBGaramond-SemiBold.ttf", 600, "normal"),
    ("EB Garamond NF", "EBGaramond-SemiBoldItalic.ttf", 600, "italic"),
    ("Plex Mono NF", "IBMPlexMono-Regular.ttf", 400, "normal"),
]
MATH_FACE = ("Garamond Math NF", "Garamond-Math.otf")


def _find_font(name):
    for d in (FONT_DIR, pathlib.Path("/usr/share/fonts/eb-garamond"),
              pathlib.Path("/usr/share/fonts/ibm-plex-mono")):
        p = d / name
        if p.exists():
            return p
    return None


def _subset_woff2(path, unicodes):
    from fontTools import subset
    from fontTools.ttLib import TTFont

    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = ["*"]          # keep smcp, c2sc, onum, lnum, tnum, liga...
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    opts.glyph_names = False
    opts.hinting = False
    font = TTFont(str(path), lazy=False)
    sub = subset.Subsetter(options=opts)
    sub.populate(unicodes=unicodes)
    sub.subset(font)
    # EB Garamond SemiBold(Italic) ships U+2009/U+2006 wider than its word space; fix the advances
    cmap, upm = font.getBestCmap(), font["head"].unitsPerEm
    for cp, em in ((0x2009, 0.10), (0x2006, 0.166), (0x200A, 0.05)):
        g = cmap.get(cp)
        if g and font["hmtx"][g][0] > em * upm * 1.2:
            font["hmtx"][g] = (round(em * upm), font["hmtx"][g][1])
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def font_face_css(doc_text):
    """Return @font-face rules for the embedded faces ('' if fonttools is unavailable)."""
    try:
        import fontTools  # noqa: F401
        import brotli  # noqa: F401  (needed for WOFF2)
    except ImportError:
        print("build_html: fontTools/brotli missing; falling back to installed/system fonts")
        return ""
    base = set(range(0x20, 0x7F)) | set(range(0xA0, 0x180)) | set(range(0x2010, 0x2060))
    base |= {0x2212, 0x00D7, 0x00B1, 0x2264, 0x2265, 0x2192, 0x2248, 0x00B7, 0x2009, 0x202F}
    used = base | {ord(c) for c in doc_text}
    rules = []
    for family, fname, weight, style in TEXT_FACES:
        p = _find_font(fname)
        if p is None:
            continue
        data = _subset_woff2(p, used)
        rules.append(
            f'@font-face{{font-family:"{family}";src:url(data:font/woff2;base64,{data}) format("woff2");'
            f"font-weight:{weight};font-style:{style};font-display:swap;}}"
        )
    p = _find_font(MATH_FACE[1])
    if p is not None:
        math_uni = used | set(range(0x370, 0x400)) | set(range(0x2100, 0x2C00)) | set(range(0x1D400, 0x1D800))
        data = _subset_woff2(p, math_uni)
        rules.append(
            f'@font-face{{font-family:"{MATH_FACE[0]}";src:url(data:font/woff2;base64,{data}) format("woff2");'
            "font-weight:400;font-style:normal;font-display:swap;}"
        )
    return "\n".join(rules)


def tidy_mathml(html):
    # texmath renders \text{ mm} as <mspace 0.333em/> + "<mtext> mm</mtext>": drop the doubled space
    html = re.sub(r'<mspace width="0\.333em"></mspace>(<mtext[^>]*>) ', r"\1&#x2009;", html)
    # texmath marks every matched ( ) [ ] as stretchy; in running text that inflates the
    # fences around subscripted arguments. Keep stretchy fences for display math only.
    html = re.sub(
        r'<math display="inline".*?</math>',
        lambda m: m.group(0).replace('stretchy="true"', 'stretchy="false"'),
        html,
        flags=re.S,
    )
    # a relation that opens a formula ($\\le 3.0$ mm) has no left operand: drop its left space
    html = re.sub(r'(<math display="inline"[^>]*><semantics>(?:<mrow>)?)<mo>([≤≥<>=±])</mo>',
                  r'\1<mo lspace="0" rspace="0.15em">\2</mo>', html)
    # glue an opening bracket and trailing punctuation to inline formulas so a line
    # never ends on "(" or starts with "," / ")"
    return re.sub(
        r'([(\[]?)(<math display="inline".*?</math>)([)\],.;:]*)',
        lambda m: (f'<span class="mglue">{m.group(1)}{m.group(2)}{m.group(3)}</span>'
                   if (m.group(1) or m.group(3)) else m.group(0)),
        html,
        flags=re.S,
    )


def inject_interactive_3d_viewer(html):
    """Embed interactive HTML5 Canvas 3D catheter-bundle & conformal-sphere viewer inside #fig:qualitative."""
    json_path = ROOT.parent / "reports" / "figures" / "data" / "qualitative_3d_viewer.json"
    if not json_path.exists():
        return html
    viewer_json = json_path.read_text(encoding="utf-8").strip()
    widget = f"""
<div class="nf-3d-widget" aria-label="Interactive 3D catheter bundle and split-conformal sphere viewer">
  <div class="nf-3d-bar">
    <div class="nf-3d-title">Interactive 3D Anatomy, Catheter Bundle &amp; 90% Conformal Tip Sphere Explorer <span>Drag to rotate · Scroll to zoom</span></div>
    <div class="nf-3d-controls">
      <select id="nf3d-case" aria-label="Select clinical test case">
        <option value="dense">Case 017 — Dense 26-Catheter Implant (23 Predicted)</option>
        <option value="adhesion">Case 031 — Severe Adhesion (2.2 mm Touching Pair)</option>
      </select>
      <button type="button" data-view="iso">Isometric 3D</button>
      <button type="button" data-view="ap">AP Coronal</button>
      <button type="button" data-view="lat">Sagittal</button>
      <button type="button" data-view="ax">Axial Top</button>
    </div>
  </div>
  <div class="nf-3d-canvas-wrap">
    <canvas id="nf3d-canvas" width="920" height="360"></canvas>
    <div class="nf-3d-hud" id="nf3d-hud">Azimuth: -38° · Elevation: 21° · Zoom: 1.00× · 90% Split-Conformal Spheres (r̂₉₀ = q̂₉₀σ̂_tip)</div>
  </div>
  <div class="nf-3d-toggles">
    <label><input type="checkbox" id="nf3d-gt" checked> Expert Reference (Charcoal Dashed)</label>
    <label><input type="checkbox" id="nf3d-pred" checked> NeedleFinder 2026 3D Tubes (Vermilion)</label>
    <label><input type="checkbox" id="nf3d-conf" checked> 90% Split-Conformal Tip Spheres (r̂₉₀; Magenta = Top-2 Review Targets)</label>
    <label><input type="checkbox" id="nf3d-anat" checked> 3D HR-CTV Envelope, Obturator &amp; Perineal Template Grid</label>
  </div>
</div>
<script>
(function(){{
  const DATA = {viewer_json};
  const canvas = document.getElementById('nf3d-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const hud = document.getElementById('nf3d-hud');
  const caseSel = document.getElementById('nf3d-case');
  const chkGt = document.getElementById('nf3d-gt');
  const chkPred = document.getElementById('nf3d-pred');
  const chkConf = document.getElementById('nf3d-conf');
  const chkAnat = document.getElementById('nf3d-anat');

  let az = -38 * Math.PI / 180;
  let el = 21 * Math.PI / 180;
  let zoom = 1.0;
  let dragging = false, lastX = 0, lastY = 0;

  function project(x, y, z, w, h) {{
    const ca = Math.cos(az), sa = Math.sin(az);
    const ce = Math.cos(el), se = Math.sin(el);
    const xr = x * ca - y * sa;
    const yr = x * sa + y * ca;
    const depth = yr * ce - z * se;
    const up = yr * se + z * ce;
    const persp = 260 / Math.max(90, 260 + depth * 0.45);
    const scale = Math.min(w, h) * 0.0108 * zoom * persp;
    return [w * 0.5 + xr * scale, h * 0.54 - up * scale, depth, scale];
  }}

  function render() {{
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {{
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
    }}
    ctx.save();
    ctx.scale(dpr, dpr);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const ds = DATA[caseSel.value] || DATA.dense;

    // 1. Optional 3D anatomy: perineal template floor, central obturator, HR-CTV wireframe
    if (chkAnat.checked) {{
      ctx.strokeStyle = 'rgba(130, 122, 110, 0.35)';
      ctx.lineWidth = 0.8;
      for (let g = -30; g <= 30; g += 10) {{
        const p1 = project(g, -30, -32, w, h), p2 = project(g, 30, -32, w, h);
        ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
        const q1 = project(-30, g, -32, w, h), q2 = project(30, g, -32, w, h);
        ctx.beginPath(); ctx.moveTo(q1[0], q1[1]); ctx.lineTo(q2[0], q2[1]); ctx.stroke();
      }}
      // Central vaginal obturator cylinder
      for (const zc of [-30, -18, -6, 6]) {{
        ctx.beginPath();
        for (let i = 0; i <= 36; i++) {{
          const th = (i / 36) * Math.PI * 2;
          const pt = project(7.5 * Math.cos(th), 7.5 * Math.sin(th), zc, w, h);
          if (i === 0) ctx.moveTo(pt[0], pt[1]); else ctx.lineTo(pt[0], pt[1]);
        }}
        ctx.strokeStyle = 'rgba(230, 159, 0, 0.42)';
        ctx.lineWidth = 1.0;
        ctx.stroke();
      }}
      // HR-CTV target ellipsoid wireframe
      for (const zc of [-10, 2, 14, 24]) {{
        const rxy = 26 * Math.sqrt(Math.max(0.15, 1 - Math.pow((zc - 8) / 26, 2)));
        ctx.beginPath();
        for (let i = 0; i <= 48; i++) {{
          const th = (i / 48) * Math.PI * 2;
          const pt = project(rxy * Math.cos(th), rxy * 0.85 * Math.sin(th), zc, w, h);
          if (i === 0) ctx.moveTo(pt[0], pt[1]); else ctx.lineTo(pt[0], pt[1]);
        }}
        ctx.strokeStyle = 'rgba(204, 121, 167, 0.38)';
        ctx.lineWidth = 0.9;
        ctx.stroke();
      }}
    }}

    // 2. Expert Reference trajectories (Charcoal dashed)
    if (chkGt.checked) {{
      ctx.save();
      ctx.strokeStyle = 'rgba(55, 52, 46, 0.65)';
      ctx.lineWidth = 1.35;
      ctx.setLineDash([4, 3]);
      for (const trk of ds.gt_tracks) {{
        ctx.beginPath();
        trk.forEach((p, idx) => {{
          const pr = project(p[0], p[1], p[2], w, h);
          if (idx === 0) ctx.moveTo(pr[0], pr[1]); else ctx.lineTo(pr[0], pr[1]);
        }});
        ctx.stroke();
      }}
      ctx.restore();
    }}

    // 3. Depth-sorted NeedleFinder 2026 3D tubes & 90% conformal tip spheres
    const items = ds.pred_tracks.map((trk, idx) => {{
      const mid = trk[Math.floor(trk.length / 2)];
      const pr = project(mid[0], mid[1], mid[2], w, h);
      return {{ trk, idx, depth: pr[2], r90: (ds.conformal_r90_mm && ds.conformal_r90_mm[idx]) || 1.9 }};
    }}).sort((a, b) => b.depth - a.depth);

    for (const item of items) {{
      const pts = item.trk.map(p => project(p[0], p[1], p[2], w, h));
      const isHighlight = item.r90 > 2.5;
      if (chkPred.checked) {{
        // Outer 3D cylinder shadow
        ctx.beginPath();
        pts.forEach((p, i) => {{ if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }});
        ctx.strokeStyle = isHighlight ? '#882E62' : '#8A3A00';
        ctx.lineWidth = 3.4;
        ctx.lineCap = 'round';
        ctx.stroke();
        // Main tube body
        ctx.beginPath();
        pts.forEach((p, i) => {{ if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }});
        ctx.strokeStyle = isHighlight ? '#CC79A7' : '#D55E00';
        ctx.lineWidth = 2.3;
        ctx.stroke();
        // Specular highlight
        ctx.beginPath();
        pts.forEach((p, i) => {{ if (i === 0) ctx.moveTo(p[0] - 0.5, p[1] - 0.5); else ctx.lineTo(p[0] - 0.5, p[1] - 0.5); }});
        ctx.strokeStyle = 'rgba(255, 232, 210, 0.75)';
        ctx.lineWidth = 0.75;
        ctx.stroke();
      }}
      if (chkConf.checked && pts.length > 0) {{
        const tip = pts[pts.length - 1];
        const radPx = Math.max(4.5, item.r90 * tip[3] * 1.35);
        const grad = ctx.createRadialGradient(tip[0] - radPx * 0.3, tip[1] - radPx * 0.3, radPx * 0.1, tip[0], tip[1], radPx);
        if (isHighlight) {{
          grad.addColorStop(0, 'rgba(255, 235, 246, 0.92)');
          grad.addColorStop(0.55, 'rgba(204, 121, 167, 0.55)');
          grad.addColorStop(1, 'rgba(136, 46, 98, 0.82)');
        }} else {{
          grad.addColorStop(0, 'rgba(255, 240, 224, 0.88)');
          grad.addColorStop(0.6, 'rgba(213, 94, 0, 0.42)');
          grad.addColorStop(1, 'rgba(150, 62, 0, 0.72)');
        }}
        ctx.beginPath();
        ctx.arc(tip[0], tip[1], radPx, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.strokeStyle = isHighlight ? '#882E62' : '#A64200';
        ctx.lineWidth = isHighlight ? 1.2 : 0.8;
        ctx.stroke();
      }}
    }}

    // 4. 3D coordinate triad in bottom-right corner
    const ox = w - 48, oy = h - 34, alen = 22;
    const o0 = project(0, 0, 0, w, h);
    const axes = [
      {{ dx: 12, dy: 0, dz: 0, col: '#D55E00', lbl: 'X' }},
      {{ dx: 0, dy: 12, dz: 0, col: '#0072B2', lbl: 'Y' }},
      {{ dx: 0, dy: 0, dz: 12, col: '#009E73', lbl: 'Z' }}
    ];
    for (const a of axes) {{
      const p = project(a.dx, a.dy, a.dz, w, h);
      const vx = p[0] - o0[0], vy = p[1] - o0[1];
      const n = Math.hypot(vx, vy) || 1;
      const ex = ox + (vx / n) * alen, ey = oy + (vy / n) * alen;
      ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ex, ey);
      ctx.strokeStyle = a.col; ctx.lineWidth = 1.6; ctx.stroke();
      ctx.fillStyle = a.col; ctx.font = '600 10px sans-serif';
      ctx.fillText(a.lbl, ex + (vx / n) * 5 - 3, ey + (vy / n) * 5 + 3);
    }}

    ctx.restore();
    const azDeg = Math.round((az * 180 / Math.PI) % 360);
    const elDeg = Math.round(el * 180 / Math.PI);
    hud.textContent = `Azimuth: ${{azDeg}}° · Elevation: ${{elDeg}}° · Zoom: ${{zoom.toFixed(2)}}× · ${{ds.label}}`;
  }}

  canvas.addEventListener('pointerdown', e => {{
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  }});
  canvas.addEventListener('pointermove', e => {{
    if (!dragging) return;
    az += (e.clientX - lastX) * 0.009;
    el = Math.max(-Math.PI / 2.1, Math.min(Math.PI / 2.1, el + (e.clientY - lastY) * 0.009));
    lastX = e.clientX; lastY = e.clientY;
    render();
  }});
  canvas.addEventListener('pointerup', () => {{ dragging = false; }});
  canvas.addEventListener('wheel', e => {{
    e.preventDefault();
    zoom = Math.max(0.55, Math.min(2.4, zoom * (e.deltaY < 0 ? 1.08 : 0.92)));
    render();
  }}, {{ passive: false }});

  [caseSel, chkGt, chkPred, chkConf, chkAnat].forEach(el => el.addEventListener('change', render));
  document.querySelectorAll('.nf-3d-controls button[data-view]').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const v = btn.getAttribute('data-view');
      if (v === 'iso') {{ az = -38 * Math.PI / 180; el = 21 * Math.PI / 180; zoom = 1.0; }}
      else if (v === 'ap') {{ az = 0; el = 0; zoom = 1.05; }}
      else if (v === 'lat') {{ az = -Math.PI / 2; el = 0; zoom = 1.05; }}
      else if (v === 'ax') {{ az = 0; el = Math.PI / 2.05; zoom = 1.05; }}
      render();
    }});
  }});
  window.addEventListener('resize', render);
  setTimeout(render, 60);
}})();
</script>
"""
    return re.sub(
        r'(<figure[^>]*id="fig:qualitative"[^>]*>.*?)(<figcaption>)',
        lambda m: m.group(1) + widget + m.group(2),
        html,
        count=1,
        flags=re.S,
    )


def keep_title_hyphens(html):
    """Hyphenated compounds in the title never break at the hyphen."""
    def fix(m):
        return re.sub(r"(\S*\w-\w\S*)", r'<span class="nobr">\1</span>', m.group(0))
    return re.sub(r'<h1 class="title">.*?</h1>', fix, html, count=1, flags=re.S)


def main():
    md_text = (ROOT / "manuscript.md").read_text(encoding="utf-8")
    refmap = label_numbers(md_text)
    inputs = re.findall(r"\\input\{(tables/[^}]+)\}", md_text)
    tables = render_tables([ROOT / p for p in inputs], refmap)
    for p in inputs:
        # a raw-HTML fenced block passes the table through untouched by the markdown reader
        block = "\n\n```{=html}\n" + tables[ROOT / p] + "\n```\n\n"
        md_text = md_text.replace("\\input{" + p + "}", block)
    md_text = md_text.replace(r"\FloatBarrier", "").replace(r"\clearpage", "").replace(r"\appendix", "")
    # Replace .pdf figure paths with .png for browser rendering
    md_text = re.sub(r"(\.\./reports/figures/[a-zA-Z0-9_]+)\.pdf", r"\1.png", md_text)

    # \ref{tbl:..} / \ref{fig:..} -> linked numbers, from the order of labels in the source
    def ref(m):
        if m.group(1) not in refmap:
            raise TexTableError(f"manuscript.md: \\ref{{{m.group(1)}}} has no matching label")
        return f"[{refmap[m.group(1)]}](#{m.group(1)})"

    md_text = re.sub(r"\\ref\{([^}]*)\}", ref, md_text)
    leftover = re.findall(r"\\(?:input|ref|label)\{[^}]*\}", md_text)
    if leftover:
        raise TexTableError(f"unconverted LaTeX left in HTML source: {leftover[:5]}")

    css = HTML_CSS.replace("/*@@FONTS@@*/", font_face_css(md_text))

    tmp_md = ROOT / "_manuscript_html_tmp.md"
    tmp_css = ROOT / "_style_html_tmp.html"
    out = ROOT / "needlefinder_2026_pmb.html"
    tmp_md.write_text(md_text, encoding="utf-8")
    tmp_css.write_text(css, encoding="utf-8")

    try:
        subprocess.run(
            [
                "pandoc",
                str(tmp_md),
                "--from",
                "markdown+tex_math_dollars+raw_html+fenced_divs",
                "--to",
                "html5",
                "--standalone",
                "--template",
                str(TEMPLATE),
                "--embed-resources",
                "--number-sections",
                "--citeproc",
                "--bibliography=references.bib",
                "--csl=style/elsevier-harvard.csl",
                "-M",
                "link-citations=true",
                "--mathml",
                "-H",
                str(tmp_css),
                "-o",
                str(out),
            ],
            cwd=str(ROOT),
            check=True,
        )
    finally:
        tmp_md.unlink(missing_ok=True)
        tmp_css.unlink(missing_ok=True)
    final_html = inject_interactive_3d_viewer(keep_title_hyphens(tidy_mathml(out.read_text(encoding="utf-8"))))
    ga_path = ROOT / "graphical_abstract.png"
    if ga_path.exists():
        import base64
        ga_b64 = base64.b64encode(ga_path.read_bytes()).decode("ascii")
        ga_and_pls_html = f"""
<div class="nf-ga-card" id="graphical-abstract">
  <div class="nf-ga-label">Graphical Abstract</div>
  <img src="data:image/png;base64,{ga_b64}" alt="NeedleFinder 2026 Graphical Abstract">
</div>
<details class="nf-pls">
  <summary>Plain-Language Summary (for patients, clinicians, and general readers)</summary>
  <p>In interstitial gynecologic brachytherapy, doctors guide 15 to 30 thin (2 mm) plastic catheters into a pelvic tumor while the patient is under anesthesia and take a 3T T2-weighted MRI scan. Because plastic tubes contain no mobile water protons, they appear only as tiny dark dots surrounded by dark blood vessels and tissue borders, requiring 45 to 50 minutes of manual slice-by-slice tracing before radiation can be planned.</p>
  <p><strong>NeedleFinder 2026</strong> automates this reconstruction in <strong>4.15 seconds</strong> on a workstation CPU inside free <strong>3D Slicer</strong> software by combining three physical and geometric steps: (1) an analytical 3D circular dark-core filter (Stage 0) that separates round tube cross-sections from one-sided muscle borders and automatically finds the base template plane; (2) a native-resolution 3D U-Net (Stage 1) pre-trained on 51 external pelvic MRI scans and fine-tuned on two independent expert teams' traces to pinpoint sub-pixel tube centers even when two tubes touch; and (3) message-passing graph networks, a whole-bundle graph transformer, and a 1D sub-slice tip localizer with calibrated 3D split-conformal uncertainty spheres (Stage 2). Across 54 clinical patients (1,063 catheters), NeedleFinder 2026 matches human expert agreement (0.47 mm median shaft error, 2.26 mm median tip error), and checking the two most uncertain tips per patient (~18 s modeled) brings simulated tumor dose within 2.8% median of manual planning.</p>
</details>
"""
        final_html = re.sub(
            r"(<p><strong>Keywords:</strong>.*?</p>)",
            lambda m: m.group(1) + ga_and_pls_html,
            final_html,
            count=1,
            flags=re.S,
        )
    # Replace numeric section numbers on Appendix A..D with hung vermilion letters A..D
    for num_s, let_s in (("8", "A"), ("9", "B"), ("10", "C"), ("11", "D")):
        final_html = re.sub(
            rf'(<span class="header-section-number">){num_s}(</span>\s*Appendix\s+{let_s}\b)',
            rf"\1{let_s}\2",
            final_html,
        )
    out.write_text(final_html, encoding="utf-8")
    (ROOT / "needlefinder_2026_media.html").write_text(final_html, encoding="utf-8")
    (ROOT / "needlefinder_2026_pmb_v2.html").write_text(final_html, encoding="utf-8")


if __name__ == "__main__":
    main()
