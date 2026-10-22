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
  font-size: 19px;
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
  line-height: 1.46;
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
  mix-blend-mode: multiply;      /* white figure ground takes the paper tone */
}
@media (prefers-color-scheme: dark) {
  figure img { mix-blend-mode: normal; background: var(--mat); padding: .8rem; box-sizing: border-box; }
}
figcaption {
  max-width: var(--measure);
  margin: 1rem auto 0;
  font-size: .84rem;
  line-height: 1.45;
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
/* the 12-column Table 2 may use the whole window, up to ~70rem */
.page > #tbl-main { grid-column: full; padding-inline: 1.1rem; }
#tbl-main .nf-scroll, #tbl-main .nf-tablenotes { max-width: 74rem; margin-inline: auto; }
#tbl-main .nf-scroll { padding-left: .8rem; }
.nf-scroll {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  padding-left: .8rem;             /* room for the vermilion row mark */
  margin-left: -.8rem;
}
table.nf-table {
  width: 100%;
  border-collapse: collapse;
  font-size: .78rem;
  line-height: 1.3;
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
  font-size: .74rem;
  line-height: 1.45;
  color: var(--ink-2);
  margin-top: .6rem;
  font-variant-numeric: lining-nums;
}
.nf-tablenotes p { margin: 0 0 .25em; text-indent: 0; }
.nf-table-caption p { display: inline; }

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
  /* the 12-column Table 2 may use the whole window, up to ~70rem */
.page > #tbl-main { grid-column: full; padding-inline: 1.1rem; }
#tbl-main .nf-scroll, #tbl-main .nf-tablenotes { max-width: 74rem; margin-inline: auto; }
#tbl-main .nf-scroll { padding-left: .8rem; }
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
  /* the 12-column Table 2 may use the whole window, up to ~70rem */
.page > #tbl-main { grid-column: full; padding-inline: 1.1rem; }
#tbl-main .nf-scroll, #tbl-main .nf-tablenotes { max-width: 74rem; margin-inline: auto; }
#tbl-main .nf-scroll { padding-left: .8rem; }
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
}
PREAMBLE_MACROS = {"NF", "Eshaft", "Etip", "Davg", "HD", "rep", "best", "second", "tnote", "quad"}
TEXT_MACROS = {"textbf", "textit", "emph", "texttt", "underline", "textsuperscript", "text", "ref", "href"}
MATH_MACROS = {"mathrm", "text", "tau", "le", "ge", "times", "to", "pm", "Delta", "cdot", "sim", "approx"}
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
    md_text = md_text.replace(r"\FloatBarrier", "")
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
                "--csl=style/institute-of-physics-numeric.csl",
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
    out.write_text(keep_title_hyphens(tidy_mathml(out.read_text(encoding="utf-8"))), encoding="utf-8")


if __name__ == "__main__":
    main()
