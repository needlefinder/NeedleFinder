---
title: "2026-10-22 NeedleFinder 2026 PMB Scientific Design System & Visual Architecture Specification"
type: plan
category: publication
created: 2026-10-22
updated: 2026-10-22
sources:
  - paper/manuscript.md
  - paper/build_paper.sh
  - plans/2026-10-22-needlefinder-2026-manuscript-draft.md
  - "arXiv:2607.11290 (Xiao et al. 2026), used only as a visual-style reference; page not re-read for this spec"
tags: [manuscript, pmb, design-system, figures, tables, latex, booktabs, siunitx, okabe-ito, raincloud, pareto]
---

# NeedleFinder 2026 · PMB Scientific Design System & Visual Architecture Specification

This spec fixes the two complaints about the first draft PDF (`paper/needlefinder_2026_pmb.pdf`):

1. **Hard-to-read tables.** Pandoc turned pipe tables into longtables with a wrapped first column and page splits.
2. **Unclear diagrams.** Fig 1 boxes were full of bullet text, and the Fig 2 Pareto plot had overlapping labels.

Visual reference: Xiao *et al.* 2026 (arXiv:2607.11290), which has a schematic overview, a module diagram with math callouts, rainclouds comparing the whole cohort with an adhesion subgroup, and booktabs tables split into Part A and Part B.

Numbers used as worked examples come from `paper/manuscript.md` as of 2026-10-22. Anything not yet measured is shown as `\TBD` and listed in §0. **Never replace a `\TBD` with an estimate.**

---

## 0. Data dependencies & blockers (resolve before drawing)

The requested figure suite needs several results that **do not exist yet** in the vault or manuscript. The design handles each one explicitly; the design must not hide them.

| # | Needed for | Status on 2026-10-22 | Design handling until resolved |
|---|---|---|---|
| D1 | **Catheter-adhesion subgroup** (Fig 3 row 2, Table 2 Part A.2) | No definition and no labels exist. Xiao's definition has not been checked: the arXiv fetch was not permitted in this session. | Proposed definition: a GT catheter is "adhesion" if its GT centerline comes within **≤ 3.0 mm** of another GT centerline on **≥ 3 consecutive axial slices**. Make this a frozen script in `NeedleFinder/` and report the subgroup's *n* (patients / catheters). |
| D2 | **nnU-Net + clDice baseline** | It appears only as related work in `wiki/research/2026-10-21-sota-nn-hybrid-needle-detection.md`. It has not been trained. | It must be trained on the **same 5 folds** with the same instance post-processing (connected components → centerline). Until then: Table row = `\TBD`, and it is absent from Fig 3. |
| D3 | **PMB 2019 baseline, per catheter** | Only reported aggregates are available: voxel recall/precision, and *D*<sub>avg</sub> = 2.00 ± 3.40 mm on a different 50-patient split. | Either re-run the 2019 3D U-Net on our folds (preferred), or show it as a **reported-only marker** (hollow symbol, §3.0 R7) and never as a distribution. |
| D4 | **Xiao 2026 baseline** | Our draft lists it as **3D CT**, Hough + physics tracking, F1 89.0 %, 0.33 mm shaft (0.90 mm tip), 11.6 s. **Check the modality against the arXiv paper**: if it is CT-only, it cannot be "benchmarked against" on our T2W MRI without reimplementation. | Option A: reimplement Sync-Track on our MRI folds and label it "Xiao 2026 (re-impl.)". Option B: reported-only marker labelled "CT, reported". Whichever is chosen, keep the cross-modality caveat in the caption. |
| D5 | **Few-shot annotation-efficiency curve** (Fig 4b: 1, 5, 10, 20, 43 training cases) | Not run. | Needs ≥ 3 random case subsets per budget, evaluated on the fixed OOF test folds. With 54 patients and 5 folds, ≈ 43 is the full training fold. |
| D6 | **Per-catheter E<sub>tip</sub>, D<sub>avg</sub>, E<sub>shaft</sub> P95 on the whole cohort** | The manuscript has the mean and median of E<sub>shaft</sub> for 1,063 catheters. P95 and per-catheter arrays are reported only for the 40/41-scene dual-observer subset. | Export per-catheter long-format CSV (§3.0 data contract). |
| D7 | **PMB 2019 "90 s" runtime** | Not in the manuscript or vault. | Find the source (paper or re-timed run) before it goes on Fig 6. |
| D8 | **"IRGM" (Implant-Region Gating Module)** | This name does **not** appear in the manuscript. The manuscript describes a "5-head anisotropic 3D ResUNet + SE" (`NeedleUNet3D`) and a Stage 0 symmetry gate. | Decide whether IRGM is (a) a renaming of the Stage 0 → Stage 1 gating or (b) a new module. Define it once in Methods with an equation, then use the name in Fig 1, Fig 2 and Table 2 consistently. |
| D9 | **Weights footprint inconsistency** | The abstract says 10.5 MB; the runtime table says 14.8 MB total. | Pick the true value. Fig 6b annotates it. |
| D10 | **Human inter-observer cohort size** | The abstract says 41 scenes / 543 catheters; the table rows use 40 / 481–488 matched pairs. | The caption must state which (scenes, matched pairs) each human distribution uses. |

---

## 1. Color System

### 1.1 Principles

- **One hue = one entity across the whole paper.** A stage keeps its color in the schematic (Fig 1–2), the ablation points (Fig 4) and the runtime breakdown (Fig 6). "Ours" is always vermillion.
- Base: **Okabe–Ito**, which is safe for deuteranopia and protanopia. **Baselines are achromatic greys**, distinguished by marker shape and by their y-axis category label. This is the Nature-style "highlight one, mute the rest" convention, and it lets the eye find NeedleFinder 2026 instantly.
- Saturated colors are for **marks** only (lines, points, fills ≤ 50 % alpha). For **text**, use the darker `text` variant: every text variant is ≥ 4.5:1 on white.
- Okabe–Ito yellow `#F0E442` is **not used** because it is invisible on white and on MRI greyscale.

### 1.2 Entity palette

| Role | Token | Mark hex | Text hex | 20 % tint (fills, table bands) | Notes |
|---|---|---|---|---|---|
| **Stage 0**: physics symmetry operator | `st0` | `#E69F00` (orange) | `#8A5F00` | `#F7ECCC` | Also the Stage 0 heatmap overlay ramp |
| **Stage 1**: multi-head 3D ResUNet / IRGM | `st1` | `#0072B2` (blue) | `#005A8C` | `#CCE3F0` | Also the Stage 1 heatmap overlay ramp |
| **Stage 2**: Triplet MPNN + Bundle Graph Transformer | `st2` | `#009E73` (bluish green) | `#00704F` | `#CCECE3` | Graph nodes and edges in Fig 1–2 |
| **NeedleFinder 2026 (unified, SOTA)** | `nf` | `#D55E00` (vermillion) | `#9E4400` | `#FBEFE6` (10 %) | The only saturated warm data color in result plots |
| Baseline: **PMB 2019** (Zaffino/Pernelle 3D U-Net) | `b_pmb` | `#4D4D4D` | `#4D4D4D` | `#EDEDED` | Marker ■ (square) |
| Baseline: **nnU-Net + clDice** | `b_nnu` | `#7F7F7F` | `#5E5E5E` | `#F2F2F2` | Marker ▲ (triangle) |
| Baseline: **Xiao 2026 Sync-Track** | `b_xiao` | `#ABABAB` | `#6B6B6B` | `#F7F7F7` | Marker ◆ (diamond). Its edge is drawn `#4D4D4D` so it survives on white |
| **Human inter-observer** floor (`CY` vs `LB`/`AMe`) | `human` | `#000000` | `#000000` | `#E0E0E0` band | Dashed reference line (1.0 pt, dash `(4,2)`) plus a light-grey band for the IQR |
| Reported-only, not head-to-head | `reported` | hollow marker, edge = baseline grey | n/a | n/a | Hatch `////` for bars |

### 1.3 Semantic colors (qualitative figures)

| Meaning | Hex | Usage |
|---|---|---|
| Ground truth (2D overlay) | `#FFFFFF` line with `#000000` 0.4 pt halo | Contours on MRI |
| Ground truth (3D) | `#BDBDBD`, opacity 0.35 | Ghost tubes behind predictions |
| True-positive prediction | `nf` `#D55E00` **or** per-catheter error colormap (below) | |
| False negative (missed) | `#000000`, dashed tube / line | |
| False positive | `#CC79A7` (Okabe–Ito reddish purple) | Used **only** for FPs |
| ROI magnification box | `#56B4E9` (sky blue), 0.8 pt | Box and connector lines |

### 1.4 Continuous colormaps

| Data | Colormap | Range | Reason |
|---|---|---|---|
| MRI intensity | `gray`, window at 1st–99.5th percentile per case | | Clinical convention |
| Stage 0 symmetry response | orange single-hue alpha ramp: RGBA `(230,159,0, α)`, α = 0 → 0.85 over response ∈ [p50, p99.5] | | Matches `st0` |
| Stage 1 centerline/peak probability | blue single-hue alpha ramp `(0,114,178, α)` | [0.2, 1.0] | Matches `st1` |
| Stage 1 implant-region gate | `#0072B2` contour at p = 0.5, 0.8 pt, plus 15 % fill | | Shows the IRGM support region without hiding anatomy |
| Per-catheter error (3D render) | `cividis` | 0–2.0 mm, capped, colorbar with "≥2" tick | Perceptually uniform and CVD-safe |
| Diverging (e.g., signed tip-Z error) | `RdBu_r` centered at 0 | symmetric | |

### 1.5 Machine-readable tokens

Create `paper/style/nf_palette.py` with:

```python
# NeedleFinder 2026 · PMB design tokens (single source of truth)
NF = dict(
    st0="#E69F00", st0_text="#8A5F00", st0_tint="#F7ECCC",
    st1="#0072B2", st1_text="#005A8C", st1_tint="#CCE3F0",
    st2="#009E73", st2_text="#00704F", st2_tint="#CCECE3",
    nf="#D55E00",  nf_text="#9E4400",  nf_tint="#FBEFE6",
    b_pmb="#4D4D4D", b_nnu="#7F7F7F", b_xiao="#ABABAB",
    human="#000000", human_band="#E0E0E0",
    gt3d="#BDBDBD", fn="#000000", fp="#CC79A7", roi="#56B4E9",
    axis="#333333", grid="#E6E6E6", arrow="#4D4D4D",
)
MARKERS = dict(nf="o", b_pmb="s", b_nnu="^", b_xiao="D", human="_")
ORDER = ["human", "b_pmb", "b_nnu", "b_xiao", "nf"]  # bottom→top in rainclouds; ours nearest the x-axis label
LABELS = {
    "human": "Inter-observer (CY vs LB/AMe)",
    "b_pmb": "PMB 2019 3D U-Net",
    "b_nnu": "nnU-Net + clDice",
    "b_xiao": "Xiao 2026 Sync-Track",
    "nf": "NeedleFinder 2026",
}
```

LaTeX twin, in the preamble (§2.2):

```latex
\definecolor{nfst0}{HTML}{E69F00}   \definecolor{nfst0tint}{HTML}{F7ECCC}
\definecolor{nfst1}{HTML}{0072B2}   \definecolor{nfst1tint}{HTML}{CCE3F0}
\definecolor{nfst2}{HTML}{009E73}   \definecolor{nfst2tint}{HTML}{CCECE3}
\definecolor{nfours}{HTML}{D55E00}  \definecolor{nfrow}{HTML}{FBEFE6}
\definecolor{nfourstext}{HTML}{9E4400}
\definecolor{nfgrey}{HTML}{4D4D4D}  \definecolor{nfrule}{HTML}{BFBFBF}
```

---

## 2. Typography & LaTeX Page / Table Architecture

### 2.1 Root cause and fix

Pandoc renders every pipe table as a `longtable` with column widths taken from the dash counts. That causes the one-word-per-line first column and the page splits. **Fix: stop authoring tables in Markdown.** Each table becomes a hand-written LaTeX float in `paper/tables/*.tex` and is pulled into `manuscript.md` with a raw block:

````markdown
```{=latex}
\input{tables/tab2_main_results.tex}
```
````

Pandoc passes raw LaTeX through untouched, so no longtable is generated. Add the preamble via `-H style/preamble.tex` in **both** pandoc calls in `build_paper.sh`. Then remove `-V geometry:margin=0.85in`, because the geometry is set in the preamble.

### 2.2 Preamble: `paper/style/preamble.tex`

```latex
% ---------- Page ----------
\usepackage[a4paper,textwidth=170mm,textheight=240mm,centering]{geometry}
% ---------- Fonts: Times-like body + math (IOP look); sans for figures is set in figure tools ----------
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage[final,protrusion=true,expansion=true]{microtype}
% ---------- Colour ----------
\usepackage[table,HTML]{xcolor}   % 'table' loads colortbl
% (colour definitions from §1.5 go here)
% ---------- Tables ----------
\usepackage{booktabs,tabularx,threeparttable,multirow,makecell,array,etoolbox}
\usepackage{siunitx}
\sisetup{
  detect-weight = true, detect-family = true, mode = match,
  group-minimum-digits = 5,        % IOP style: 1063 (no separator for 4 digits). Use 4 to get "1 063" if preferred
  table-number-alignment = center,
  separate-uncertainty = true,     % 0.55 ± 0.36
  uncertainty-mode = separate,
}
\robustify\bfseries                 % lets \bfseries reach inside S columns
\newcommand{\best}{\bfseries}                                   % usage: \best 0.55
\newcommand{\second}[1]{\underline{\num{#1}}}                   % usage: {\second{0.62}}
\newcommand{\TBD}{\textcolor{red}{\textbf{TBD}}}                % never ship with this visible
\newcommand{\rep}{\textsuperscript{r}}                          % reported-only value marker
\newcommand{\oursrow}{\rowcolor{nfrow}}                         % SOTA shading
\newcommand{\NF}{NeedleFinder\,2026}
\newcommand{\Eshaft}{$E_{\mathrm{shaft}}$}
\newcommand{\Etip}{$E_{\mathrm{tip}}$}
\newcommand{\Davg}{$D_{\mathrm{avg}}$}
\newcolumntype{L}{>{\raggedright\arraybackslash}l}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\tabcolsep}{4.5pt}
\renewcommand{\arraystretch}{1.12}
\setlength{\aboverulesep}{0.35ex}  \setlength{\belowrulesep}{0.5ex}
\setlength{\cmidrulewidth}{0.4pt}
% ---------- Captions & floats ----------
\usepackage[font=small,labelfont=bf,labelsep=period,justification=justified,
            singlelinecheck=false,skip=4pt]{caption}
\captionsetup[table]{position=top}
\usepackage{placeins}             % \FloatBarrier at section ends: floats never drift past their section
\usepackage{float}
\renewcommand{\topfraction}{0.9}  \renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.8}
\setcounter{topnumber}{3}  \setcounter{totalnumber}{4}
% ---------- Misc ----------
\usepackage{rotating}             % sidewaystable for the literature table only
\usepackage[hidelinks]{hyperref}  % IOP prints in B/W: no coloured link boxes
```

Body text: 10 pt, newtx (Times). Tables: `\footnotesize` (8 pt). Captions: `\small` (9 pt) with a bold label.

### 2.3 Table rules

| Rule | Spec |
|---|---|
| **T1. No `longtable`, no Markdown tables** | Every table is a `table` float `[tbp]` in a `threeparttable`, so it can never split across pages. If a table does not fit in ¾ of a page, it is too big: split it (Part A/B) or move it to the supplement. |
| **T2. No `\resizebox`/`\scalebox`** | Scaling makes font sizes inconsistent between tables. Reduce columns instead. |
| **T3. Single-line method names** | Column 1 uses `L` (natural width, never wraps), with **≤ 28 characters** per name. Architecture details go to `\tnote`s or Methods. Example: `NeedleUNet3D + tangent tubes`, not "Stage 1 NeedleUNet3D (Anisotropic 3D ResUNet + SE) + Tangent Tubes". |
| **T4. One quantity per cell** | Never `992 / 1,063 (93.32%)`. Percent goes in its own column with the unit in the header, and counts go to the supplement or a separate *n* column. |
| **T5. Units in headers, not cells** | `E_shaft (mm)` with cells `0.55`, not `0.55 mm`. |
| **T6. Precision = information** | Detection percentages use **1 decimal** (93.3; with 1,063 catheters, 0.1 % ≈ 1 catheter). Distances use **2 decimals** in mm. Times use 2 significant figures in s. |
| **T7. Grouped headers** | Two-tier headers with `\cmidrule(lr){a-b}` per group: *Detection (1-to-1, %)*, \Eshaft *(mm)*, \Etip *(mm)*, *Time (s)*. |
| **T8. Hierarchy** | **Bold** = best per column **within a block** (Part A, Part B). __Underline__ = second-best. The NeedleFinder 2026 row is shaded `nfrow`. Never bold a whole row. Reported-only values get `\rep` and are excluded from best/second ranking. |
| **T9. Column alignment** | siunitx `S[table-format=2.1]` for %, `S[table-format=1.2]` for mm. Text columns are `L`. |
| **T10. Block separators** | `\midrule` between Part A and Part B. Part headers use `\multicolumn{n}{@{}l}{\textit{Part A — …}}`. A sub-block label (Whole cohort / Adhesion subgroup) is an indented italic row followed by `\addlinespace[2pt]`. |
| **T11. Notes** | Use `threeparttable` `tablenotes[flushleft]` with `\footnotesize`. Define every abbreviation, the matching rule (Hungarian, ≤ 3.5 mm?), the operating point τ, and fold protocol. |
| **T12. Never color text in tables** | Only the row tint. Tables must survive B/W printing. |

### 2.4 Table set (main text)

| Table | Content | Replaces current |
|---|---|---|
| **Table 1** | Datasets and provenance: BWH 54 pts / 1,063 catheters (759 long + 304 obturator); dual-observer subset (scenes, catheters); MSD Task05 48 zero-needle volumes; spacing and slice thickness | prose plus scattered numbers |
| **Table 2** | Main results. **Part A**: end-to-end vs baselines, Whole Cohort then Adhesion Subgroup. **Part B**: controlled ablation on identical folds | manuscript tables at lines 211, 221, 242 |
| **Table 3** | Agreement with human observers (dual-observer subset) | line 268 |
| **Table 4** | Zero-needle specificity (MSD Task05), 4 configurations only | line 290 (the 7-row τ sweep goes to the supplement; Fig 6a shows it) |
| Supp. S1 | Runtime and artifact breakdown per stage | line 306 (Fig 6b shows it) |
| Supp. S2 | Literature landscape (`sidewaystable`), with a **Modality** column and CT rows shaded `#F2F2F2` | line 330 |

### 2.5 Worked macro: Table 2 (`paper/tables/tab2_main_results.tex`)

Values come from `manuscript.md`, with the **Default operating point τ = 50 %**. τ must be fixed before reporting; the τ sweep is in Fig 4a. `\TBD` marks blockers from §0.

```latex
\begin{table}[tbp]
\centering
\begin{threeparttable}
\caption{\textbf{Catheter reconstruction accuracy on 54 BWH patients (1\,063 catheters), 5-fold patient-level out-of-fold.}
Part~A compares end-to-end methods; Part~B ablates \NF{} components on identical folds.
Bold: best; underline: second best within each block. Shaded: \NF{} (pre-specified $\tau=50\,\%$).}
\label{tbl:main}
\footnotesize
\begin{tabular}{@{}L
  S[table-format=2.1] S[table-format=2.1] S[table-format=2.1]
  S[table-format=1.2] S[table-format=1.2] S[table-format=1.2]
  S[table-format=1.2] S[table-format=2.2]@{}}
\toprule
 & \multicolumn{3}{c}{Detection, 1-to-1 (\%)} & \multicolumn{3}{c}{\Eshaft{} (mm)} & {\Etip{} (mm)} & {Time (s)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}\cmidrule(l){9-9}
Method & {Rec.} & {Prec.} & {F1} & {Mean} & {Median} & {P95} & {Median} & {CPU}\\
\midrule
\multicolumn{9}{@{}l}{\textit{Part A — End-to-end comparison}}\\
\addlinespace[1pt]
\multicolumn{9}{@{}l}{\quad\textit{Whole cohort (54 patients, 1\,063 catheters)}}\\
\quad PMB 2019 3D U-Net\tnote{a}     & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD\\
\quad nnU-Net + clDice\tnote{b}      & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD\\
\quad Xiao 2026 Sync-Track\tnote{c}  & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD\\
\oursrow \quad \NF                    & 88.8 & 81.9 & 85.2 & 0.63 & 0.53 & \TBD & \TBD & 0.61\\
\addlinespace[2pt]
\multicolumn{9}{@{}l}{\quad\textit{Catheter-adhesion subgroup (\TBD{} patients, \TBD{} catheters)}\tnote{d}}\\
\quad PMB 2019 3D U-Net              & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & {--}\\
\quad nnU-Net + clDice               & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & {--}\\
\quad Xiao 2026 Sync-Track           & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & {--}\\
\oursrow \quad \NF                    & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & \TBD & {--}\\
\midrule
\multicolumn{9}{@{}l}{\textit{Part B — Controlled ablation (whole cohort, identical folds)}}\\
\addlinespace[1pt]
\quad Physics + HybridNeedleNet      & 72.4 & \best 88.6 & 79.7 & {\second{0.62}} & 0.54 & \TBD & \TBD & \best 0.27\\
\quad NeedleUNet3D + tubes           & {\second{93.3}} & 62.9 & 75.2 & 0.66 & 0.56 & \TBD & \TBD & 0.52\\
\quad AxialTransUNet3D + tubes       & \best 93.8 & 61.6 & 74.4 & 0.67 & 0.57 & \TBD & \TBD & 0.57\\
\quad Triplet MPNN only              & 88.9 & 74.4 & {\second{81.0}} & \best 0.55 & \best 0.43 & \TBD & \TBD & {\second{0.42}}\\
\oursrow \quad \NF{} (full)           & 88.8 & {\second{81.9}} & \best 85.2 & 0.63 & {\second{0.53}} & \TBD & \TBD & 0.61\\
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item Rec./Prec.: one-to-one Hungarian matching; \Eshaft: in-plane centerline distance on matched slices; \Etip: 3D distal-tip distance. P95: 95th percentile over catheters.
\item[a] Zaffino \emph{et al.}\ 2019, retrained on our folds. \item[b] Isensee \emph{et al.}\ 2021 with clDice loss (Shit \emph{et al.}\ 2021). \item[c] Re-implemented on T2W MRI (original: \TBD{} modality). \item[d] GT centerlines $\le 3.0$\,mm apart on $\ge 3$ consecutive slices.
\end{tablenotes}
\end{threeparttable}
\end{table}
```

Note on second-best: `{\second{…}}` is braced text inside an S column, so it is centered rather than decimal-aligned. Every value in a column has the same `table-format`, so it still lines up visually.

Width check: 1 natural-width L column (≤ 28 chars at 8 pt ≈ 42 mm) + 8 numeric columns ≈ 13 mm each = 146 mm, which is less than the 170 mm text width, with no wrapping. **Rule: with an L column of ≤ 28 characters, at most 9 numeric columns fit.**

---

## 3. Figure Suite Blueprint

### 3.0 Global figure rules (apply to every figure)

**Canvas.** Full width is **170 mm**; single-column is **84 mm**; maximum height is **225 mm**, leaving room for the caption. Author at final size (1:1) and never rescale in LaTeX: `\includegraphics[width=\textwidth]` must equal the authored width. Check current IOP/PMB artwork guidelines for exact limits before submission.

**Type.** Sans: TeX Gyre Heros (Helvetica metric), falling back to Arial. Math: `stixsans` (matplotlib) or `sansmath` (TikZ).

| Element | Size |
|---|---|
| Panel letter | **9 pt bold**, lowercase **a**, **b**, top-left, outside the axes at (−8 mm, +2 mm) |
| Axis labels | 8 pt |
| Tick labels | 7 pt |
| Annotations / direct labels | 7 pt |
| Minimum anywhere | **6 pt** |
| Schematic block labels | 8 pt, ≤ 4 words |

**Lines.** Axes 0.6 pt `#333333`; top/right spines removed; ticks out, 2.5 pt long. Data lines 1.2 pt. Reference lines 0.8–1.0 pt dashed. Gridlines only on the value axis, 0.4 pt `#E6E6E6`, behind the data.

**Output.** Vector PDF straight from matplotlib (`savefig(..., format="pdf")`, `pdf.fonttype=42`). MRI and 3D renders are embedded as **≥ 600 dpi** raster inside the vector PDF. Drop the SVG → rsvg-convert path: it substitutes fonts.

**`paper/style/nf.mplstyle`:**

```ini
font.family: sans-serif
font.sans-serif: TeX Gyre Heros, Helvetica, Arial, DejaVu Sans
mathtext.fontset: stixsans
font.size: 7
axes.labelsize: 8
axes.titlesize: 8
xtick.labelsize: 7
ytick.labelsize: 7
legend.fontsize: 7
legend.frameon: False
axes.linewidth: 0.6
axes.edgecolor: 333333
axes.spines.top: False
axes.spines.right: False
xtick.direction: out
ytick.direction: out
xtick.major.size: 2.5
ytick.major.size: 2.5
xtick.major.width: 0.6
ytick.major.width: 0.6
lines.linewidth: 1.2
lines.markersize: 4
axes.grid: False
grid.color: E6E6E6
grid.linewidth: 0.4
savefig.dpi: 600
savefig.bbox: standard
savefig.pad_inches: 0
pdf.fonttype: 42
figure.constrained_layout.use: True
```

**Anti-collision rules (mandatory, checked automatically).**

- **R1. Direct labels only from a placement table.** Each label has a fixed `(dx, dy)` offset in points from its anchor, stored in the plotting script next to the data. Labels are not auto-placed at render time without review. `adjustText` may be used once to *propose* offsets, which are then frozen.
- **R2. Clearance.** Text-to-text ≥ 1.5 mm; text-to-data-mark ≥ 1 mm; text never crosses a data line except via a leader line (0.5 pt, `#808080`).
- **R3. No legends inside the data region.** Use direct labels at line ends, or a legend row above the panel (`loc="lower left", bbox_to_anchor=(0,1.02)`, `ncol=n`).
- **R4. Text budget.** Schematics: ≤ 4 words per box and **no bullet lists**. Detail goes in the caption.
- **R5. Overflow.** Data beyond the axis limit is clipped and annotated with a right-edge count, e.g. `▸ 7 > 3 mm`. Never silently cropped.
- **R6. Automated check.** Every figure script ends with `assert_no_overlap(fig)`, which fails the build:

```python
def assert_no_overlap(fig, pad_px=4):
    """Fail if any two visible Text artists' bounding boxes overlap (after padding)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(lambda a: hasattr(a, "get_text"))
             if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(r).expanded(1, 1).padded(pad_px)) for t in texts]
    bad = [(a.get_text(), b.get_text()) for i, (a, ba) in enumerate(boxes)
           for (b, bb) in boxes[i + 1:] if ba.overlaps(bb)]
    if bad:
        raise AssertionError(f"Overlapping labels: {bad[:10]}")
```

- **R7. Reported vs measured.** Measured-on-our-folds = filled marks or distributions. Literature-reported values that are not head-to-head = **hollow marker or hatched bar**, with a superscript "r" on the label and "reported, different cohort/modality" in the caption.

**Statistics conventions.**

- Catheters are nested in patients. All CIs are a **patient-level cluster bootstrap** (resample patients, 10,000 reps, percentile 95 % CI).
- Paired method comparisons on matched catheters use a Wilcoxon signed-rank test with patient-clustered permutation. Report effect size and CI in the table; at most **one** significance bracket per panel.

**Data contract.** One long-format file `reports/figures/data/per_catheter.csv`:

```
patient_id, catheter_id, method, cohort_flag{whole,adhesion,dual_observer}, fold,
matched{0,1}, e_shaft_mean_mm, d_avg_mm, e_tip_mm, e_tip_z_mm, n_slices
```

Every figure script reads only this file plus `runtime.csv`, `pareto_tau.csv`, `fewshot.csv` and `msd_fp.csv`. There are no numbers hard-coded in plotting code.

---

### Fig 1 · Pipeline overview (2 rows, visual-first)

**Canvas:** 170 × 72 mm, full width. **Tool:** TikZ `standalone` (fonts match the paper) with raster tiles rendered from a real case. Use **the same case as Fig 5a** for continuity.

```
Row 1 (52 mm) — data flow, left → right, 5 visual tiles of 26×26 mm, 8 mm gaps with arrows
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│ T2W    │ → │ Stage0 │ → │ Stage1 │ → │ Stage2 │ → │ 3D     │
│ stack  │   │ sym.   │   │ heads  │   │ graph  │   │ bundle │
└────────┘   └────────┘   └────────┘   └────────┘   └────────┘
Row 2 (14 mm) — stage strip: tinted bands under tiles 2/3/4, time-proportional runtime bar
```

| Tile | Visual content (no bullets) | Label (≤ 4 words) | Tint band |
|---|---|---|---|
| ① Input | 3 axial T2W slices offset in pseudo-3D perspective (8 % shear), with a faint z arrow | "3T T2W volume" | none |
| ② Stage 0 | One axial slice, orange symmetry overlay (§1.4) | "Symmetry operator" | `st0_tint` |
| ③ Stage 1 | 3 stacked mini-maps (centerline prob., tip, implant-region gate contour) in blue ramps, fanned like cards | "Multi-head ResUNet + IRGM" | `st1_tint` |
| ④ Stage 2 | Isometric 3D scatter of per-slice detections (green nodes, 2.5 pt) with triplet edges (0.5 pt) on 3–4 catheters; one rejected track greyed | "Triplet MPNN + Transformer" | `st2_tint` |
| ⑤ Output | 3D tube rendering of the full bundle in vermillion, ghost GT in grey, obturator in dark grey | "Catheter bundle" | none |

**Row 2 (stage strip).** A horizontal bar, 150 mm long, split by **measured CPU time** (0.060 / 0.420 / 0.135 s = Stage 0 / Stage 1A+1B / Stage 2A+2B+2C) and filled with the stage colors. Total at the right: "0.61 s CPU · 0.35 s GPU". A single 7 pt descriptor sits above each segment, e.g. "radial + half-ring · 60 ms".

**Arrows.** 0.75 pt `#4D4D4D`, `-{Stealth[length=2mm]}`, centered vertically on the tiles. Stage tiles get a 0.6 pt border in the stage color; input and output tiles get none.

**Total text budget:** ≤ 30 words outside the caption. Everything else (anisotropic kernels, SE, 5 heads, τ) belongs in Fig 2 and the caption.

---

### Fig 2 · Network & graph modules (math callouts)

**Canvas:** 170 × 125 mm. **Tool:** TikZ, because the math is native and the fonts match. **Grid:** 2 rows. **a** = 170 × 68 mm, Stage 1; **b** = 170 × 52 mm, Stage 2.

**Panel a: Multi-head anisotropic 3D ResUNet + IRGM**
- U-shape drawn as isometric cuboids. Cuboid height ∝ spatial size and depth ∝ channels (log scale). Tensor shape `C×D×H×W` is written **once per level**, 6 pt, below the encoder blocks.
- Block legend (bottom-left, 3 glyphs): ▭ residual 3D conv (`1×3×3` / `3×3×3` annotated on the first block only), ◐ SE, ⬓ IRGM.
- **IRGM inset** (dashed `st1` rounded box, 45 × 30 mm, linked by a leader to the bottleneck/skip it gates). The equation is typeset in the inset:
  `g = σ(W_g * [F ; S₀])`,  `F̃ = F ⊙ (1 + g)`. Here `S₀` is the Stage 0 response, entering from an orange arrow that originates at the Stage 0 glyph in the left margin. **Use the actual formulation once D8 is resolved.**
- Heads: 5 small output tiles on the right (8 × 8 mm each), each with its symbol and loss (e.g. `P_cl`, `L_cl`). **Populate the 5 head names from `NeedleUNet3D` source code; do not invent them.**
- Color: everything in panel a uses the `st1` family; the S₀ input arrow is `st0`.

**Panel b: Synchronous Triplet MPNN → Bundle Graph Transformer**
- Left (80 mm): 3 consecutive axial planes drawn as thin parallelograms at z−1, z and z+1, with candidate nodes `v_i^z` as green dots. One highlighted triplet `(v_i^{z−1}, v_j^z, v_k^{z+1})` with the callout
  `m_{ijk} = φ(h_i, h_j, h_k, cos θ_{ijk}, ‖Δx‖)`, `h_j ← h_j + Σ m_{ijk}`, with a "×4 layers" badge.
- Right (80 mm): tracks become **tokens** (one rounded pill per track), flowing into "2× edge-biased GAT" then "2× Set Transformer", to the per-track score `p_t`, then "keep if p_t ≥ τ". Rejected tracks are drawn in grey with a strike line.
- All panel b marks use the `st2` family.

**Callout style.** Math at 8 pt. Callout boxes are white with a 0.5 pt stage-color border and 1.5 mm padding. Leader lines are 0.5 pt `#808080`. **At most 3 callouts per panel.**

---

### Fig 3 · Error distributions: rainclouds (Whole Cohort vs Catheter-Adhesion Subgroup)

**Canvas:** 170 × 115 mm. **Grid:** 2 rows × 3 columns, `constrained_layout`, shared y within each row.

| | col 1: **E<sub>shaft</sub> (mm)** | col 2: **D<sub>avg</sub> (mm)** | col 3: **E<sub>tip</sub> (mm)** |
|---|---|---|---|
| **row a: Whole cohort** | x 0–3.0 | x 0–3.0 | x 0–15, linear; clip with R5 counts. Alternative: symlog with linthresh 2 |
| **row b: Adhesion subgroup** | same limits as row a (never re-scale between rows) | same | same |

**Orientation.** Horizontal rainclouds. Methods run along **y** in `ORDER`, so labels read horizontally and are never rotated. Labels appear on column 1 only.

**Per-method raincloud anatomy** (row pitch 1.0):
- **Cloud:** half-violin above the baseline (KDE, Scott bandwidth, `cut=0`, height-normalized to 0.42), fill = entity color at 35 % alpha, 0.6 pt edge in the solid color.
- **Box:** at baseline, height 0.10. IQR box white with 0.8 pt edge; median 1.2 pt; whiskers to P5/P95. Whiskers are **not** 1.5 IQR, so they are the P95 bar the user asked for: the P95 end gets a 0.12-high cap.
- **Rain:** jittered points below the baseline (y offset −0.15 to −0.38, uniform jitter), 1.8 pt markers at 25 % alpha. For n > 600, subsample to 600 for display only; statistics always use the full set.
- **Mean:** ◆ white fill, 0.8 pt black edge, on the box line.
- **Annotation:** right-aligned at the panel's x-max, 6.5 pt, in the text color: `median / P95`, e.g. `0.45 / 1.19`. This is the only per-method number text.

**Human floor.**
- The Human row is drawn like the others, in black.
- A **vertical dashed black line at human P95** and a light-grey band (`#E0E0E0`, 60 % alpha) spanning the human IQR run across the full height of each panel, behind all marks.
- Caption must say: the human distribution comes from the dual-observer subset (state scenes and matched pairs; D10). Where whole-cohort human data do not exist, mark the band "†dual-observer subset".

**Fairness options** (state the choice in the caption):
- (i) Default: row a = whole cohort for methods, and the human floor is shown as a reference only.
- (ii) Stricter, recommended as a **supplementary twin** of Fig 3: all methods restricted to the dual-observer subset and paired with the human comparison against `CY`.

**Reported-only baselines (D3/D4).** Only a hollow marker at the reported mean, with a ±SD line if given, and the label suffix "ʳ". No cloud, box or rain.

**Data available today** (manuscript): ours vs `CY` 0.55 ± 0.36, median 0.45, P95 1.19 mm. Human 0.71 ± 0.58, median 0.51, P95 2.02 mm. Tip medians: 3.50 (ours) vs 3.19 mm (human). The adhesion row and all baseline clouds are blocked by D1–D4 and D6.

---

### Fig 4 · Precision–Recall Pareto frontier + few-shot efficiency

**Canvas:** 170 × 72 mm. **Grid:** 1 × 2, width ratio 1.15 : 1.

**Panel a: 1-to-1 Precision–Recall (1,063 catheters, OOF)**
- **Axes:** x = recall 70–100 %, y = precision 55–95 %. Equal data-unit aspect.
- **Iso-F1 contours** F1 ∈ {0.75, 0.80, 0.85, 0.90}: 0.5 pt `#BFBFBF`, labels "F1 = 0.80" at the **top edge** of the axes, rotated along the curve, 6.5 pt grey. No labels inside the data cluster.
- **NeedleFinder 2026 τ sweep:** τ ∈ {25, 35, 45, 50, 60} %, points (R, P) = (93.7, 75.1), (92.8, 77.1), (90.2, 80.3), (88.8, 81.9), (87.9, 83.6). Drawn as a vermillion 1.2 pt line with ○ 4 pt markers. τ = 50 (default operating point) is a filled ● 5.5 pt with a 0.8 pt black ring.
- **τ labels** use a frozen offset table (R1), alternating sides to avoid the line. Since the sweep runs from lower-right to upper-left, labels go **above-right** of each point, (+4, +3) pt, as "25", "35", … and only the first is written "τ = 25 %".
- **Ablation points** (same folds), in stage colors with distinct markers:

  | Configuration | (R, P) | Mark |
  |---|---|---|
  | Physics + HybridNeedleNet | (72.4, 88.6) | `st0` ■ |
  | NeedleUNet3D + tubes | (93.3, 62.9) | `st1` ▲ |
  | AxialTransUNet3D + tubes | (93.8, 61.6) | `st1` △ hollow |
  | Triplet MPNN only | (88.9, 74.4) | `st2` ◆ |

- **Arrow** from NeedleUNet3D + tubes to the τ = 25 point: 0.6 pt grey arrow annotated "+12.2 pp precision at matched recall". This uses the manuscript's +12.21 pp at matched 95.20 % closest recall; verify that the pair plotted is the same matched pair.
- **Direct labels:** ablation labels sit outside the cluster, to the left for high-precision points and below for low-precision points. They use 0.5 pt leaders and are placed by the offset table.
- **Baselines** (once D2/D4 exist): grey markers per §1.2.
- **Human:** do **not** place a human point in P–R space. The inter-observer "match rate" is not precision/recall of a detector. Mention it in the caption only.

**Panel b: Few-shot annotation efficiency (BLOCKED, D5)**
- x = number of training patients {1, 5, 10, 20, 43}, log₂ axis, ticks labelled with the raw numbers. y = 1-to-1 F1 (%) on fixed OOF test folds.
- **Three lines:**
  - Stage 1 only (`st1`, dashed)
  - Stage 1 + Stage 0/IRGM gating (`st1` solid)
  - Full NeedleFinder 2026 (`nf` solid)
- Each line gets a **95 % bootstrap CI band** at 20 % alpha in its color, from ≥ 3 subset draws × patient-cluster bootstrap.
- A horizontal dotted line marks Stage 0 + physics tracking with **zero** training cases, as a zero-shot floor (`st0`, labelled at the right end).
- Lines carry direct labels at their right ends, using the frozen offset table. There is no legend.

---

### Fig 5 · Qualitative 3T T2W results (dense case + adhesion case)

**Canvas:** 170 × 118 mm. **Grid:** 2 rows (cases) × 4 columns, plus a 4 mm colorbar column.

| | col 1 (40 mm) | col 2 (40 mm) | col 3 (40 mm) | col 4 (44 mm) |
|---|---|---|---|---|
| **row a: dense case** (`<case_id>`: 24 catheters + obturator) | Axial T2W, GT contours (white w/ halo) and predictions (vermillion ×-marks); **ROI box** (sky blue) plus a **3× magnified inset** in the top-right corner of the same tile, linked by 2 connector lines | Same slice, Stage 0 orange overlay | Same slice, Stage 1 centerline prob. (blue ramp) + IRGM gate contour | 3D bundle render |
| **row b: adhesion case** (`<case_id>` from D1 with the most adhesion pairs) | as above, ROI on the touching pair | | | |

Selection rule for cases: pick the dense case by maximum catheter count with obturator present, and the adhesion case by maximum count of adhesion pairs. Both come **from the OOF predictions**, and the case IDs and per-case metrics go in the caption. Do not cherry-pick on error. If you also want a failure case, add a row c showing the worst-decile patient. Reviewers trust that more.

**Slice tiles.**
- No axes or ticks.
- A 10 mm white scale bar, 1.5 pt, bottom-left, with a "10 mm" label at 6.5 pt.
- Orientation letters "A" and "R" at 6 pt in two corners of the first tile only.
- All 4 tiles in a row show the **same slice index**, written as "z = 14/24" at 6 pt top-left.

**3D render** (PyVista, offscreen, 2400 px):
- Orthographic camera, same azimuth/elevation for both rows (e.g. az 35°, el 20°), white background.
- Predicted tubes radius 0.8 mm, colored per catheter by mean E<sub>shaft</sub> with `cividis` 0–2 mm. GT tubes are ghosted (`#BDBDBD`, opacity 0.35). FN are black dashed polylines; FP are `#CC79A7` tubes.
- The obturator is drawn as a dark-grey semi-opaque cylinder for context.
- One shared vertical colorbar (4 × 40 mm) for both rows, labelled "E_shaft (mm)", with the top tick "≥2".
- An in-render legend of 4 swatches (GT, FN, FP, Pred.) sits below row b only, as a single horizontal line.

---

### Fig 6 · Zero-needle specificity (MSD Task05, 48 volumes) & runtime

**Canvas:** 170 × 70 mm. **Grid:** 1 × 3, width ratios 1.1 : 1 : 0.9.

**Panel a: Cumulative patient-level specificity (ECDF)**
- x = false-positive tracks per volume, symlog with linthresh 1, ticks {0, 1, 2, 5, 10, 20, 50}. y = fraction of the 48 volumes with ≤ x FP, as a step plot (0–100 %).
- **Four configurations:**
  - Unpruned candidates (`#ABABAB`)
  - Stage 1 tubes, no gate/GNN (`st1` dashed)
  - Full NeedleFinder 2026 at τ = 60 % (`nf`)
  - Stage 0 gate + Stage 1 tubes (`st0`/`st1` two-tone dashed, or solid `st0`)
- Direct-label the two key readouts on the gated curve with leader lines: **"64.6 % with 0 FP"** at x = 0 and **"97.9 % with ≤1 FP"** at x = 1.
- Why an ECDF and not boxes: the clinically relevant statement is "fraction of patients with ≤ k false tracks", and the ECDF shows it directly without overplotting 48 × 4 points.
- Source: manuscript Table (line 290). Four rows go here; the full τ sweep goes in Supp.

**Panel b: End-to-end runtime vs literature (log axis)**
- Horizontal bars, x = seconds, **log₁₀**, from 0.1 to 200.
- **Rows, top → bottom:**
  - PMB 2019 (D7, hatched if reported)
  - Xiao 2026 11.6 s (hatched, reported; "CT" tag if D4 confirms)
  - NeedleFinder 2026 CPU 0.61 s
  - NeedleFinder 2026 GPU 0.35 s
- Value labels sit at bar ends (+2 pt, 7 pt). Hardware goes in a 6 pt grey sub-label under each name, e.g. "8-core desktop CPU" or "RTX 4090".
- Caption: "Reported runtimes were measured on different hardware and data; they indicate order of magnitude only."

**Panel c: NeedleFinder 2026 stage breakdown (linear axis)**
- Two horizontal **stacked** bars (CPU, GPU), linear 0–0.7 s. Stacked bars must be on a linear axis, never log.
- **Segments**, in stage colors with lighter variants for the A/B/C sub-stages:

  | Segment | CPU (s) | GPU (s) |
  |---|---|---|
  | Stage 0 | 0.060 | 0.060 |
  | 1A | 0.210 | 0.210 |
  | 1B | 0.210 | 0.030 |
  | 2A | 0.040 | 0.040 |
  | 2B | 0.080 | 0.010 |
  | 2C | 0.015 | 0.002 |
  | **Total** | **0.615** | **0.352** |

- Segments narrower than 6 mm get no inline label; they get an outside label with a leader, or appear only in the color key.
- Footnote text in the panel: weights footprint (D9).

---

## 4. Captions

Captions follow PMB/Nature style:

1. **Bold title sentence** that states the finding, not the content. Example: "**NeedleFinder 2026 falls within human inter-observer variability on shaft error.**"
2. Define *n* (patients / catheters / matched pairs), protocol (5-fold OOF), operating point τ and matching rule.
3. Decode every encoding: what the box, whiskers (P5–P95), diamond (mean), band and hollow marks mean.
4. State the statistical test and CI method (§3.0).
5. Give the reported-vs-measured caveat where R7 applies.

Captions should be ≤ 120 words for data figures and ≤ 80 words for schematics.

---

## 5. Build integration checklist

- [ ] Create `paper/style/{preamble.tex, nf.mplstyle, nf_palette.py}` and `paper/tables/*.tex`.
- [ ] `build_paper.sh`: add `-H style/preamble.tex`, drop `-V geometry…`, `-V colorlinks`, `-V linkcolor/citecolor/urlcolor` (hyperref is configured in the preamble), and drop the SVG→PDF loop once figures are emitted as PDF.
- [ ] Replace every pipe table in `manuscript.md` with ```` ```{=latex} \input{tables/…} ``` ````.
- [ ] Put a raw `\FloatBarrier` at the end of the Results and Discussion sections.
- [ ] `scripts/figures/make_fig{1..6}.py` (+ `fig1.tex`, `fig2.tex` TikZ). Each reads only `reports/figures/data/*.csv` and ends with `assert_no_overlap(fig)`.
- [ ] Build fails if `TBD` appears in the PDF text: `pdftotext needlefinder_2026_pmb.pdf - | grep -q TBD && exit 1`.
- [ ] Grayscale proof: convert the PDF to greyscale and confirm every figure is still decodable (markers, dash patterns, hatching).
- [ ] CVD proof: simulate deuteranopia and protanopia (e.g. `daltonlens`) on Figs 3, 4 and 6.
- [ ] Resolve blockers D1–D10 (§0) and record the decisions on this page.

Related: [[plans/2026-10-22-needlefinder-2026-manuscript-draft]], [[wiki/syntheses/2026-10-22-multi-agent-paper-review-synthesis]], [[wiki/entities/pmb-2019-zaffino-pernelle-pmc6726393]], [[wiki/research/2026-10-21-sota-nn-hybrid-needle-detection]].
