# Visual Design, Accessibility, and Storytelling Overhaul (`NeedleFinder 2026` PMB V2)

- **Date:** 2026-10-24
- **Linear Issue:** `AUD-526` (`Visual storytelling, CVD accessibility, and typographic overhaul of NeedleFinder 2026 PMB manuscript & figures`)
- **Review Squad:**
  - **Codex Astra (`gpt-6-astra`)** — `[[sources/2026-10-24-visual-review-codex-astra]]`
  - **Claude Opus 5.5 (`claude-opus-5-5`)** — `[[sources/2026-10-24-visual-review-opus-5-5]]`
  - **Claude Fable 5.1 (`claude-fable-5-1`)** — `[[sources/2026-10-24-visual-review-fable-5-1]]`
  - **Antigravity (`AGY`)** — Direct pixel-by-pixel inspection of all rendered PDF pages (`pdftoppm`) and PNG figures

---

## 1. Executive Summary & Design Philosophy

To make the *Physics in Medicine & Biology* (PMB) manuscript (`[[paper/manuscript]]`, `paper/needlefinder_2026_pmb.{pdf,html,tex}`) and its 8-figure suite (`paper/generate_pmb_figures.py`, `reports/figures/fig1..8.{pdf,png,svg}`) a joy to read and immediately understandable at a glance, we conducted a 4-model visual audit and executed a complete typographic, color-accessibility, and visual-storytelling overhaul inspired by two exemplars:

1. **Claudia Clopath's Visual Pedagogy (*Nature Neuroscience* / *Nature Communications* / *eLife*):**
   - **"Why before how" visual pairing:** Every technical module is introduced by pairing the physical/anatomical failure mode of existing methods with the geometric mechanism that resolves it (e.g., Figure 1b's 3-clue mechanism cards; Figure 2a–d's 4-quadrant architectural blueprint).
   - **One clear claim per panel:** Each panel title states what the panel shows in bold (`8.5–9.0 pt` at 1:1 print scale), and key numerical take-aways are annotated directly on the canvas with zero overlapping labels.
   - **Full-distribution transparency:** Raincloud plots (Figure 3a, b) display the unclipped KDE density, IQR box, 5th–95th percentile whiskers, mean diamond, individual catheter points, and a dedicated right-hand tabular summary column (`med / P95`) separated from the data canvas by a vertical hairline so text never collides with density clouds.

2. **Tina Kapur's Clinical Brachytherapy & 3D Slicer Translation (*MedIA* / *PMB* / *Brachytherapy*):**
   - **Operating-room grounding:** Figure 1a contrasts the 45–50 minute manual slice-by-slice clicking bottleneck in the BWH AMIGO suite against the 4.15 s CPU (`2.67 s` GPU) `NeedleFinder 2026` workflow in 3D Slicer.
   - **Severe adhesion proof on clinical MRI:** Figure 5b (`Case 031`) includes a dedicated callout box highlighting a touching 6F catheter pair (`2.2 mm` center-to-center separation) resolved into two sharp Stage 1 centerline peaks (`P_peak`).
   - **Honest separation of measured vs. simulated clinical steps:** Figure 6c (`fig7_calibration_and_reasons`) and Figure 8a, b explicitly distinguish measured headless runtime (`4.15 s` CPU / `2.67 s` GPU, solid borders) from modeled 1-click tip review (`~18 s*` for `k=2` tips, dashed borders and `*` notation).

---

## 2. Eleven Defects Identified by the 4-Model Visual Audit & How Each Was Resolved

| # | Defect Identified (`Astra` + `Opus 5.5` + `Fable 5.1` + `AGY`) | Root Cause | Resolution Implemented |
|---|---|---|---|
| **1** | **Canvas-to-print shrinkage (`66%`–`89%`) & tiny sub-6 pt fonts** | Figures were drawn at `180–240 mm` width and scaled down to `160 mm` in LaTeX, shrinking `5.5–6.5 pt` text to `4.2–5.2 pt`. | Standardized `W2 = 160.0 * MM` and `W1 = 78.0 * MM` in `style/nf_palette.py` (`1:1` physical placement, `0%` shrinkage), set `style/nf.mplstyle` to `IBM Plex Sans` (`8.5 pt` base, `>= 7.4 pt` minimum), and enforced `min_fontsize=7.4` in `assert_no_overlap()`. |
| **2** | **Page 20 (`Table 2`) and Page 32 (`Table 6`) bottom-of-page clipping** | Single giant `\scriptsize` floats exceeded `232 mm` text height (`y = 859 pt > 841.89 pt` A4 page bottom). | Upgraded all 6 tables (`tab1`–`tab6`) to `\footnotesize`, set `\global\setlength{\columnwidth}{\textwidth}` (`160 mm`), and split `tab2_main_results.tex` and `tab6_dosimetry_and_generalization.tex` into two consecutive `\begin{table}[tbp]` floats (`Part A` + `\addtocounter{table}{-1}` `Part B`/`Part C`). Verified via PyMuPDF (`fitz`): **0 text spans below `780 pt`** across all 39 pages, and **0 `Overfull \hbox` warnings**. |
| **3** | **Column collisions in `Table 5` (`tab5_literature.tex`)** | Inline strings (`56 plans`, `203 fx`, `214 vol.`, `34 scans`) collided with adjacent numeric columns. | Replaced inline unit strings with footnote `\tnote{o}` and balanced `\tabcolsep`. |
| **4** | **Broken inline bullet/numbered lists in `manuscript.md`** | Pandoc requires a blank line before `-` or `1.` lists; 12 lists were rendered as run-on paragraphs in PDF/HTML. | Inserted blank lines before all 12 lists in `paper/manuscript.md` with zero word or number changes. |
| **5** | **CVD hazard in `Figure 6b` (`fig7_calibration_and_reasons`)** | Stacked bar placed Vermilion (`#D55E00`) directly adjacent to Coral Red (`#CC3311`), which merge under deuteranopia/protanopia. | Replaced `#CC3311` with Okabe–Ito Reddish Purple (`#CC79A7` + `///` hatch) for false alarms, Sky Blue (`#56B4E9` + `..` hatch) for redundant duplicates, and Dark Slate (`#1E293B`) for lost catheters. |
| **6** | **Missing Stage 2C 1D Axial Tip Localizer in `Figure 2`** | `fig2_architecture_modules` only showed 3 panels (Stage 0, Stage 1, Stage 2A/B), omitting Stage 2C (`91-step` `0.5 mm` dilated 1D ResNet + 4-head attention + split-conformal sphere). | Redesigned `fig2_architecture_modules` into a 4-quadrant mechanism plate (`a`: Stage 0; `b`: Stage 1 + 51-vol SSL; `c`: Stage 2A/B + SynthBrachy3T; `d`: Stage 2C 1D Tip Localizer + split-conformal `90.2%` sphere). |
| **7** | **Summary labels overlapping KDE violins in `Figure 3a, b`** | Median/P95 text was drawn inside the plot area (`x = 29.5`), colliding with upper whiskers and density tails. | Created a dedicated right-hand tabular column (`col_x = data_max * 1.54`) separated by a dotted hairline (`data_max * 1.03`), and added **Panel 3c** (Whole vs. Severe Adhesion F1) and **Panel 3d** (Grubbs' 3-cornered-hat intrinsic error SD). |
| **8** | **Four peer-reviewed baselines hidden in a text box in `Figure 4a`, and $k=0$ omitted from `Figure 4b`** | `ax1` clipped Y to `[60, 93]%`, hiding `Aleong/Kim (69.5%)`, `Kostoulas (66.5%)`, `Dai (58.2%)`, `Xiao (54.2%)`, and `clDice (38.7%)` in a corner note. | Expanded `ax1` Y-axis to `[18, 95.5]%` so all 6 external baselines + Stages 0/1B/2A/2B/4d are plotted at their true $(R, P)$ coordinates with leader lines; added $k=0$ (`78.2%` whole-cohort, `83.4%` long-shaft zero-shot `SynthBrachy3T` vs. `75.3%` Stage 0) to `ax2`. |
| **9** | **Faint 3D reference bundles in `Figure 1c` and `Figure 5`** | Reference curves used `#94A3B8` at `alpha=0.45`, making missed/reference trajectories nearly invisible. | Upgraded 3D reference trajectories to high-contrast charcoal dashed curves (`#475569`, `lw=1.8`, `ls="--"`, `alpha=0.72`) and added a `2.2 mm` touching-pair callout box on `Case 031` (`Figure 5b`). |
| **10** | **Dual-axis visual confusion & omitted `T1 VIBE` failure boundary in `Figure 8`** | Panel 8d overlaid depth bending (`mm` vs. depth) and lateral tissue push (`mm` vs. distance) on dual X/Y axes (`ax_d` + `ax_d_top`), and Panel 8c omitted the `AX T1 VIBE POST` (`56.1%` F1) sequence. | Expanded `fig8` into a 5-panel layout (`a`: TG-43U1 Protocol A vs. B + median markers; `b`: $\hat{\sigma}_{\text{tip}}$ review curve vs. random & oracle; `c`: full-width zero-shot transfer bar chart **including `Post-contrast 3D T1 VIBE (56.1% F1)`**; `d`: dedicated 4D depth-dependent bending axis; `e`: dedicated 4D lateral tissue-push axis). |
| **11** | **HTML `mix-blend-mode: multiply` tinting figures** | `build_html.py` applied `mix-blend-mode: multiply` on `figure img`, tinting white plots and dark MRI slices to warm beige. | Switched `figure img` to `mix-blend-mode: normal` on a crisp `#FFFFFF` card with subtle border, and increased HTML body (`19.5px`, `line-height: 1.56`), table (`0.83rem`), and caption (`0.88rem`) typography. |

---

## 3. Unified Semantic & Colorblind-Safe Palette (`style/nf_palette.py`)

All 8 figures and all 6 LaTeX/HTML tables share a single semantic palette combining the **Okabe–Ito (2008)** and **Paul Tol** colorblind-safe scales, verified for deuteranopia, protanopia, tritanopia, and grayscale printing via redundant shape, dash, and hatch encodings:

- **Stage 0 (3D Radial Dark-Core Prior):** Okabe–Ito Orange `#E69F00` (text-safe `#8A5A00`, tint `#FDF3DC`) — plus marker `+`, dotted/solid line.
- **Stage 1 (3D ResUNet + Pelvic SSL):** Okabe–Ito Blue `#0072B2` (text-safe `#004D7A`, tint `#E3F1FA`) — triangle `^`, dash-dot line.
- **Stage 2A/2B (Triplet MPNN & Bundle Graph Transformer):** Okabe–Ito Bluish Green `#009E73` (text-safe `#006B4D`, tint `#E0F5EE`) — diamond `D`, solid line.
- **Stage 2C / Proposed `NeedleFinder 2026` (`4b` / `4d`):** Okabe–Ito Vermilion `#D55E00` (text-safe `#9E4400`, tint `#FCEEE4`) — star `*` / circle `o`, thick solid line.
- **False-Positive / Polarity-Shift Boundary:** Okabe–Ito Reddish Purple `#CC79A7` (text-safe `#882E62`, hatch `///`).
- **Redundant Duplicate Track:** Okabe–Ito Sky Blue `#56B4E9` (text-safe `#1B6998`, hatch `..`).
- **Human Expert Reference (`Team 1 CY` / `Team 2 LB/AMe`):** Dark Charcoal `#1E293B` / `#475569` — square `s`, dashed line `--`.
- **Reproduced Baselines (`PMB 2019`, `Aleong/Kim`, `Kostoulas`, `Dai`, `Xiao`, `clDice`):** Monotonic neutral slate ramp (`#334155` $\to$ `#78879B`) with distinct markers (`s, p, v, <, d, X`).

---

## 4. Automated Layout & Typography Linter (`assert_no_overlap`)

In `paper/style/nf_palette.py`, `assert_no_overlap(fig, pad_px=1.0, min_fontsize=7.4)` is called before saving every figure:
1. Executes `fig.get_layout_engine().execute(fig)` and `fig.canvas.draw()` at 300 DPI.
2. Verifies that every visible `matplotlib.text.Text` artist has `fontsize >= 7.4 pt` at 1:1 physical print width (`160 mm`).
3. Checks all pairs of visible text bounding boxes within the same axes (and figure-level text) to guarantee zero bounding-box overlaps (`pad_px = 1.0`).
4. Alongside `generate_pmb_figures.py`, a PyMuPDF (`fitz`) verification pass on `needlefinder_2026_pmb.pdf` confirms that every page's maximum text Y-coordinate remains strictly above `780 pt` (well inside the `841.89 pt` A4 page boundary) and that `tectonic` emits **zero `Overfull \hbox` warnings**.
