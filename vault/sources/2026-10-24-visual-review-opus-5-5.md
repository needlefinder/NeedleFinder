# Visual design, figure storytelling and typography review of the NeedleFinder 2026 PMB manuscript

**Scope.** I read `nf.mplstyle`, `nf_palette.py`, `preamble.tex`, the figure and CSS sections of `build_html.py`, `build_paper.sh`, the six table `.tex` files, and the manuscript captions and method sections. I looked at all eight rendered PNGs and the generator code behind each one. I didn't edit any files.

**Caveats.**
- The PNGs I reviewed (08:08) are about five minutes older than the PDFs and SVGs the paper uses (08:13). Re-check the PDFs before acting on small layout points.
- I couldn't run a colour-blindness simulation script: running Python in `/tmp` needed permission I didn't have. The colour-blindness verdicts below rest on well-known behaviour of the Okabe–Ito colours, not on numbers I measured. Item 0.6 says how to add a measured check.

---

## 0. The six problems that matter most

1. **The text in the figures is too small.** The style file sets 7 pt, not 7.5 pt as the brief says. The generator then overrides it in almost every call with 4.9–6.6 pt. The figures are drawn 170 mm wide but placed at the 160 mm float width (`width=100%`), so everything shrinks by another 0.94. In print, the legend text in Figs 6a, 8b and 8d is about **4.6 pt**, and the annotations in Fig 4 are about 5 pt. That is below any comfortable reading size.

2. **Colours mean different things in different figures.** Okabe–Ito orange `#E69F00` stands for "Stage 0" in Figs 1, 2 and 5. It also stands for "SSL Stage 1 + gate" in Fig 6a, "0-shot analytical" in Fig 4b, "Paired CT" in Fig 8c and "lateral push" in Fig 8d. Green means Stage 2, "NF + Synth", "calibrated" and "redundant". Vermilion means "ours", "Stage 2B" and "kept". Three pairs are hard to separate for readers with red-green colour blindness:
   - vermilion `#D55E00` vs orange `#E69F00` (Fig 6a, the two best curves);
   - vermilion vs brick red `#B03F41` (Fig 7b "kept" vs "lost"; Fig 8d P90 vs mean);
   - steel blue `#3B6EA5` vs purple `#6A51A3` (Fig 3, Aleong vs Kostoulas).

3. **Five captions don't match their figures.** A reviewer will catch these.
   - **Fig 7 (calibration).** The caption says "before (dashed) and after (solid vermilion)". In the figure, raw is solid vermilion with circles and calibrated is solid green with squares.
   - **Fig 8d.** The caption says depth is "bottom axis, red" and push is "top axis, blue". In the figure, depth uses three colours (brick red, vermilion, green) and push is orange.
   - **Fig 5.** The caption mentions an obturator gate drawn as "dashed orange". It isn't visible in either row.
   - **Fig 3.** The caption says the grey band and dashed line show inter-observer agreement (median 0.49 mm). In the code, panel a draws the human **IQR band plus a P95 line** (at about 1.9 mm). Panel b draws **only a median line** (3.24 mm) and no band, because the band collapses to zero width. The same dashed line means P95 in one panel and the median in the other.
   - **Fig 2.** The caption describes the 91-step dilated 1D ResNet + self-attention tip localizer and its uncertainty interval. Neither is drawn.

4. **Several claims are drawn but unsupported, or supported but not drawn.**
   - Fig 2 shows **"Prunes 77% of vessel FPs"**. That claim isn't in the manuscript (grep finds no "77%"). Check it against the benchmark JSON or remove it.
   - The T1 VIBE result (56.1% F1, the honest failure case in the results text) is **missing from Fig 8c**.
   - The physics-zeroed ablation (0.0% F1) appears in Fig 8c as if it were a transfer result. It belongs with the ablations.

5. **The figures don't follow the paper's own story.** Manuscript line 104 gives the paper's best storytelling device: a physicist uses three clues, in order — round dark spots, spots lining up across slices, then a plausible bundle that stops at the tip. No figure uses it. There's also no figure for the core physical puzzle (why a catheter looks different from a fascia edge, a vessel or bowel gas). And there's no picture of the operating-room workflow, even though the abstract and Discussion are built around it.

6. **Figure numbers don't match file names.** The manuscript places `fig7_calibration…` (line 278) before `fig6_specificity…` (line 296). pandoc-crossref numbers figures by order, so the calibration file prints as **Figure 6** and the specificity file as **Figure 7**. Rename the files to match print order. The rest of this review uses the file numbers.

---

## 1. Proposed narrative arc

The current figures are a methods-and-results catalogue. A storytelling sequence would introduce three new illustrations (marked NEW) and split Fig 8 in two:

| # | Take-home sentence (goes in bold as the first line of the caption) | Source |
|---|---|---|
| **1 · NEW** | *"A catheter is a dark dot surrounded by brightness on every side; most look-alikes are not."* Physics illustration paired with real image patches. | new |
| **2** | *"NeedleFinder reads an MRI the way a physicist does: spots, then lines, then a bundle."* | redesigned Fig 1 |
| **3** | *"Each stage is small and inspectable."* Stage 0 operator, Stage 1 heads, Stage 2 graph, and the tip localizer with conformal error bars. | Fig 2 + NEW tip/conformal panel |
| **4** | *"Along the shaft, the software agrees with experts as well as experts agree with each other."* | Fig 3 + three-cornered-hat panel |
| **5** | *"Each stage moves the method up the precision–recall frontier, and one annotated patient is already enough to beat zero-shot."* | Fig 4 |
| **6** | *"What it sees, case by case, including its misses."* | Fig 5 |
| **7** | *"Its confidence can be trusted, and every rejection carries a reason."* | Fig 7 + Fig 6a |
| **8 · NEW** | *"In the operating room: one click, two tip checks, a dose within 2.8 % of manual tracing (in simulation)."* | Fig 8a, b + NEW workflow/Slicer panel + runtime |
| **9** | *"Beyond the training domain, including where it fails."* | Fig 8c, d split out |

Pairing a mechanism with data (Clopath) means Figs 1 and 3 each show a schematic next to the real data it explains. Grounding in the clinic (Kapur) means Figs 2 and 8 put time on a real operating-room clock, using the numbers already in the text: 46–51 min of manual tracing inside a 100-min procedure, 4 s, 18 s and 5 s. Bars that are modeled rather than timed are drawn hatched, so the honesty rule is visible in the figure itself.

---

## 2. A shared style system (one change, every figure benefits)

### 2.1 Colour semantics: a cool ramp for the machinery, warm vermilion for the answer

The idea: pipeline stages are ordered, so encode them by **lightness in one hue family**. Lightness survives every type of colour blindness and greyscale printing. That frees vermilion to mean only "NeedleFinder's output", and stops orange or green being reused for unrelated things.

```python
# nf_palette.py — proposed replacement tokens
NF = dict(
    # ── Pipeline stages: one blue family, ordered light → dark ──
    st0="#9CC3E0", st0_text="#2E6A96", st0_tint="#EAF2F9",   # Stage 0 physics
    st1="#3D85BE", st1_text="#24618F", st1_tint="#DCEAF5",   # Stage 1 network
    st2="#16466E", st2_text="#16466E", st2_tint="#D3DEE8",   # Stage 2 graph/tip
    # ── The answer ──
    nf="#D55E00", nf_text="#9E4400", nf_tint="#FBEFE6", nf_light="#F0A776",  # ours / ablation-of-ours
    # ── Outcome semantics (identical in Figs 5, 7, 9 and in the Slicer module) ──
    ref="#1A1A1A",            # expert reference (black / dark grey line or open ring)
    fp="#CC79A7",             # false track: Okabe-Ito reddish purple + hatch '////'
    fn="#1A1A1A",             # missed catheter: black dashed, open marker
    redundant="#BDBDBD",      # duplicate of kept track: light grey + hatch '....'
    good="#009E73",           # reserved for "improved by calibration / review" only
    # ── Baselines: grey ramp with distinct markers, never a hue ──
    b1="#3A3A3A", b2="#5E5E5E", b3="#828282", b4="#A6A6A6", b5="#C4C4C4",
    human="#000000", human_band="#E3E3E3",
    axis="#3A3A3A", grid="#EBEBEB", arrow="#5E5E5E",
)
MARKERS = dict(nf="o", nf_abl="o_open", b1="s", b2="^", b3="D", b4="v", b5="P", human="|")
LINESTYLES = dict(nf="-", ref="--", b="-", fn=(0, (2, 1.5)), oracle=":")
```

Rules to write at the top of `nf_palette.py`:
- **Only NeedleFinder's own results are vermilion.** Ablations of NeedleFinder use `nf_light` or open vermilion markers.
- **Baselines are always grey, told apart by marker and direct label, never by hue.** Fig 3 then needs no legend: the y-tick labels already name each row.
- **Every colour carries a second cue:** a marker, line style, hatch or direct label. Nothing relies on hue alone.
- **Red is banned.** `#B03F41` is gone. "Lost real catheter" uses `fn` black dashed or open; "false track" uses `fp` purple with hatching.
- **Green appears only where something improved** (calibrated vs raw, reviewed vs unreviewed). That gives it a single meaning.

Also replace `"#3B6EA5"`, `"#6A51A3"`, `"#8C6D31"`, `"#90A4AE"`, `"#B03F41"`, `"#888888"` and `"#DD7A2E"` (the alpha-blended NF bars in Fig 6b) with tokens. **No hex literals should remain in `generate_pmb_figures.py`.** A grep for `"#[0-9A-F]{6}"` in that file should return zero.

### 2.2 Overlays on MRI (Figs 1, 5)

- **Stage 0 map.** `YlOrBr` drawn as dense speckle reads as noise. Draw only the local maxima above a threshold, as small rings in the `st0` colour, with a 0.6 pt dark outline. If a continuous map is needed, use a single-hue sequential map with a colourbar, for example a light-to-dark blue ramp ending at `st0`.
- **Stage 1 peaks.** Use solid `st1` dots with a white halo (`path_effects.withStroke(linewidth=1.2, foreground="white")`). The current yellow ring plus blue fill uses two stage colours for one stage.
- **Contrast.** Show the MRI at full opacity. Darken outside the region of interest with a 30% black mask instead of fading the whole image to alpha 0.6–0.8. Columns 2 and 3 of Fig 5 currently look washed out.

### 2.3 Typeface and size scale

The current figure font, DejaVu Sans, is wide and heavy. Next to EB Garamond it looks like it came from a different paper.

**Recommendation: IBM Plex Sans** for figures, with **IBM Plex Sans Condensed** for dense tick labels. The paper already uses IBM Plex Mono for code, so the type family is consistent. Plex Sans has proper tabular figures, a clear `1`/`l`/`I` distinction, a true minus sign, and pairs cleanly with Garamond. Embed it with `pdf.fonttype 42`, as now.

```ini
# nf.mplstyle — proposed (sizes are FINAL PRINTED points at 160 mm width)
font.family: sans-serif
font.sans-serif: IBM Plex Sans, Source Sans 3, DejaVu Sans
mathtext.fontset: custom
mathtext.rm: IBM Plex Sans
mathtext.it: IBM Plex Sans:italic
mathtext.bf: IBM Plex Sans:semibold
font.size: 8
axes.titlesize: 8.5
axes.titleweight: semibold
axes.titlelocation: left
axes.titlepad: 4
axes.labelsize: 8
xtick.labelsize: 7.5
ytick.labelsize: 7.5
legend.fontsize: 7.5
legend.handlelength: 1.6
axes.linewidth: 0.7
xtick.major.width: 0.7
ytick.major.width: 0.7
xtick.major.size: 3
ytick.major.size: 3
lines.linewidth: 1.6
lines.markersize: 4.5
lines.markeredgewidth: 0.8
axes.grid: True
axes.grid.axis: y
grid.color: EBEBEB
grid.linewidth: 0.5
axes.axisbelow: True
axes.unicode_minus: True
figure.constrained_layout.use: True
figure.constrained_layout.h_pad: 0.04
figure.constrained_layout.w_pad: 0.04
savefig.dpi: 600
savefig.bbox: standard          # NOT tight: tight silently rescales the figure away from 160 mm
svg.fonttype: path
pdf.fonttype: 42
```

**Size rules:**
- Minimum **7 pt** for any text in print (in-panel annotations included).
- Ticks and legends 7.5 pt; axis labels 8 pt; panel titles 8.5 pt semibold; panel letters 10 pt bold.
- Use at most two text weights per figure, regular and semibold. Today almost every annotation is bold, so nothing stands out.

**Geometry fix in `nf_palette.py`:**
```python
W2 = 160.0 * MM   # = \columnwidth in build_paper.sh; was 170 mm → 0.94× shrink of every glyph
W1 = 76.0 * MM    # half-width panels
```
Remove every inline `fontsize=` from the generator. Use named roles instead, for example `FS = dict(note=7, tick=7.5, label=8, title=8.5, letter=10)`, and add a lint that fails below 7 pt (§2.5).

**Graphics conventions to add:**
- Panel letters as bold lowercase **a b c** in the top-left corner of each panel, with the title on the same baseline in a separate text call. This is the Nature style.
- Label lines directly at their right end and drop legends wherever there are four series or fewer. Use `ax.annotate` with `xytext=(4,0), textcoords="offset points"`.
- Put a white halo on any text drawn over data or images, using the `path_effects` stroke from §2.2.
- Use a log axis whenever more than 10% of points would be off-scale (see Fig 3b).

### 2.4 `preamble.tex` and `build_html.py`

**PDF:**
- EB Garamond 11 pt on a 120 mm measure is beautiful and should stay. Garamond's small x-height makes 11 pt read like about 10 pt Times, so consider `fontsize=11.5pt` (needs `extsizes`, or `\fontsize{11.5}{15}` in the preamble) or `\linespread{1.1}`.
- **Tables are all `\scriptsize`** (8 pt Garamond reads like about 7 pt). Raise them to `\footnotesize` at minimum. Table 2 is 73 source lines; split it into a main table and a supplementary table rather than shrinking it. Table notes at `\scriptsize` (tab2, tab5) should become `\footnotesize`.
- Captions: keep `\small`, but add the bold take-home first sentence with `\captionsetup{font=small}` plus a `\textbf{}` lead. Most captions already start with a bold phrase; make it a claim rather than a label, for example "Shaft errors match expert-to-expert agreement" instead of "Per-catheter shaft and tip error distributions".
- Add `\definecolor` entries mirroring the new tokens so the table "ours" rows and the figures share exact colours.

**HTML:**
- Line 889 rewrites `.pdf` to `.png` (300 dpi). Switch it to `.svg`: the SVGs already exist, stay sharp at any zoom, and keep the text in a figure selectable. Keep the PNG only for `fig5` if its SVG (900 KB with rasters) turns out slow.
- Let figures break out wider than the 29.5 rem measure on wide screens. Add something like `figure{grid-column: wide;}` up to about 52 rem. The figures are 160 mm print designs, and squeezing them into 560 px roughly halves the type.
- Add a click-to-zoom lightbox (about 15 lines of JS: a full-viewport `<dialog>` holding the SVG).
- Write real `alt` text for each figure: one sentence stating the finding. pandoc's alt text is currently the full caption.
- `mix-blend-mode: multiply` is good, but figures should have transparent backgrounds. Set `savefig.transparent: True` for SVG, and give panels like the Fig 1 output tile tokenised tints rather than `#FAF8F7`, so they don't fight the paper tone.

### 2.5 Automated guardrails (extend `assert_no_overlap`)

The current check missed Fig 2's overflowing boxes and every collision between text and data. Add these checks:

```python
def lint_figure(fig, min_pt=7.0):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]
    fb = fig.bbox
    for t in texts:
        if t.get_fontsize() < min_pt: raise AssertionError(f"<{min_pt}pt: {t.get_text()!r} {t.get_fontsize()}")
        bb = t.get_window_extent(r)
        if not (fb.contains(bb.x0, bb.y0) and fb.contains(bb.x1, bb.y1)):
            raise AssertionError(f"clipped: {t.get_text()!r}")
    # text must sit inside its FancyBboxPatch if it is tagged with .nf_box = patch
    for t in texts:
        box = getattr(t, "nf_box", None)
        if box is not None:
            pb = box.get_window_extent(r).padded(-2)
            if not pb.contains(*t.get_window_extent(r).p0) or not pb.contains(*t.get_window_extent(r).p1):
                raise AssertionError(f"overflows box: {t.get_text()!r}")
    # legends must not cover data: compare legend bbox with every Line2D vertex in display coords
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None: continue
        lb = leg.get_window_extent(r)
        for ln in ax.get_lines():
            pts = ln.get_transform().transform(ln.get_xydata())
            if any(lb.contains(x, y) for x, y in pts):
                raise AssertionError(f"legend covers data in {ax.get_title()!r}")
```

Add a **colour-blindness check** as well: render each figure through Machado-2009 deuteranopia, protanopia and tritanopia matrices (the `daltonlens` or `colorspacious` package) and save `figN_cvd.png` contact sheets next to the originals for a visual check. Then assert a minimum CIEDE2000 colour difference of 10 or more between any two series colours that share an axis, under all three simulations.

---

## 3. Figure-by-figure review

### Fig 1 — pipeline overview (`fig1_pipeline_overview`)

**What works:** the left-to-right flow, a real image in each stage, the runtime bar on a shared time axis, and the stage colours echoed in the bar.

**Problems:**
- The "10 mm" scale-bar label sits on top of the bar and on bright anatomy, so it's unreadable. The same happens in Fig 5.
- The header baselines don't line up: "Stage 2" and "Output" sit about 80 px higher than the first three.
- Panel 4 is a cartoon while the other four are data. That is an inconsistent visual language.
- Names don't match across the figure: the header says "Stage 0 · Radial prior", the legend says "Stage 0: Symmetry", and the caption says "dark-core symmetry". "IRGM" sits under Stage 1 here, but Fig 2 calls the same gate "Stage 2B". "OOF", "IRGM", "MPNN+GraphTF" and "Triplet + bundle pruning" are jargon for a figure that is supposed to orient a non-specialist.
- Stage 2B is vermilion in the legend and bar, the same colour as the output, so the meaning of "ours" leaks into a stage.
- The bars are labelled twice ("CPU (4.15 s)" on the left and "4.15 s total (CPU; …)" on the right), in bold, with three numbers each.
- The output panel has no legend: grey means expert and vermilion means prediction, but only the caption says so. The single vermilion tip dot is unexplained.
- The template-plane cropping step described in the caption isn't shown.

**Redesign ("the physicist's three clues"):**
- Add a top row of three cartoon icons with plain titles:
  - "Clue 1 · Find round dark spots" → Stage 0 + Stage 1
  - "Clue 2 · Link spots across slices" → Stage 2A
  - "Clue 3 · Check the whole bundle; stop at the tip" → Stage 2B/2C
- Below that, the real data strip: MRI → spots (dots, not speckle) → linked 3D tracks in a **real 3D render** of Case 017, with slices as translucent planes → final bundle. Colour the final bundle by agreement: vermilion for predicted, black dashed for expert-only (missed), purple for false. Make the cropping step a thin box on the MRI tile with the note "automatic template crop".
- **Runtime bar:** keep a single bar per device, label segments inside them in 7 pt white or ink text (S0 0.27 s, …), and put one plain total at the right end: "4.15 s CPU". Move the "1.26 s std / 5.85 s hi-res" breakdown to the caption, which already states it.
- **Delete Fig 6c**: it duplicates this bar exactly.

### Fig 2 — architecture (`fig2_architecture_modules`)

**Problems:**
- Text overflows its boxes in four places: "Implant-region gate (Stage 2B)" and "center = median XY … (conf. ≥ 55%)" spill out on both sides; "Seed tracking + 4× Triplet MPNN" and "edge update from hᵢ, hⱼ, eᵢⱼ…" overflow; "Bundle Graph Transformer" touches its border; "2D Gaussian peak" touches its border.
- The arrows from "Up+Cat" fan out and run through the head boxes.
- The Stage 0 arrow feeds the gate, not the network input, which contradicts the caption ("MRI and Stage 0 maps feed" the U-Net).
- The gate is drawn in Stage 1 blue but labelled Stage 2B.
- In panel b, an orange dash-dot track (orange being Stage 0's colour) appears next to a green track with no legend. A reader can't tell that orange means "rejected / vessel-like".
- The tip localizer and conformal interval described in the caption are missing.
- "Prunes 77% of vessel FPs" is unsupported (see §0, item 4).

**Redesign as three panels (a, b, c):**
- **(a) Stage 0 operator, mechanism and data together.** Draw a 2D patch: on the left, a catheter cross-section with the six diameters × four radii (1.35, 1.85, 2.5, 3.3 mm) as concentric dotted rings; on the right, a fascia edge with the same rings.
  - Show the "darker of the two opposite ring points" rule as small paired dots, filled dark and light.
  - Show the rank-2/rank-3 robust ring as a short sorted bar of the six values, with ranks 2 and 3 highlighted.
  - Below, show the real S₀ response on the same two patches. The catheter lights up; the edge does not.
  - Add an annotation: "one dark neighbour tolerated, e.g. a touching twin catheter".
- **(b) Stage 1 + Stage 2A/2B.**
  - Use one encoder-decoder silhouette (a "U" of trapezoids) instead of five rectangles, with the five heads as small output thumbnails of real maps from Case 017 rather than text boxes.
  - The gate is a dashed circle on a real image, labelled with its rule in two short lines.
  - The graph cartoon uses green for accepted and purple dashed for rejected, with a legend.
- **(c) NEW: tip localizer + split-conformal error bar.** Details in §4, S3.

### Fig 3 — rainclouds (`fig3_raincloud_distributions`)

**Problems:**
- The reference marks mean different things in each panel and carry no labels (§0, item 3).
- **The stats label boxes mask the clouds.** The white `bbox` (alpha 0.78) behind "2.26 / 27.42 (+113 off-scale)" erases the cloud from x ≈ 7 to 15 mm. That is the abrupt "cut" visible in every tip cloud.
- In panel b, between 112 and 301 tip errors per method fall beyond 15 mm. The P95 values (27–63 mm) are all off-axis, so the panel hides the very tail the paper discusses.
- There are eight hues, including off-palette purple, brown and steel blue. The two NeedleFinder variants are vermilion and *green*, which looks like Stage 2.
- "Inter-observer (CY vs LB/AMe)" uses raters' initials, which mean nothing to a reader.
- The diamond markers are too small (`s=11`).

**Redesign:**
- **Panel a (shaft):**
  - Draw the human IQR as a grey band labelled inside the axis at the top: "Expert vs expert (IQR)". Draw the human median as a black dashed line labelled "expert median 0.49 mm". Drop the P95 line, or keep it dotted and labelled "expert P95".
  - Colour NeedleFinder rows vermilion, with the "+ Synth" variant as open or light vermilion. Make every baseline a grey ramp (b1–b5). Separate the two groups with a thin rule and a small-caps label on the left: "NeedleFinder" / "Published baselines".
- **Panel b (tip):** use a **log x-axis from 0.3 to 100 mm**, so nothing is off-scale and the "(+N off-scale)" notes disappear. Draw the human median as a labelled dashed line ("expert median 3.24 mm"). If you want a band here too, use the tip IQR; otherwise state "median only available".
- **Stats text:** move it out of the plot area into a right-hand column. Make it a pseudo-table in tabular figures: `median | P95`, with column headers once at the top. This gets rid of every bbox.
- **NEW panel c: "three-cornered hat".** Draw a triangle whose vertices are Team 1, Team 2 and NeedleFinder. At each vertex, draw a circle whose radius is that source's intrinsic shaft standard deviation, with the value next to it: 0.35, 0.48 and 0.39 mm (from the three-cornered-hat paragraph in the manuscript, n = 432).
  - This is the single most persuasive "as good as a second expert" visual. It's also very Clopath: one glyph, one idea.
  - Keep tip σ_z out of it, or show it faded with the note "ranking only", because the manuscript explicitly says the tip values are a ranking.

### Fig 4 — Pareto and few-shot (`fig4_pareto_and_fewshot`)

**Problems:**
- The F1 iso-line labels ("F1=0.80" etc.) sit outside the axes at the top-left, and "F1=0.80" is clipped at the figure edge.
- "τ=60%" collides with the star and the line.
- "All 1,063 F1 (89.1%)" is struck through by the vermilion line and marker.
- The panel b title touches the right edge of the figure.
- `F1_adj` appears with a raw underscore.
- The off-scale baselines are relegated to a 4.9 pt grey text box.
- The arrow to the star crosses the curve.

**Redesign:**
- **(a) Make it a path up the frontier.**
  - Connect the stage points in pipeline order with thin arrows: Physics + HybridNet (`st0`) → NeedleUNet3D + tubes (`st1`) → Triplet MPNN only (`st2`) → NeedleFinder at τ = 50% (vermilion star) → + Literature Fusion.
  - With the stage ramp, the colours themselves show the progression from light blue to dark blue to vermilion. This turns a scatter plot into a narrative.
  - Write the iso-F1 labels **along the curves**, rotated, inside the axis ("F₁ = 0.90").
  - Draw the τ sweep as small vermilion ticks with only the end labels (τ = 25%, τ = 60%), offset away from the line.
  - Draw off-scale baselines as grey downward triangles **on the lower spine** at their true recall, labelled "below 60% precision" with values in a footnote. Or extend the y-axis to 35% with a broken-axis marker. Grey-ramp all published baselines; PMB 2019 is already dark grey.
- **(b) Few-shot.**
  - Label all lines at their right ends, outside the plot area.
  - Rename "F1_adj" to "Long shafts (F₁, adjusted)" and define it in the caption.
  - Move the two zero-shot reference lines to the left side, labelled "zero-shot: physics only 75.3%" and "zero-shot: synthetic pre-training 78.2%". Use dark grey dotted and dashed lines rather than `st0`/`st2`, because these aren't pipeline stages.
  - Put "1 patient: 79.7%" as a callout at x = 1.
  - Use a symlog or categorical x-axis with equal spacing, since k ∈ {1, 5, 10, 20, 43}.
  - The caption claims k = 0 is included; the axis starts at 1. Plot the zero-shot values as points at x = 0 on a broken axis, or fix the caption.

### Fig 5 — qualitative (`fig5_qualitative_mri_3d`)

**Problems:**
- No symbol key: white ring means expert and vermilion × means prediction, but only the caption says so.
- Missed catheters and false detections aren't distinguished.
- The "10 mm" label is clipped at the left edge in row b.
- The "z = 15/30" text sits on noisy anatomy without a halo.
- Column titles are repeated in both rows.
- "OOF" and "IRGM" jargon again.
- The "3D bundle" column is a 2D projection with no axes, orientation or depth cue.
- The obturator gate described in the caption isn't visible (§0).
- The S₀ speckle overlay reads as noise.

**Redesign:**
- Column titles only on the top row, in plain words: "MRI slice + detections", "Clue 1: dark-core map", "Clue 1′: network centre-lines", "3D result".
- Row labels on the left: "a · Dense implant (Case 017, 26 catheters)" and "b · Adhesion: 10 of 23 catheters touch a neighbour (Case 031)".
- **Detection markers, CVD-safe and shape-coded:**
  - matched: vermilion filled dot inside a black ring;
  - missed: black **dashed** open ring;
  - false: purple open diamond.
- Add a compact key strip under the figure: ● matched ◌ missed ◇ false.
- Add a zoomed inset on Case 031's touching cluster, with a 3 mm scale.
- Add A/P/L/R orientation letters in grey 7 pt.
- **3D column:** render true 3D. Use a matplotlib `Axes3D` with `elev=15, azim=-60`, light depth shading via alpha by depth, and a mm grid on the floor; pyvista offscreen gives nicer tubes.
  - Expert tracks: thin black. Prediction: vermilion tubes. Missed: black dashed. False: purple.
  - Mark tips as small spheres. Draw the slice shown in column 1 as a translucent plane so the 2D and 3D views are visibly linked. This is the key "mechanism-to-data" link.
- Give the scale bar and z-label a white text halo with a black stroke, placed bottom-right, away from the image edge.

### Fig 6 — specificity and runtime (`fig6_specificity_and_runtime`)

**Problems:**
- In panel a, the legend sits over the step curves; the blue and grey curves pass underneath it.
- The two best curves are orange and vermilion, the worst colour-blindness pair (§0).
- Arrow callouts cross the plot.
- In panel b, bars are unsorted (660 s sits between 120 s and 14.8 s). The "≥2 min" value is drawn as exactly 120 s. The NF bars use an alpha-blended, off-token orange.
- Panel c duplicates Fig 1.

**Redesign:**
- **(a)** Title: "The physics gate removes almost all false tracks on needle-free scans".
  - Use a log-ish ECDF x-axis.
  - Label curves at their right ends: "raw network 6.38/vol", "NF, no gate 4.10/vol", "NF + gate 0.33/vol", "self-supervised + gate 0.125/vol".
  - Colours: grey for the first two, vermilion for the NF + gate system, and vermilion dashed (not orange) for the SSL variant, which is an ablation of NF.
  - Mark the two headline points (87.5% at 0 FP, 100% at ≤ 1 FP) with small labelled dots and no long arrows.
- **(b)** Sort by runtime as a dot plot (lollipops) on a log axis. Draw "≥ 2 min" as an open-ended bar with a right-pointing arrowhead. Mark reported-not-measured values with open circles and a legend glyph instead of a superscript ʳ. Show the 46–51 min manual-tracing figure as a grey vertical band labelled "manual tracing", using the number already in the Introduction. That makes the "seconds vs minutes" story visible.
- **(c)** Delete; it duplicates Fig 1.
- Move panel a next to the calibration figure (narrative slot 7, "trust"). Move panel b into the clinical figure (slot 8).

### Fig 7 — calibration and reason codes (`fig7_calibration_and_reasons`)

**Problems:**
- Caption and figure disagree (§0).
- The legend hides the calibrated point at p ≈ 0.75.
- The twin-axis histogram (grey right axis) competes with the main data.
- In panel b, "kept" vermilion vs "lost" brick red is the vermilion/red colour-blindness pair.
- White gridlines run through the bars.
- Number formatting is inconsistent ("1,015" vs "n = 1118").

**Redesign:**
- **(a)** Standard reliability diagram.
  - Raw: grey dashed line with open circles. Calibrated: vermilion solid line with filled circles. This **matches the existing caption**, so only the code changes.
  - Label "ECE 0.101 → 0.059" once, as a before→after pair (house style rule 3).
  - Move the histogram into a short strip **under** the x-axis that shares it, with no second y-axis.
  - Label the diagonal "perfectly calibrated" along the line.
- **(b)** Rows: kept / below threshold / duplicate. Segments, in fixed order:
  - real catheter kept: vermilion;
  - false track: purple with `////` hatch;
  - redundant real: light grey with `....` hatch;
  - lost real: white fill with a thick black outline.
  
  Label each segment with its count inside, or just outside if narrow. Use thousands separators everywhere ("1,118"). Set `ax.set_axisbelow(True)` so the grid sits behind the bars.
- **Add a connection to the operating room:** draw 2–3 small "reason chips", rounded tags as the Slicer module would show them (for example "below threshold", "duplicate of #12"). Use only reason names that actually exist in the code; check `reason code` around manuscript line 280.

### Fig 8 — clinical translation and generalization (`fig8_clinical_translation_and_generalization`)

**Problems by panel:**
- **(a)** The x-axis runs to 64% when the largest value is 37.1%. The A-vs-B comparison is buried in "27.6% / 37.1% (γ 64%)" text strings, and the bottom legend covers part of the axis.
- **(b)**
  - The legend sits over the random-review line and the "(90.2% conf. cov)" label is clipped.
  - The red line at 10 mm has no label.
  - Only k = 8 is annotated, but the configuration carried through the dose simulation is **k = 2**.
  - P95 and mean share one axis with no visual grouping.
- **(c)**
  - The panel mixes ablation, transfer and CT recall on one "F1 or recall" axis that runs to 154%.
  - Colours are arbitrary.
  - It uses code identifiers ("auto k_lim", "X[1:4] = 0", "template ACSV").
  - The T1 VIBE result is missing.
  - The title wraps, and its baseline doesn't line up with panel d's.
- **(d)**
  - A dual x-axis puts two unrelated experiments on one plot.
  - The legend says "(bottom axis)" four times.
  - Colours don't match the caption.
  - The 4.9 pt legend is the smallest text in the paper.

**Split into two figures.**

**Fig 8 (new) — "From geometry to dose, and to the operating room":**
- **(a)** Dumbbell plot per method: a grey open circle for Protocol A, a vermilion filled circle for Protocol B, joined by a line. x from 0 to 40%. Show γ pass rate as a right-hand text column with the header "γ 2%/2 mm". Order with NeedleFinder at the top. Show "Human rater 2" as a black row so the human-level comparison is explicit.
- **(b)** Two small stacked panels sharing k:
  - top, P95 tip error: random (grey dashed, triangles), uncertainty-guided (vermilion solid), oracle (black dotted);
  - bottom, mean tip error (vermilion).
  
  Label the 10 mm line "10 mm" at its right end. Mark k = 2 with a vertical band labelled "k = 2: used in dose simulation" and k = 8 with "P95 9.90 mm". Put "simulated, oracle tip placement" in grey italics inside the panel (house style rule 4). Move the coverage figure (90.2%) to the caption.
- **(c) NEW workflow panel** (see S4).

**Fig 9 (new) — "Beyond the training domain":**
- **(a)** Dot plot of F1 by domain, x from 0 to 100%.
  - Groups, separated by small-caps labels: *Other organ* (prostate, two operating modes), *Other MRI sequences* (wbSSFP 92.9%, thick 2D T2 76.9%, **T1 VIBE 56.1%**), *4D serial* (82.8%), *CT* (recall 79.2%, drawn as an open marker with a "recall" glyph so the metric difference is visible).
  - Show baselines as grey ticks on the same row where they exist (clDice 71.8% on prostate, 57.6% on thick 2D; PMB 59.4% and Stage 0 61.5% on VIBE).
  - Plain-language labels only, for example "Prostate implant, reference-free count".
- Move the physics-zeroed ablation (0.0% F1) to the ablation story (Fig 4 or a small inset): "remove the physics channels → nothing is found". That is a powerful one-liner, but it isn't a transfer result.
- **(b)** Depth deflection vs insertion depth: three lines for median, mean and P90 in vermilion with different line styles, labelled directly at the right end.
- **(c)** Lateral push vs distance to the new needle, as its own small panel with its own x-axis (dark grey, diamonds).

---

## 4. New illustrations to commission (generate as vector, same tokens)

**S1 — "Why catheters are hard to see" (narrative Fig 1).** Two rows:
- **Top: physics cartoons** of four axial cross-sections:
  - a 2 mm (6F) plastic catheter: no mobile protons give a proton-density void, with a thin ring for the mild Δχ interface;
  - a flow-void vessel;
  - a fascial plane edge;
  - rectal or vaginal air.
- **Bottom:** a real 9 × 9 mm crop of each from Case 017, plus its radial intensity profile.
- **Overlay** the Stage 0 diameters on each, with a ✓/✗ for "bright in almost every direction".
- **Be honest that a vessel also passes Stage 0.** Show it with a ✓ and the annotation "same in 2D, different in 3D → Stages 1–2". The manuscript's story then flows naturally.
- **Wording caution:** the brief says "asymmetric susceptibility void", but the manuscript (line 45) claims only a *mild* Δχ transition and a proton-density void. Draw what the text says; don't add an asymmetry claim.
- **Add a thin time bar:** 100-min procedure, with 46–51 min of manual tracing highlighted. Clinical stakes in one glance (Kapur).

**S2 — the Stage 0 operator.** This is the mechanism half of Fig 2a. To add one quantitative touch, plot S₀ against offset from a catheter centre and from an edge, from real data.

**S3 — tip localizer + conformal intervals** (Fig 2c, or a standalone figure). All numbers are from manuscript lines 158–164; no new claims. Left to right:
1. A 3D track whose cranial end is sampled at 91 steps from −25 to +20 mm in 0.5 mm steps, with a slice-thickness ruler showing why the tip is ambiguous.
2. A **24 × 91 heatmap** of the real feature sequence for one tip: 22 measurements plus the 2 epistemic-uncertainty channels, rows grouped and labelled (intensities, derivatives, S₀, Stage 1, context, uncertainty), single-hue colour map.
3. A dilated-convolution receptive-field cartoon (dilations 1, 2, 4), then a self-attention arc diagram.
4. Output: a 91-bin stopping distribution with the expert tip marked. Contrast a sharp, confident case with a broad, uncertain one; entropy turns into σ̂_tip.
5. A conformal inset: a histogram of rᵢ = E_tip/σ̂ on calibration patients, the q̂ quantile marked, and the resulting radius r̂ = q̂·σ̂ drawn as a bracket along the track. Add the note "90.2% of test tips covered at nominal 90%".

**S4 — operating-room workflow** (Fig 8c). Two parts:
- **(i) A 3D Slicer mock-up.** Use a real screenshot of the module if available; otherwise a clean vector re-drawing. It should show:
  - the bundle;
  - a **tip list ranked by uncertainty bar**, with the top two highlighted;
  - a rejected-candidate ghost track with its reason chip and a "restore" button, annotated "0.65 restores per patient: recall 91.4% → 93.0% (simulated)".
- **(ii) A timeline on a log time axis.** Manual tracing 46–51 min (reported) against NeedleFinder: click 4 s (measured, headless), two-tip check 18 s (modeled), count reconciliation 5 s (assumed).
  - Encode the evidence level of each bar: solid fill for measured, hatched for modeled, outline only for assumed.
  - This puts the house rule "say what was simulated" directly into the graphic.

**S5 (optional) — graphical abstract / plain-language figure.** A single 76 mm-wide graphic: MRI → one click → 3D bundle → "two tips to check". Reuse the S4 elements.

---

## 5. Caption pattern (all figures)

Use the same five-part structure for every caption:
1. **One bold claim sentence:** the finding, not the method.
2. One plain sentence on what is plotted.
3. An encoding key in words ("vermilion: NeedleFinder; grey: published methods; dashed: expert-vs-expert median").
4. n, cohort and statistics.
5. The honesty clause, in the same caption ("simulated", "modeled, not timed", "oracle tip placement").

Drop parenthetical number walls. Each headline number should appear in the figure *or* the caption, not both. Remove code identifiers from captions (`NeedleUNet3D`, `MSD Task05_Prostate`, `Case 100`, `Cases 68–71` in monospace); say "an open prostate dataset" and cite it.

---

## 6. Do-not-change checklist

Every number in this review is quoted from the current figures or manuscript. The redesign changes encoding and layout only; every value comes from the same JSON keys already used in `generate_pmb_figures.py`. Things to verify while implementing:
- **The "77% of vessel FPs" label (Fig 2).** Find its source or delete it.
- **Fig 3 reference lines.** Decide median vs IQR vs P95 and make the caption say the same thing.
- **Fig 4b k = 0.** Plot the zero-shot points at k = 0 or remove "k = 0" from the caption.
- **Fig 7 and Fig 8d captions.** Make the code match the captions, or the captions match the new code.
- **T1 VIBE (56.1%).** Add it to the generalization figure.
- **File numbering.** Rename the files to follow print order (calibration is Figure 6 in print).

Suggested implementation order, by payoff per hour:
1. §2.3 sizes, font and 160 mm width, plus the §2.5 lint.
2. §2.1 palette tokens.
3. The §0 caption mismatches.
4. Figs 3, 6, 7 and 8 rework.
5. New S1, S3 and S4.
6. Figs 1, 2 and 5 redesign.

I didn't regenerate or edit anything, as you asked, so I haven't updated the vault log either. Once changes land, add a `wiki/log.md` entry with the palette and typography decisions. To view the current build next to this review:

```bash
rsync -avz think:/mnt/SSD/github/needlefinder/vault/paper/needlefinder_2026_pmb.{pdf,html} ./
```
