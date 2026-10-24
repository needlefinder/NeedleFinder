Probing commands are being blocked by the sandbox, so I am stopping the metric checks and writing the review from the files and rendered figures I have already inspected. Print sizes below are derived from the style files: figures are drawn at 170 mm and placed at 160 mm, so every figure font shrinks by a factor of 0.94 in the PDF.

# Visual design review of the NeedleFinder 2026 PMB manuscript

**Verdict.** The figure suite is scientifically honest and already uses a colorblind-safe base palette, but it does not yet tell the story the prose tells. The prose was rewritten in the Clopath and Kapur style, with "why before how" and operating-room grounding. The figures were not. They still speak in module names (IRGM, MPNN, GraphTF, OOF), their text is too small to read in print or on screen, four of eight have collisions or clipping, two captions disagree with their figures, and the three mechanisms that make the method work (dark-core symmetry, peak-not-mask, tip profiling with conformal intervals) have no illustration at all. Below are the five changes that matter most, then the style system, then a figure-by-figure plan with new schematics.

**The five changes that matter most**

1. **Raise all figure type to 8 pt minimum at final size and draw at the placed width.** Today the base is 7 pt and annotations go down to 4.9 pt before the 0.94 print rescale, so printed labels are 4.6 to 6.6 pt. On the HTML page the wide column is about 845 px, which makes a 7 pt label roughly 12 px tall next to 19 px body text. Set the figure width to 160 mm so one point stays one point.
2. **Add three mechanism schematics.** Why a 6F catheter is a round dark core while a fascial plane is a one-sided edge, how the six-diameter half-ring operator scores them differently, and how the 91-step tip profile becomes a calibrated interval. These are the "why" panels and they are missing.
3. **Add one clinical workflow figure.** The 3D Slicer loop that the Discussion describes in words: Detect Needles, review the two widest-interval tips by sliding along the shaft, reconcile the count with reason-coded restoration. Mark every modeled time as modeled inside the figure.
4. **Fix the caption-figure mismatches and the missing T1 VIBE bar.** Figure 7's caption describes dashed-versus-solid lines that are not drawn. Figure 2's caption describes a tip localizer that is not drawn. Figure 8c omits the post-contrast T1 VIBE result (56.1% F1) that the text reports, which reads as selective reporting.
5. **Make hue mean one thing across the whole paper.** Vermilion is always NeedleFinder, black is always the human reference, greys are always literature baselines, and the three stage colors appear only in pipeline figures. Figure 3 currently assigns four decorative hues to baselines, and Figures 7b and 8d pair vermilion with a second red.

## Style system: concrete specifications

**Palette tokens** (`nf_palette.py`). The Okabe-Ito core is right. The problems are the off-palette additions and the unused or duplicated tokens.

- Keep the eight Okabe-Ito hues and the stage mapping (orange for Stage 0, blue for Stage 1, green for Stage 2, vermilion for the output). This set survives deuteranopia, protanopia and tritanopia by design.
- Remove the off-palette hues: the Figure 3 blue, purple and brown, the dark red used for "lost" and "P90", and the blue-grey used for Protocol A. Under protanopia and deuteranopia the dark red and vermilion converge toward the same brown, and the purple converges toward the blue. None of these hues carries meaning that a grey or a line style could not carry.
- Encode "failure" or "lost" as black with a white hatch, never as a second red. Encode the human reference as black with dash-dot lines. Encode literature baselines as a four-step grey ramp with distinct marker shapes.
- Delete the duplicate keys (two PMB greys, two nnU-Net greys) and either use or delete the unused tokens (false-negative black, false-positive pink, ROI sky blue). The missing-catheter token is exactly what Figure 5 needs for the three catheters it does not find.
- Make the label and marker dictionaries the single source of truth. Today Figures 3, 4, 6, 7 and 8 each hard-code their own method names and markers, which is why the same configuration is called "4b", "NeedleFinder 2026 + Synth", "NF 2026 (4b)" and "NeedleFinder 2026" in different places. One name per configuration, used in tables, figures and prose.

A proposed encoding table, to be written at the top of the palette file and obeyed everywhere:

| Entity | Hue | Marker | Line |
|---|---|---|---|
| NeedleFinder 2026 (default, 4b) | vermilion | filled circle | solid, 1.8 pt |
| NeedleFinder + literature priors (4d) | vermilion | filled star | solid, 1.8 pt |
| Human reference (Team 1 or inter-observer) | black | horizontal tick | dash-dot |
| Literature baselines | four greys | square, triangle, diamond, plus | dashed |
| Stage 0 / Stage 1 / Stage 2 (pipeline figures only) | orange / blue / green | n/a | solid |
| Lost or missed | black | cross | hatched fill |

**Matplotlib style** (`nf.mplstyle`). Proposed values, all at final placed size:

```
font.family: sans-serif
font.sans-serif: Source Sans 3, Fira Sans, TeX Gyre Heros, Helvetica, Arial
mathtext.fontset: custom        # map rm/it/bf to the same sans
font.size: 8
axes.labelsize: 8.5
axes.titlesize: 9.5             # panel titles, bold, left-aligned sentences
xtick.labelsize: 7.5
ytick.labelsize: 7.5
legend.fontsize: 7.5
lines.linewidth: 1.5
lines.markersize: 5
axes.linewidth: 0.7
svg.fonttype: none              # real text in the HTML SVGs
```

Set the two width constants to 84 mm and 160 mm, not 170 mm, and never place a figure at a width other than the one it was drawn at. Add a hard floor of 7 pt for any annotation and a 11 pt bold lowercase panel letter placed by one shared helper. The helper already exists in the palette file but is unused, so every figure places its letter differently (Figure 2 at 9 pt, the others inside the title string at 6.8 to 7.5 pt).

DejaVu Sans is the weakest typographic choice on the page. It is wide, mechanical and clashes with EB Garamond. Source Sans 3 or Fira Sans pair well with a Garamond, have true tabular figures for axis ticks, and are free. I could not verify which of these fonts is installed, so keep the fallback chain.

**Overlap checker.** The current assertion only tests text against text. That is why it passed Figures 2, 6 and 8 where text collides with box borders, legends sit on curves, and a label is clipped outside the axes. Extend it to test every text box against legend and patch boxes, and to assert that every text box lies inside the figure canvas.

**Print edition** (`preamble.tex`, `build_paper.sh`). The page design is handsome and should be kept. Three adjustments:

- EB Garamond has a small x-height, so 11 pt reads like 10 pt in Times. Either go to 12 pt on the same 120 mm measure or keep 11 pt and raise the leading to 1.10. Captions at `\small` are the hardest text on the page to read; set them at `\normalsize` or 10.5 pt.
- Table 2 is set in `\scriptsize` (7 pt) with ten columns and three blocks. Move Part B to a second table or to supplementary material and set Part A at `\footnotesize`. The same applies to Table 6.
- The superscript "r" that marks reported rather than measured runtimes is used in Figure 6b and the tables but is defined only in the table notes. Define it once in the caption of every figure that uses it.

**HTML edition** (`build_html.py`). The 19 px Garamond body, the 29.5 rem measure and the margin-hung labels are excellent. Figures are the weak link: they are 300 dpi PNGs scaled into an 845 px column, so figure text lands at roughly half the body size. Embed the SVGs inline (fonts as real text or outlined paths), add a click-to-enlarge lightbox because every figure is 160 mm wide and dense, and keep the multiply blend for light mode. In dark mode the PNGs sit on a light mat, which is fine, but inline SVG would allow a true dark variant with the same tokens.

## Figure-by-figure critique and redesign

Each figure gets a take-home title (a sentence, in the Clopath manner), the defects I saw in the rendered PNG, and the redesign. Every number cited is the manuscript's own.

**Figure 1, pipeline overview.** Take-home: "A 45-minute manual tracing task becomes a 4-second automatic reconstruction plus a short check." Defects: tile titles are in module jargon (ResUNet+IRGM, MPNN+GraphTF, OOF) that the prose deliberately avoids; the Stage 2 tile is a cartoon whose grey crosses are never explained; the legend is drawn with square glyph characters at 5.5 pt; only the first tile has a scale bar; and the runtime bar duplicates Figure 6c exactly. Redesign as two rows. Row 1 is the clinical strip: operating room insertion through the perineal template, T2W scan, then two branches, manual tracing at 45 to 50 minutes versus NeedleFinder at 4.15 s plus a modeled 18 s review, ending in the dose plan. Row 2 keeps the five tiles with plain titles: "Dark-core symmetry", "3D U-Net centerlines and tilt", "Link, stitch, judge the bundle", "Tips with uncertainty bars". Drop the runtime bar from this figure.

**Figure 2, architecture.** Take-home: "Each stage answers one failure of the stage before it." Defects: the title and first line of the implant-region gate box overflow its dashed border on both sides; the message-passing box text overflows its border in panel b; the gate is labeled "Stage 2B" inside the Stage 1 panel; the caption promises a 91-step tip localizer that does not appear; "Prunes 77% of vessel FPs" has no anchor in the prose, which reports 43% of false tracks removed by Stage 2A and 80.6% of false candidates removed at threshold, so please confirm its source or remove it. Also missing are the three ideas the Methods call crucial: six input channels, in-plane-only pooling on anisotropic voxels, and the merged-mask-versus-two-peaks contrast. Redesign as a four-panel mechanism figure: (a) anisotropic voxel cartoon with in-plane-only pooling and the four heads; (b) the touching-pair strip, one MRI crop, one merged mask blob, two separate peaks; (c) the three Stage 2 rules as cartoons, a kink that two plausible links form, a dropout gap stitched when the midpoint agrees within 2.2 mm, and a vein track outside the bundle corridor; (d) the tip localizer as a 91-step heat strip from -25 to +20 mm with its 91-bin distribution, the offset, and the conformal radius drawn as a bar.

**Figure 3, rainclouds.** Take-home: "Along the shaft every method is sub-millimeter; at the tip only NeedleFinder halves the error." Defects: the grey band and dashed lines are unlabeled in the panel; the tip-panel band has zero width because both edges are set to the median; four decorative hues on baselines imply a meaning they do not have; "+113 off-scale" labels crowd the top of each row; tick and row labels are 6.8 pt. Redesign: baselines in greys, both NeedleFinder rows in vermilion (star for the literature-prior variant), the human inter-observer reference as a black dash-dot line with an in-panel label "two human teams, 40 scans: 0.49 mm" and "3.24 mm". Give the tip panel a broken axis or a log axis so the P95 values are visible and the off-scale counts disappear. Name the two NeedleFinder rows as the table does.

**Figure 4, Pareto and few-shot.** Take-home: "Precision, not recall, is what the bundle stage buys; five annotated patients already match the 2019 pipeline." Defects: the "F1=0.80" iso-line label is placed outside the axes at the top-left corner; the 0.75 iso-line is unlabeled; the overflow box for the sub-57% baselines is set at 4.9 pt; the right-hand label reads "F1_adj" with a raw underscore. Redesign: extend the y axis to 20% so all baselines sit on the plot as grey markers with direct labels, and label the iso-F1 curves inside the frame along the curve. In panel b, write "F1 on long shafts (obturator-adjusted)" and mark the 2019 pipeline's 83.1% as a grey reference line so the "k = 5 matches 43-patient baseline" claim is visible.

**Figure 5, qualitative.** Take-home: "Stage 0 lights catheters and vessels alike; Stage 1 separates touching pairs; Stage 2 returns a clean bundle." Defects: no symbol legend for white circles and vermilion crosses; nothing marks which of the 26 and 23 reference catheters are missed; overlays are too small to read at this scale; the 3D panels have no orientation cue; blue peak overlays on a grey image are low in luminance contrast. Redesign: add 2x zoom insets on one vessel that Stage 0 scores but Stage 1 rejects, one touching pair, and one dropout that the stitcher bridges; draw missed reference catheters in black (the unused token); add a small cranial arrow and a 10 mm bar to the 3D panels; use the sky-blue token with a white halo for peaks; put the symbol key as one line under the row.

**Figure 6, specificity and runtime.** Take-home: "With the symmetry gate, 42 of 48 needle-free scans produce zero false tracks." Defects: the legend in panel a sits on top of the blue dashed and grey dotted curves; the two annotation arrows cross the curves and the text sits in the plot's middle; vermilion and orange are both solid lines; panel b uses log-scale bars whose lengths are not meaningful, mixes an assisted human time with algorithm times, and does not explain the superscript "r"; panel c duplicates Figure 1. Redesign: panel a with the legend outside the frame, direct labels at the curve ends, and dash-dot for the 0.33-per-scan line; panel b as a lollipop on a log axis with hollow markers for reported times and filled markers for measured ones, defined in the caption; fold the stage breakdown into the lollipop as a small stacked inset so the figure is two panels.

**Figure 7, calibration and reason codes.** Take-home: "The confidence a physicist sees is calibrated, and every discarded track carries a reason." Defects: the caption describes "before (dashed) and after (solid vermilion)", but the figure draws two solid lines, raw in vermilion with circles and scaled in green with squares; the "lost" segment is a second red next to vermilion in the same bars; the right axis label is in 6.5 pt grey. Redesign: raw as a grey dashed line, scaled as solid vermilion to match the caption; panel b as a flow from the 1,960 candidates into kept, below-threshold and duplicate branches with real, false, redundant and lost as the terminal colors, lost in hatched black. This flow is also what the Slicer reason-code list shows, so it links to the workflow figure.

**Figure 8, clinical translation.** Take-home: "Two reviewed tips per patient bring median target dose error from 4.2% to 2.8%, and the method transfers to other sequences, except post-contrast T1." Defects: the post-contrast T1 VIBE result (12 scans, 217 catheters, 56.1% F1) is absent from panel c; panel c's axis runs to about 150% to make room for labels; it mixes F1 and recall without a visual cue; the label "Physics X[1:4] = 0" disagrees with the prose's $X_{1:3} = 0$ and "985x peak drop" has no anchor in the prose; panel b's legend covers the random-review curve and its 10 mm line is unlabeled; panel d has four series on two x axes with a 4.9 pt legend over the data. Redesign as five panels: (a) a before-and-after dumbbell per method from unreviewed to top-2 review, which is the house style's preferred pair; (b) the triage curve with the legend outside and "10 mm" written on the line; (c) transfer bars capped at 100% with the T1 VIBE bar included in grey, recall bars hatched; (d) bending versus depth alone; (e) lateral push versus distance alone, three points, with the 1.44 mm and 1.00 mm medians labeled.

**New figures to add.**

1. **Physics figure (new Figure 2).** Left: real 2x crops of a catheter, a vein, a fascial plane and an air pocket on T2W MRI. Middle: the six diameters and four radii drawn on each, with the two ring points per diameter and the "darker of the pair" rule. Right: the sorted six values and the resulting score, high for the catheter, low for the edge, tolerant of one dark neighbor for the touching twin. This is the one figure a clinical reader will remember.
2. **Cohort and validation figure.** A CONSORT-style box diagram of the 54 patients and 1,063 catheters (759 long, 304 obturator), the 40 dual-traced, 23 adhesion and 38 CT-paired subsets, the 51 external volumes, the 243 secondary scans and the 23 serial scans, next to a drawing of the nested 5 x 5 split with the locked outer vault. Reviewers misread nested cross-validation more than anything else, and Table 1 alone does not prevent it.
3. **3D Slicer workflow figure.** A screenshot or faithful mockup of the module: the Detect Needles button, the tip list sorted by interval width, the slide-along-shaft correction, and the count reconciliation with the reason-coded candidate that one click restores. A time strip underneath: 45 to 50 minutes manual, 4.15 s automatic, 18 s modeled review, with "modeled, not timed" printed on the strip.
4. **Tip profile example.** One real catheter whose void fades under bowel gas: the 91-step profile, the predicted 91-bin distribution, the 90% conformal radius, and the two human teams' tips. Mechanism next to data, in the Clopath manner.

**Captions.** Open every caption with the take-home sentence, then the encoding, then the data source and n. Keep the caption numbers exactly as they are, but move encoding details that are now only in captions (the grey band, the superscript "r", the white circles) into the panels as direct labels so the figure stands alone.

**Numbers to confirm before redrawing.** None of these need changing in the prose; they need a source or removal in the figures: "Prunes 77% of vessel FPs" (Figure 2), "985x peak drop" and "X[1:4] = 0" (Figure 8c), and the 120 s and 660 s reported runtimes in Figure 6b, which should carry the reported-value marker defined in the caption.

What I could not verify: the exact printed point sizes and the colorblind simulation distances, because the sandbox blocked every Python run. The print sizes above are computed from the style constants and the 160 mm placement. The colorblind judgments rest on the known behavior of Okabe-Ito and of red-versus-vermilion and purple-versus-blue pairs, which are the standard confusions under protanopia and deuteranopia.
