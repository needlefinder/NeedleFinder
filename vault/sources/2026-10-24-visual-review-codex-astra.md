**The paper needs a clearer visual narrative, larger figure typography, and more space—not more decoration.** Its strongest material is already compelling: tiny ambiguous MRI voids become coherent catheter trajectories, uncertainty identifies difficult tips, and simulated review connects reconstruction errors to dosimetry. The figures should make that progression visible.

I inspected the requested manuscript, generator, styles, HTML builder, all eight PNGs, and all six current table sources. I also inspected representative pages and text geometry in `needlefinder_2026_pmb.pdf`, calculated palette contrast, and simulated palette appearance under protanopia, deuteranopia, and tritanopia. **No files were changed.**

Throughout this report, “Fig 6” and “Fig 7” refer to the **requested image filenames**. They currently appear in the opposite order in the manuscript.

**1. Fix these issues before polishing the artwork**

| Priority | Verified finding | Required action |
|---|---|---|
| Critical | Table 2 extends below PDF page 20; the clinical-translation table extends below page 32. Notes are cut off. | Split these tables into continued parts with repeated headers. Preserve every cell and note. |
| Critical | Figure annotations are frequently 4.9–6 pt in the generator, then reduced again in the manuscript. | Redesign at final placement size; establish an 8 pt minimum for ordinary figure text. |
| Critical | Fig 8c omits the post-contrast T1 VIBE result and mixes F1 with CT recall under different matching thresholds. | Restore T1 VIBE and separate metrics and experimental conditions. |
| High | Fig 2 does not actually show the tip localizer or conformal calibration described in its caption. | Add an explicit tip-localization panel. |
| High | Fig 3 uses different human-reference statistics for its two dashed lines without labeling them. | Label shaft P95 and tip median separately; identify the reference cohort. |
| High | Fig 7’s caption describes line styles and colors that differ from the rendered reliability plot. Fig 8d’s caption also misidentifies colors. | Generate caption encoding descriptions from the same figure specification. |
| High | The overlapping-text checker cannot detect text escaping boxes, legends covering data, or page clipping. | Add containment and rendered-output checks. |
| High | Stage colors are reused for unrelated meanings, and some colored text has insufficient contrast. | Separate stage identity, method identity, and decision-status encodings. |

There is also a numbering trap:

| Source file | Current manuscript number |
|---|---|
| `fig6_specificity_and_runtime` | Figure 7 |
| `fig7_calibration_and_reasons` | Figure 6 |
| `tab5_literature.tex` | Table 6 |
| `tab6_dosimetry_and_generalization.tex` | Table 5 |

Use semantic identifiers throughout the build and derive displayed numbers from document order.

**2. The style I recommend**

Keep **EB Garamond for prose**, paired with **Source Sans 3 for figures, diagram labels, and table headings**. Retain Garamond Math for manuscript equations and IBM Plex Mono only for actual code identifiers.

The intended appearance is a restrained scientific atlas:

- White figure backgrounds.
- Near-black labels.
- Grayscale MRI with precise, sparse overlays.
- Thin structural rules and generous panel gutters.
- One prominent result per panel.
- Small conceptual drawings immediately beside the evidence they explain.
- Color used consistently, with shape, line style, or explicit labels carrying the same information.

The current warm-paper HTML treatment is attractive for prose. Keep it around the figures, but present scientific images on an unmodified white surface.

The relevant lessons from the requested authors are structural:

- In **Tomé et al., with Clopath, “Dynamic and selective engrams emerge with memory consolidation”**, Fig 1 starts with model and protocol schematics before quantitative results; later experimental figures connect the proposed mechanism to its test. NeedleFinder should similarly connect each failure mode to a mechanism and then to evidence. [Nature Neuroscience, 2024](https://www.nature.com/articles/s41593-023-01551-w)
- **Kapur et al., “3-T MR-guided brachytherapy for gynecologic malignancies”**, grounds the technical work in the operating suite, applicators, and sequence-dependent catheter appearance. NeedleFinder currently lacks that physical orientation. [Magnetic Resonance Imaging, 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3468320/)
- **Mastmeyer et al., with Kapur**, links a mechanical catheter model to image evidence and shows an illustrative comparison of image-only, model-only, and combined reconstruction. That is a particularly useful precedent for explaining why NeedleFinder combines physics and learning. [Medical Image Analysis, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5654713/)

These are relevant exemplars, rather than an objective ranking of their “best” papers. External image retrieval was incomplete; these comparisons rely on accessible article text and figure descriptions, not a font-level audit of every source figure.

**3. Typography: the figure text is substantially too small**

The checked-in [`nf.mplstyle`](/home/guillaume/github/needlefinder/vault/paper/style/nf.mplstyle:1) actually specifies a **7 pt base**, rather than 7.5 pt. More importantly, the generator overrides it repeatedly:

- Fig 1 subtitles: **5.6 pt**; runtime legend: **5.5 pt**.
- Fig 2 output descriptions: **5.3 pt**.
- Fig 4 baseline textbox: **4.9 pt**.
- Fig 6 legends: **5.0–5.1 pt**.
- Fig 8 annotations and legends: approximately **4.9–5.5 pt**.

Figures are authored at approximately 170 mm wide, while the PDF float block is 160 mm. Consequently, a 4.9 pt annotation becomes approximately **4.6 pt**. Mathematical subscripts are smaller still.

Increasing PNG resolution will not fix this. The figures need fewer words per panel and more physical space.

Use this final-size specification:

| Element | Target size |
|---|---:|
| Panel letters | 11 pt, semibold |
| Short panel titles | 10 pt, semibold |
| Axis labels | 9 pt |
| Tick labels | 8.5 pt |
| Legends and annotations | 8.5 pt |
| Ordinary figure text minimum | 8 pt |
| PDF body | 11–11.5 pt |
| PDF captions | 10 pt |
| Table body | 9 pt where feasible |
| Table notes | 8.5–9 pt |

A suitable starting style is:

```text
font.family: sans-serif
font.sans-serif: Source Sans 3, DejaVu Sans
font.size: 9
axes.labelsize: 9
axes.titlesize: 10
axes.titleweight: semibold
xtick.labelsize: 8.5
ytick.labelsize: 8.5
legend.fontsize: 8.5
axes.linewidth: 0.7
lines.linewidth: 1.5
lines.markersize: 5
grid.linewidth: 0.45
legend.frameon: False
pdf.fonttype: 42
ps.fonttype: 42
```

Bundle and register Source Sans 3 explicitly; do not depend on its installation on the build machine. Verify mathematical glyphs against the chosen sans family.

**Changing the stylesheet alone is insufficient:** replace the generator’s local font-size literals with shared typography tokens.

**4. Color accessibility: retain the palette, strengthen its implementation**

The existing palette is a reasonable starting point. Its principal weakness is using attractive plotting colors directly for tiny text.

Calculated contrast against white:

| Role | Current mark color | Contrast | Existing darker text color | Contrast |
|---|---|---:|---|---:|
| Stage 0 | `#E69F00` | 2.25:1 | `#8A5F00` | 5.65:1 |
| Stage 1 | `#0072B2` | 5.19:1 | `#005A8C` | 7.39:1 |
| Stage 2 | `#009E73` | 3.42:1 | `#00704F` | 6.12:1 |
| NeedleFinder | `#D55E00` | 3.87:1 | `#9E4400` | 6.40:1 |

The darker variants already exist in [`nf_palette.py`](/home/guillaume/github/needlefinder/vault/paper/style/nf_palette.py:2), but figure labels often use the lighter mark colors.

Use near-black for most text and the darker variants when colored text is necessary. As a screen-accessibility target, ordinary text should meet **4.5:1** contrast; the current tiny labels do not qualify for the large-text exception. [W3C contrast guidance](https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html)

I simulated the palette in linear RGB using the full-severity Machado matrices documented in the Colorspacious implementation. These are diagnostic simulations, not a guarantee of accessibility for every reader. [Simulation implementation](https://github.com/njsmith/colorspacious/blob/master/colorspacious/cvd.py)

The important findings are:

| Viewing condition | Vulnerable distinction | Remedy |
|---|---|---|
| Protanopia | Vermilion, brown, and the “lost” red lose much of their hue separation; several blues/purples converge. | Distinct markers, outlines, hatches, and direct labels. |
| Deuteranopia | Stage 1 blue and the Aleong/Kostoulas method colors become particularly similar. | Avoid assigning each baseline a new hue. Use labeled rows and neutral marks. |
| Tritanopia | Stage 1 blue and Stage 2 green move toward related cyan/teal appearances. | Encode stage number and shape as well as color. |
| Grayscale | Several stage and method colors have similar lightness; pale reference trajectories nearly disappear. | Dark reference lines, distinct stroke patterns, and explicit legends. |

Concrete encoding rules:

- **Stages:** retain amber / blue / green, always accompanied by `0`, `1`, `2A`, `2B`, or `2C`.
- **NeedleFinder variants:** use the same vermilion family with hollow versus filled markers and different line styles.
- **Baselines:** use neutral gray with distinct markers when they share an axis; directly labeled rows need no unique hue.
- **Human references:** dark gray or black, with clearly identified dashed or dotted lines.
- **Lost real catheters:** use a dark, outlined, hatched segment with a count. Red alone is insufficient.
- **MRI overlays:** apply a narrow contrasting halo to markers; their visibility must survive both bright and dark tissue.
- **Reference trajectories:** darken `gt3d` from `#BDBDBD`; its current contrast against white is only about **1.88:1**, before transparency.

For heatmaps, the current `YlOrBr` and `Blues` choices are preferable to a rainbow palette, but they lack scales and are composited over different effective MRI backgrounds. Use a common normalization and a compact colorbar. For standalone quantitative response maps, prefer a monotonic sequential map such as `cividis`. For overlays, keep sparse contours and explicitly document thresholds; avoid making hue the only indication of response strength.

**5. Fig 1 — Pipeline overview**

The current image has a useful left-to-right progression, but it starts with technical stages before explaining the clinical problem. It also spends considerable space duplicating Fig 6’s runtime breakdown.

Specific problems:

- MRI panels and schematic panels have different visible heights and title baselines.
- Stage names contain dense abbreviations before readers understand their purpose.
- The graph illustration mixes reference-derived geometry with illustrative rejected tracks without marking the panel as schematic.
- The output lacks an on-figure reference/prediction legend.
- The code places the endpoint marker outside the prediction loop, so only the final plotted trajectory receives that marker.
- Full CPU/GPU stage legends are too small.

**Redesign:** make this the clinical orientation figure, approximately 160 × 110 mm.

Use three regions:

1. A simple pelvic illustration showing template, obturator, target region, catheter shaft, and distal tip.
2. Three larger transformations: **find candidate centers → connect coherent tracks → locate and review tips**.
3. An authentic MRI/trajectory example with a small result strip.

Keep the timing strip to total reconstruction times: **4.15 s CPU / 2.67 s GPU**, explicitly labeled **headless production timing**. Move the full breakdown to Fig 6.

Make the anatomy schematic explanatory rather than a simulated patient-specific segmentation. Show the radioactive source path separately from the physical catheter tip.

Suggested caption opening:

> **NeedleFinder converts ambiguous catheter appearances on planning MRI into three-dimensional trajectories for subsequent review.**

Follow with panel descriptions, the example case, cohort timing, and the distinction between measured inference and modeled clinical review.

**6. Fig 2 — Architecture and physical mechanism**

This needs the most substantial redesign.

The visible text overflows are real: the implant-region gate explanation extends outside its box, and the triplet-message text crosses the panel boundary. More seriously, the diagram’s arrows imply that a Stage 2B candidate-derived gate feeds the Stage 1 decoder. That should be reconciled with the actual computational sequence.

Other issues:

- The displayed encoder omits the 144-channel scale described in the manuscript.
- Five boxes represent four task heads because the direction head is split into \(V_x\) and \(V_y\).
- Fragment stitching is absent.
- The tip localizer and uncertainty calibration are absent despite appearing in the caption.
- “Prunes 77% of vessel FPs” needs a clearly identified configuration, denominator, and source before being retained as a prominent callout.
- The figure uses \(p_t\), while the manuscript’s bundle confidence is \(p_m\).

**Redesign:** a four-panel methods plate, approximately 160 × 170 mm, with an adjacent detailed architecture supplement if necessary.

**Panel a: explain the physical observation and operator.** Show three matched drawings and MRI crops: a catheter cross-section, a flow-void vessel, and a one-sided tissue boundary. Overlay the six diameter directions and four physical radii. Illustrate the opposite-point minimum and robust aggregation.

Crucially, show that **both a catheter and a vessel can produce circular dark-core evidence**. Stage 0 suppresses one-sided boundaries; learned anatomy and bundle context address remaining vascular ambiguity.

The brief’s requested “asymmetric susceptibility void” and “half-ring operator” require qualification. The current manuscript describes a proton-density void with mild interface susceptibility contrast and a six-diameter robust ring operator. Half-ring terminology occurs in older project material, but is not specified in the current methods. Do not introduce it into the main diagram until its role in the evaluated implementation is established.

**Panel b: explain why peaks and directions help.** Show two touching catheter cross-sections:

- A merged occupancy mask.
- Two distinct centerline peaks.
- Local direction arrows.

Then place a compact network diagram underneath: six input channels, four scales, four task heads. Group \(V_x,V_y\) inside one direction-head container.

**Panel c: explain local versus bundle reasoning.** Show:

- Three consecutive detections with a smooth and a kinked continuation.
- A signal dropout with two fragments.
- The stitched candidate.
- A whole bundle with an anatomically implausible candidate marked for rejection.

Use explicit schematic labels. Do not present illustrative curves as measured ablation results.

**Panel d: explain the tip localizer.** Draw:

> Trajectory tangent → sampled axial profile → dilated ResNet → self-attention → stopping distribution and offset → calibrated prediction radius.

Preserve the actual architecture distinctions:

- **91 samples**, from **−25.0 to +20.0 mm**, spaced **0.5 mm**.
- **22 channels** in the base model.
- **24 channels** in the literature-fusion variant.
- Three residual blocks, kernel size 5, dilations 1/2/4.
- Four attention heads.
- A 91-bin distribution plus continuous offset.

Add a small inset separating **inner-validation calibration** from **outer-test prediction**. The evaluated uncertainty quantity is a **3D prediction radius**. A one-dimensional bar along the shaft can illustrate the interface concept, but must not silently redefine the calibrated quantity as a longitudinal confidence interval.

Suggested caption opening:

> **Local dark-core evidence proposes candidates; learned centerlines, trajectory continuity, bundle context, and axial tip profiles resolve progressively larger ambiguities.**

**7. Fig 3 — Shaft and tip error distributions**

The raincloud format is appropriate, and retaining both central tendency and tails is valuable. The present implementation needs clearer statistical semantics.

Verified issues:

- The shaft band represents the human **IQR**; its dashed line represents **P95 = 1.90 mm**.
- The tip dashed line represents the human **median = 3.24 mm**.
- The caption does not make that distinction explicit.
- The human comparison concerns long shafts in 40 dual-traced scans, while the method distributions concern their matched catheters from the full cohort.
- Tip distributions are cut off at 15 mm even though their P95 values extend much farther.
- The KDE is fitted after removing off-scale observations, changing the displayed density.
- Dots are subsampled to at most 220 per method; the caption currently calls them individual catheters without disclosing that sampling.

**Redesign:**

- Keep the two aligned shaft/tip columns.
- Use a neutral style for baseline rows and vermilion accents for the two principal NeedleFinder configurations.
- Move numerical summaries to an aligned text column outside the plotting area.
- Include a compact key: median line, IQR box, mean diamond, P5–P95 whisker.
- Label every human reference with its statistic and cohort.
- Add matched-catheter counts per method.
- Show the tip tail in a separate full-range ECDF or tail panel. Keep the central distribution view, but do not let it stand in for the complete distribution.
- Fit densities to the complete data, with appropriate treatment of the nonnegative boundary, then restrict only the view.
- State that displayed dots are a reproducible subsample.

The story should be balanced: NeedleFinder’s value is not that it wins every shaft-error statistic. Table 2 shows some baselines with lower matched-shaft errors. Detection, tip accuracy, and tail behavior must remain visible together.

Suggested caption opening:

> **NeedleFinder combines sub-millimeter matched-shaft errors with lower tip errors than the reproduced baselines, while substantial tip-error tails remain.**

**8. Fig 4 — Precision–recall tradeoffs and annotation efficiency**

The plot contains useful results but asks one panel to perform three jobs: compare methods, explain ablation progression, and show a threshold sweep.

Specific problems:

- `F1=0.80` is placed outside the intended axes area.
- Most baselines are relegated to a tiny textbox.
- The filled orange point is not clearly identified.
- The fusion star and base threshold curve are different configurations.
- `F1_adj` appears as a raw identifier.
- “F1 accuracy” is unnecessarily ambiguous.
- The annotation-budget axis uses logarithmic spacing without stating that explicitly.
- Only the all-catheter curve receives a confidence band.

**Redesign:**

- Show a full-range precision–recall overview containing every baseline.
- Add a small, clearly bounded zoom for the NeedleFinder threshold sweep.
- Label the base default operating point and fusion operating point separately.
- Place constant-F1 labels inside the visible plot, using one consistent percentage convention.
- Move stage-ablation detail into a neighboring compact progression panel if the comparison remains crowded.
- Replace `F1_adj` with typeset \(F_{1,\mathrm{adj}}\), followed by “obturator-adjusted” in the key.
- Label the horizontal axis “Annotated patients for Stage 2B fine-tuning per fold.”
- Explain that zero-shot here concerns the bundle-label budget; it does not mean the entire pipeline has never seen clinical training data.
- Identify exactly which curve has patient-bootstrap confidence intervals.
- Put endpoint labels in reserved space rather than across lines.

Suggested caption opening:

> **Bundle scoring improves the precision–recall tradeoff, and synthetic pre-training is most useful when clinical bundle annotations are scarce.**

Preserve the manuscript’s caution that the five-patient result is not established as significantly better than the fully trained 2019 baseline.

**9. Fig 5 — Qualitative MRI and 3D reconstruction**

This should be the paper’s strongest visual evidence. Currently it is an attractive gallery with too little guidance.

Specific problems:

- The white circles and vermilion crosses lack an on-figure key.
- Tiny dots do not clearly demonstrate touching-catheter separation.
- The columns repeat much of Fig 1.
- The reference curves are too pale.
- Projected bundles lack orientation and physical scale.
- The 3D projections do not explicitly enforce equal metric aspect.
- `P_obt` contours are called an “IRGM gate,” although obturator segmentation and the candidate-derived implant-region gate are different concepts.
- The differences between 23 predictions/26 references and 20 predictions/23 references are not explained.

**Redesign:**

Use two larger case rows, each with:

1. A raw MRI overview and marked crop.
2. A magnified local problem.
3. The corresponding reconstruction and an annotated 3D view.

For the dense case, highlight a real vessel-like distractor and a correctly reconstructed catheter. For the adhesion case, highlight an actual touching pair and show the two peaks separately.

Add a shared symbol key:

- Open circle: expert intersection.
- Cross: predicted intersection.
- Solid vermilion curve: prediction.
- Dashed dark-gray curve: reference.
- Distinct endpoint symbol: predicted tip.

Use matching crop extents and MRI windowing within each comparison. Add orientation labels derived from image metadata, a scale bar to each independent crop, and a cranial-direction marker on the bundle.

Identify a real miss or residual discrepancy with a leader line. **Do not infer “three misses” merely from the difference between prediction and reference counts**; use the actual one-to-one matching.

Suggested caption opening:

> **Two held-out examples show how local evidence becomes a coherent bundle, including the ambiguities and residual errors that remain.**

**10. Fig 6 — Specificity and runtime**

The ECDF is a good choice. Its legend currently covers part of the data, and the most prominent callouts emphasize the configuration exposed to these scans during unlabeled pre-training.

That exposure distinction should be visible in the figure itself:

- **Strictly unseen scans:** 0.33 false tracks per scan.
- **Scans included as unlabeled pre-training backgrounds:** 0.125 false tracks per scan.

Both results should remain, with equal clarity about their interpretation.

**Redesign panel a:**

- Place the legend below the axes.
- Use distinct dash patterns and occasional markers for all four curves.
- Label the x-axis transformation: linear near zero, logarithmic thereafter.
- Consider a linear 0–5 false-track inset, because that is where the practical story lies.
- Put the unseen-scan and background-exposed results in separate labeled callouts.
- Describe the plot as a distribution of false tracks per scan, with specificity defined explicitly.

**Redesign panel b:**

Use a **log-scale dot plot**, rather than bars starting from an implied zero on a logarithmic axis.

Separate:

- Same-study measured implementations.
- Published runtimes on other hardware.
- Assisted workflow timings.

Replace the unexplained superscript `r` with a clear key. Preserve “≥120 s” as a bound rather than presenting 120 s as an exact observation. Avoid implying that assisted reconstruction, headless inference, and different-hardware measurements are interchangeable.

**Redesign panel c:**

Keep the linear stage breakdown, enlarge it, and label segments by stage number. The tiny Stage 2B/2C segment needs an external callout. Add the standard/high-resolution split beneath the overall means.

Suggested caption opening:

> **Physical gating reduces false tracks on needle-free MRI, while measured headless reconstruction completes in a few seconds.**

**11. Fig 7 — Calibration, filtering, and review**

This is the natural bridge between automatic reconstruction and human inspection. Currently it stops at an aggregate audit.

Verified issues:

- Both reliability curves are solid; raw confidence is orange and calibrated confidence is green. The caption says otherwise.
- A count histogram shares the plot with a reliability curve through a second y-axis.
- The diagonal is unlabeled.
- Lost-catheter segments are extremely small and rely on color.
- The 1,015 reference-near kept candidates can be mistaken for the strict one-to-one true-positive count.

**Redesign:**

- Show raw confidence with a dashed line and open circles.
- Show calibrated confidence with a solid line and filled squares.
- Label the diagonal “perfect calibration.”
- Move counts into a short histogram below the reliability axes; identify whether they use raw or calibrated confidence bins.
- Retain ECE values **0.101 → 0.059**, without implying every individual bin improves.
- Use outlined, hatched lost-catheter segments with external counts.
- Explain that candidate-level reference proximity differs from strict one-to-one detection. Do not relabel 1,015 as the headline true-positive count.

Add a third panel showing the intended review sequence:

> Inspect uncertain tip → adjust endpoint → reconcile implant count → inspect reason-coded rejected candidate → restore when appropriate.

Use an authentic, de-identified Slicer screenshot if available. Otherwise label the interface drawing **proposed review interface**. Display a specific rejection reason beside the candidate and make restoration visually distinct from automatic acceptance.

Keep measured headless inference, simulated recovery, and unmeasured user-interface timing visibly separate.

Suggested caption opening:

> **Calibrated confidence supports interpretable filtering, while retained rejection reasons make selected candidates available for review.**

**12. Fig 8 — Split this into three figures**

This image currently contains enough material for three coherent figures. Enlarging its fonts within the existing layout will not solve its conceptual overload.

**Current panel a: dosimetry**

Problems:

- Protocol A gamma values appear beside both protocols’ dose-error bars without an explicit protocol label.
- The human row uses 40 patients while most other rows use 54.
- The headline manuscript result uses a median, while the bars show means.
- The panel does not visibly say that anatomy consists of standardized target/OAR envelopes.
- The role of oracle tip correction is too easy to miss.

Replace it with:

- A small schematic of catheter, dwell positions, target envelope, and proxy OAR shells.
- Separate Protocol A and Protocol B result panels.
- Separate 54-patient and matched 40-patient comparisons.
- Clearly labeled mean and median quantities.
- A paired before/after view for top-two-tip review, using existing patient data.
- A visible note: **simulated dosimetry; proxy anatomy; oracle correction**.

Retain both the benefit and its limits, including residual proxy OAR differences. Do not visually imply clinical dose equivalence.

**Current panel b: uncertainty-guided review**

Keep the review-budget plot, but:

- Put “simulated review” in its title.
- Separate P95 comparisons from the mean-error curve, or give the mean its own small panel.
- Label the 10 mm line as a reporting threshold, not a demonstrated clinical safety boundary.
- Distinguish the **review-order oracle** from the **correction oracle**: uncertainty-ranked review also assumes an idealized correction.
- Place the 90.2% conformal coverage in a separate nominal-versus-empirical coverage panel, identified as unreviewed \(k=0\).
- Keep modeled time separate from measured inference. Where time includes the 2-second queue-opening assumption, say so.

Preserve the difference between the dose experiment’s exact reference-depth correction and the geometry experiment’s retained longitudinal residual.

**Current panel c: generalization**

The omitted **post-contrast axial T1 VIBE** result must be restored prominently:

- **56.1% F1**
- **12 scans**
- **217 catheters**

Use the acquisition terminology verified in the source rather than silently adding “3D” from the brief.

Replace the mixed bar chart with grouped horizontal dot plots:

1. Prostate: automatic crop, manual crop, and exposure condition.
2. Secondary MRI sequences: wbSSFP, thick-slice T2, and T1 VIBE.
3. Serial MRI: independent 3D versus causal 4D.
4. CT correspondence: a separate recall panel.

Keep percentage axes within **0–100** and put annotations in an adjoining text column. The current axis reaches 154 only to make room for labels.

Show CT recall at both matching thresholds:

- **68.9% at ≤3.5 mm**
- **79.2% at ≤5.0 mm**

Do not place CT recall on the same undifferentiated ranking as MRI F1. Move “52% extra MRI detections confirmed on CT” into an adjudication inset with its own denominator.

Make the single-patient prostate result visibly a single-patient result.

**Current panel d: biomechanics**

The dual x-axis joins two unrelated categorical variables:

- Insertion-depth bins.
- Distance from a newly inserted needle.

Their shared horizontal positions imply correspondence that does not exist.

Replace it with two separate panels:

- **Deflection versus insertion-depth category.**
- **Lateral displacement versus proximity category.**

Pair each with a small physical schematic. Keep median, mean, and P90 visually distinct where shown. State **4 patients, 23 scans**, and identify the 212 persisted transitions for tissue-push analysis. Preserve the manuscript’s caution about treating clustered transitions as independent.

Suggested caption openings for the resulting three figures:

> **Idealized uncertainty-guided tip correction reduces simulated target-dose error on standardized implant-derived anatomy.**

> **Transfer performance varies across acquisition settings, with weaker results on post-contrast T1 MRI and pelvic CT.**

> **Retrospectively tracked catheter motion suggests depth-dependent bending and proximity-associated displacement during insertion.**

**13. Additional visual explanations worth adding**

Several important ideas currently require too much prose.

| Illustration | Placement | What it should explain |
|---|---|---|
| Catheter, template, obturator, shaft, tip, and dwell positions | Fig 1 | What is being reconstructed and why endpoint accuracy matters. |
| Catheter / vessel / tissue-boundary comparison | Fig 2a | What the symmetry prior can and cannot distinguish. |
| Anisotropic voxel stack and tilted catheter | Fig 2 or methods supplement | Why in-plane shaft localization differs from longitudinal tip localization. |
| Tip-error vector decomposition | Tip/localization or clinical figure | Along-track versus perpendicular error; preserve the “98.8% of squared error” definition. |
| Nested evaluation and calibration map | Methods supplement | Patient separation, calibration placement, and the special inner-OOF configuration. |
| Human-reference comparison and CT adjudication | Companion to Table 3 | Why disagreement is not identical to physical error. |

For the last item, a paired non-inferiority interval plot would be more informative than another bar chart. Show zero difference, the prespecified margin, and the existing confidence intervals. Keep the Team 1 reference explicit and preserve the weaker flipped-reference results.

The physical illustrations should be vector drawings with measured quantities labeled. Any conceptual profile or anatomy should say **schematic**; actual MRI, trajectories, and outcome plots should remain traceable to source data.

**14. Tables need restructuring, not scaling down**

All six tables use `\scriptsize`. The dense content, tight column spacing, and lengthy notes undermine the otherwise elegant typography.

| Table source | Recommended treatment |
|---|---|
| `tab1_cohorts.tex` | Separate patients, volumes, and catheters into explicit columns. Visually distinguish overlapping subsets from independent cohorts. |
| `tab2_main_results.tex` | Split benchmark cohorts and ablations into continued tables. Repeat headers and preserve every note. |
| `tab3_interobserver.tex` | Separate detection comparisons, intrinsic-error estimates, and non-inferiority tests. Do not reuse the same visual columns for unrelated quantities. |
| `tab4_specificity.tex` | Add a visible training-exposure grouping; the distinction currently depends on a footnote. |
| `tab5_literature.tex` | Preserve modality grouping and the warning that cohorts and metrics are not directly comparable. Avoid visual winner rankings across heterogeneous studies. |
| `tab6_dosimetry_and_generalization.tex` | Separate dosimetry, simulated review, conformal coverage, and transfer into continued parts with their own headers. |

Specific safeguards:

- Separate mean and median columns instead of relying on repeated slash notation.
- Put units in headers.
- Use tabular lining numerals and decimal alignment.
- Define whether bold indicates best performance, a selected configuration, or a primary endpoint.
- Recheck “best/second-best” formatting within genuinely comparable blocks.
- Do not emphasize a lower-bound-clamped intrinsic variance as evidence of perfect accuracy.
- Explain modeled review time with and without queue-opening overhead.
- Resolve the cohort table’s “0 catheters” external-pool summary against its inclusion of the prostate implant; do not silently rewrite the count during a layout change.
- Reconcile the secondary-sequence summary with the T1 VIBE cohort before producing a new cohort illustration.

For PDF, continued tables or a deliberately designed landscape page are preferable to `\resizebox`. For HTML, use readable type and horizontal scrolling with persistent row labels.

**15. Concrete changes by file**

**[`nf_palette.py`](/home/guillaume/github/needlefinder/vault/paper/style/nf_palette.py)**

Expand it into a genuine design specification:

- Separate `mark`, `text`, and `tint` roles.
- Add shared font-size, stroke-width, marker, dash, and hatch tokens.
- Define method identity separately from pipeline-stage identity.
- Include canonical configuration labels: base, synthetic pre-training, inner cross-fit, and literature fusion.
- Add dark reference-line and MRI-marker halo tokens.
- Update stale labels such as the `nnU-Net + clDice` label where the current evaluated method is identified differently.
- Make figure width depend on the intended output profile.

**[`nf.mplstyle`](/home/guillaume/github/needlefinder/vault/paper/style/nf.mplstyle)**

Apply the typography specification above. Use white backgrounds, outward ticks, modest line weights, and selective grids. Keep vector PDF output with embedded fonts.

Avoid using `bbox_inches="tight"` as a layout repair mechanism. It can change exported dimensions when text escapes the intended canvas. Prefer an explicitly sized canvas with validated internal bounds.

**[`generate_pmb_figures.py`](/home/guillaume/github/needlefinder/vault/paper/generate_pmb_figures.py)**

- Replace absolute `/mnt/SSD/...` paths with paths derived from the script location.
- Replace manual positioning where practical with `GridSpec`, subfigures, and dedicated legend regions.
- Remove local tiny-font overrides.
- Create reusable helpers for MRI panels, scale bars, legends, summary columns, and schematic labels.
- Centralize data-derived labels and rounding.
- Keep method variants and exposure conditions explicit.
- Correct the endpoint-marker loop placement.
- Preserve equal physical aspect in trajectory projections.
- Align CSV exports with the exact data objects used by their figures; the few-shot export and plotted curve currently select differently named source sections.
- Make missing data fail visibly rather than silently substituting a different configuration.
- Update stale comments describing six figures or an error-colored bundle that is actually uniformly vermilion.

Extend `assert_no_overlap` to check:

- Text containment inside diagram boxes.
- Text containment inside the figure canvas.
- Legends covering plotted data.
- Label/arrow collisions.
- Minimum ordinary text size after placement.
- Consistent physical dimensions across exported formats.

Text bounding-box overlap alone is not a publication-layout test.

**[`preamble.tex`](/home/guillaume/github/needlefinder/vault/paper/style/preamble.tex)**

Keep the current EB Garamond/Garamond Math pairing. The nominal 11 pt body and approximately 120 mm prose measure are reasonable; the urgent problem is figure and table typography.

Recommended adjustments:

- Trial 11.5 pt body with approximately 14–14.5 pt leading for the reader edition.
- Keep captions at 10 pt.
- Give dense captions approximately 135–140 mm of measure to reduce their vertical bulk.
- Use lining numerals consistently for quantitative statements, tables, and figure references.
- Make section hierarchy slightly more visible; currently several levels depend heavily on small caps or italics.
- Support multipage tables and prevent notes from extending outside page bounds.
- Revisit frequent `\FloatBarrier` placement after the table redesign to reduce awkward page breaks.

Treat the reader edition and journal submission layout as separate output profiles. Do not assume custom typography supersedes PMB’s eventual submission requirements.

**[`build_html.py`](/home/guillaume/github/needlefinder/vault/paper/build_html.py)**

The existing 19 px desktop body is broadly comfortable. Its subsidiary text is much smaller:

- Captions: approximately **16 px**.
- Tables: approximately **14.8 px**.
- Notes: approximately **14.1 px**.
- On mobile, the 17.5 px root reduces those further.

Recommended targets:

- Body: **20 px**, line-height **1.55**.
- Mobile body: **18 px**.
- Captions: **17–18 px**.
- Tables and notes: **at least 16 px**, with scrolling rather than forced compression.
- Preserve a prose measure of roughly 60–70 characters.

Other concrete changes:

- Remove `mix-blend-mode: multiply` from scientific figures.
- Use vector SVG for diagrams/charts, with raster MRI embedded where necessary.
- Provide accessible full-size figure viewing and downloads.
- On mobile, expose individual panels at readable width rather than shrinking a complete eight-panel image to the viewport.
- Replace caption-length image alt text with a concise purpose/trend description and an associated longer description.
- Add real table captions or explicit accessible associations, row/column header scopes, and keyboard-accessible scroll regions.
- Preserve figure numbering through the same semantic map used for cross-references.
- Allow large tables to break across printed pages; the current print `break-inside: avoid` treatment is unsuitable for tables exceeding one page.

**16. Caption storytelling**

Use the same sequence for every caption:

1. **What the figure establishes.**
2. What each panel shows.
3. Cohort, configuration, and denominator.
4. Meaning of symbols, bands, and lines.
5. Statistical or experimental qualifications.

Do not make readers search the methods to decode an essential mark. Conversely, avoid repeating the complete architecture in the overview caption.

A figure should remain understandable when separated from the paper. That requires visible distinctions between:

- Measured and simulated outcomes.
- Automatic and manually assisted settings.
- Unseen and unlabeled-background-exposed scans.
- Candidate proximity and one-to-one detection.
- Full-cohort and matched-subset comparisons.
- Base and literature-fusion configurations.
- Prediction radii, confidence intervals, and distribution percentiles.

**17. Recommended implementation order and acceptance criteria**

First repair page clipping, numbering ambiguity, caption mismatches, missing T1 results, and ambiguous reference lines. Then establish the typography and encoding system. Redesign Figs 1–2 around the clinical and physical explanation, improve the evidence figures, and split Fig 8 into dosimetry/review, generalization, and biomechanics.

Before accepting the redesigned set:

- Inspect the PDF at actual placement size and on an ordinary printed page.
- Require no ordinary figure label below 8 pt.
- Inspect every figure in grayscale and under all three CVD simulations.
- Confirm that shapes, dashes, labels, and borders preserve meaning without hue.
- Verify every legend, threshold, band, and off-scale indication.
- Check every table and note against page boundaries.
- Confirm identical numeric content and citations across PDF, HTML, tables, and exported data.
- Reconcile existing numerical or configuration discrepancies against archived results; never resolve them by aesthetic preference.
- Test HTML at desktop and phone widths, with enlarged text and keyboard navigation.

**The most valuable investment is in the physical-methods plate and the clinical-review story.** Together, they can let a reader understand what the catheter looks like, why each stage exists, where the method still fails, and what a clinician would inspect—before asking that reader to absorb the dense benchmark tables.