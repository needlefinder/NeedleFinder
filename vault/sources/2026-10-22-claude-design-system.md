I couldn't save the spec: writing to `vault/plans/2026-10-22-pmb-scientific-design-system.md` needs your permission. The full document is written. If you approve the write, I'll save it and add it to `wiki/index.md` and `wiki/log.md`, or I can paste the whole markdown here instead. I also couldn't open the Xiao arXiv page (fetch not permitted), so nothing in the spec relies on its content.

**What I found before designing.** Several results the figure brief asks for don't exist yet in the vault or manuscript. The spec shows each as a red `\TBD` and adds a build check that fails if `TBD` is left in the PDF, rather than filling in estimates:
- **No adhesion subgroup has been defined.** I propose: catheters whose centerlines come within 3.0 mm of each other on at least 3 consecutive slices.
- **nnU-Net+clDice has never been trained.** PMB 2019 only has published summary numbers from a different 50-patient split.
- **Xiao 2026 is listed as a CT method in our draft.** If that's right, it can't be compared head-to-head on our T2W MRI without reimplementing it. Otherwise it appears only as a hollow "reported" marker.
- **The few-shot curve (1, 5, 10, 20, 43 cases) hasn't been run.**
- **Some whole-cohort numbers are missing:** per-catheter E_tip, D_avg and the P95 shaft error (whole-cohort P95 has never been computed).
- **Not in the manuscript:** the "90 s" runtime for PMB 2019, and the name "IRGM" (the paper says "5-head ResUNet + SE" and a "Stage 0 gate").
- **Two numbers disagree:** model size is 10.5 MB in the abstract but 14.8 MB in the runtime table, and the human comparison is 41 scenes / 543 catheters in the abstract but 40 scenes / 481 pairs in the table.

**Main design decisions:**
- **Colors:**
  - Stage 0 orange `#E69F00`, Stage 1 blue `#0072B2`, Stage 2 green `#009E73`.
  - NeedleFinder 2026 vermillion `#D55E00`, with table rows shaded `#FBEFE6`.
  - The three baselines are greys (`#4D4D4D`, `#7F7F7F`, `#ABABAB`) told apart by marker shape. The human floor is a black dashed line with a grey band.
  - Each color means the same thing throughout the paper, and yellow is not used.
- **Tables:** The first draft's problems came from Pandoc turning Markdown tables into page-splitting `longtable`s. Tables become hand-written LaTeX files pulled in with `\input`, built with `booktabs`, `threeparttable` and `siunitx`:
  - Method names stay on one line (28 characters at most) and each cell holds one value.
  - Percentages get one decimal and distances two; best is bold, second-best underlined.
  - Shrinking tables with `\resizebox` is banned.
  - There's a complete example of Table 2 (Part A end-to-end, Whole Cohort then Adhesion; Part B ablation) using the real numbers at the default threshold τ=50%.
- **Figures:**
  - Fixed sizes of 170 mm full width and 84 mm single column, 6 pt minimum text, a shared matplotlib style file, and PDF output directly. The SVG-to-PDF conversion is dropped because it swaps fonts.
  - Each label has a fixed position, and a small `assert_no_overlap(fig)` check fails the build if any labels collide.
  - Confidence intervals resample patients, since catheters within a patient aren't independent.
- **The six figures:**
  - **Fig 1:** five image tiles from a real case, with a runtime bar underneath split by stage. At most 30 words.
  - **Fig 2:** network and graph diagrams in TikZ with equation callouts.
  - **Fig 3:** horizontal 2×3 rainclouds whose whiskers run to P5 and P95.
  - **Fig 4:** precision–recall plot with F1 contours and the threshold sweep; the few-shot panel is blocked until that experiment runs.
  - **Fig 5:** 2×4 qualitative panels with zoomed insets and 3D renders colored by error.
  - **Fig 6:** cumulative false-positive curve across the 48 volumes, runtime on a log axis, and a stage breakdown on a linear axis (stacked bars can't go on a log axis).

No HTML or PDF was produced, so there is no rsync line.
