# Chronological Log (`OVNF`)

> Append-only chronological record of ingests, benchmarks, syntheses, and vault operations.
> Format: `## [YYYY-MM-DD] <action> | <title>`

---

## [2026-10-22] benchmark & paper | Claude Design System Overhaul, `arXiv:2607.11290` Integration (`IRGM + SyncTrack + Poly3`), Severe Adhesion & Few-Shot Benchmarks, and Final PMB PDF/HTML Build
- Analyzed **Xiao et al. (`arXiv:2607.11290`, July 2026, *"Multi-Catheter Digitization in Brachytherapy via Few-Shot Synthetic-to-Real Learning and Structure-Aware Tracking"*)** and implemented its core architectural strategies on our 54-patient (`1,063`-catheter) BWH 3T T2W MRI cohort (`sources/2026-10-22-unified-apples-to-apples-benchmark.json`):
  1. **Head-to-head 5-fold OOF comparison**: Evaluated `Baseline 1: PMB 2019 3D U-Net + DBSCAN` (`81.3%` F1, `3.35 mm` median tip error), `Baseline 2: Native 3D ResUNet + clDice + Skeleton` (`37.9%` F1, `23.9%` precision due to 4,015 fragmented segments), `Baseline 3: Xiao 2026 Sync-Track (IRGM + 3D Hough + PCA + Synchronous Hungarian Inward Tracking + Poly3)` (`53.7%` F1 on OOF masks; `89.7%` F1 and `0.11 mm` median shaft error on oracle GT masks), and **`NeedleFinder 2026 + IRGM + Poly3`** (**`89.8%` recall, `82.5%` precision, `86.0%` F1, `0.46 mm` median shaft error [`0.43 mm` on long shafts], and `2.82 mm` median tip error [`2.72 mm` on long shafts]**; $p = 0.0085$ vs PMB 2019, $p < 10^{-9}$ vs Xiao 2026 and clDice).
  2. **Severe Catheter-Adhesion Subgroup (`23/54` patients, `233` catheters within $\le 3.0\text{ mm}$ across $\ge 3$ slices)**: `NeedleFinder 2026 + IRGM + Poly3` achieves **`87.1%` F1 (`91.9%` recall, `82.8%` precision, `0.45 mm` median shaft error, `3.33 mm` median tip error)**, outperforming `PMB 2019` (`85.9%` F1, `3.68 mm` tip), `Xiao 2026` (`63.1%` F1, `7.25 mm` tip), and `ResUNet + clDice` (`47.8%` F1).
  3. **Few-Shot Clinical Annotation Budget (`k = 0, 1, 5, 10, 20, 43` patients/fold)**: Our zero-parameter Stage 0 radial dark-core symmetry prior + IRGM achieves **`74.0%` F1 zero-shot ($k=0$)**, **`78.1%` F1 at $k=1$**, **`84.0%` F1 at $k=5$ (recovering `97.7%` of full 43-patient accuracy without synthetic training data)**, and **`86.6%` F1 at $k=10$**.
- Designed and executed the **Claude Scientific Design System** ([[plans/2026-10-22-pmb-scientific-design-system]]):
  - Replaced all Pandoc `longtable` tables with 5 hand-crafted LaTeX `booktabs` + `threeparttable` floats (`paper/tables/tab1..5.tex`).
  - Generated 6 collision-checked vector PDF/PNG/SVG figures (`reports/figures/fig1..6.{pdf,png,svg}`) via `paper/generate_pmb_figures.py` with `assert_no_overlap(fig, pad_px=1)` and real 3T T2W MRI qualitative slices (`Case 017` and `Case 031`).
  - Compiled zero-warning `paper/needlefinder_2026_pmb.pdf`, `paper/needlefinder_2026_pmb.tex`, and self-contained `paper/needlefinder_2026_pmb.html`.

## [2026-10-22] synthesis | Multi-Agent Paper Review Council (`Codex Astra`, `Opus 5.5`, `Fable 5.1`, `AGY Argon`), 41-Case Inter-Observer Benchmark & Manuscript Draft
- Executed 5-fold cross-validation and production ONNX/NPZ export across all 54 BWH clinical cases (`1,063` total needles: `759` CY long + `304` HT obturator) for `NeedleUNet3D`, `AxialTransUNet3D`, `NeedleSliceMPNN` (4-layer Triplet Collinearity GraphNet), and `DenseGraphTransformer` (2-layer Edge-Biased GAT + 2-layer Set Transformer); integrated into 3D Slicer (`0.61 s` CPU / `0.35 s` GPU).
- Resolved all 41 dual-observer scenes (`543` `LB`/`AMe` catheters) in `543LB_AMe_scenes.txt` (`sources/2026-10-22-interobserver-543-benchmark.json`), proving that `NeedleFinder 2026` achieves **`0.55 ± 0.36 mm` mean (`0.45 mm` median, `1.19 mm` P95) shaft error** against reference observer `CY` (**22% lower mean and 41% lower P95 error than human inter-observer variability `0.71 ± 0.58 mm` mean / `0.51 mm` median / `2.02 mm` P95**) and **`0.73 ± 0.57 mm` mean (`0.51 mm` median)** zero-shot against independent observers `LB`/`AMe`.
- Convened multi-agent paper review & literature council (`Codex Astra gpt-6-astra`, `Claude Opus 5.5`, `Claude Fable 5.1`, `AGY Argon`), saving raw reviews to [[sources/2026-10-22-review-codex-astra]], [[sources/2026-10-22-review-opus-5-5]], and [[sources/2026-10-22-review-fable-5-1]].
- Corrected 2017 lineage citation to *Medical Image Analysis* 42:173–188 (`PMC5654713`) in [[wiki/entities/jcmc-2017-mastmeyer-pernelle]] and updated [[wiki/entities/pmb-2019-zaffino-pernelle-pmc6726393]].
- Generated publication SVG figures (`reports/figures/fig1_pipeline_architecture.svg`, `fig2_pareto_precision_recall_f1.svg`, `fig3_shaft_error_and_runtime.svg`) and compiled [[wiki/research/2026-10-22-full-54case-unet-transformer-graphnet-benchmark]], [[wiki/research/2026-10-22-literature-clinical-relevance-and-public-datasets]], [[wiki/syntheses/2026-10-22-multi-agent-paper-review-synthesis]], and [[plans/2026-10-22-needlefinder-2026-manuscript-draft]].

## [2026-10-21] ingest | Historical Datasets & Codebase Audit (2013–2019)
- Located and audited all historical datasets, spreadsheets, and deep learning training scripts on `mbp-2` (`/Users/guillaume/Dropbox/`):
  - **MICCAI 2013**: `/Users/guillaume/Dropbox/2013 MICCAI/Latex/miccai_submission.tex`, `/Users/guillaume/Dropbox/GYN Cases/MICCAI13scenes.txt` (10 clinical cases: `024, 028, 029, 030, 031, 033, 034, 037, 038, 040`, 101 needles; 1 gel-wax phantom with 12 needles; 5 calibration cases).
  - **MICCAI 2015 / MedIA 2017**: `/Users/guillaume/Dropbox/GYN Cases/760CYScenes.txt` (54 cases, 760 needles), `543LB_AMe_scenes.txt` (41 cases, 543 needles), `short needle segment data of 50 cases.xlsx`.
  - **RSNA 2016**: `/Users/guillaume/Dropbox/2016-RSNA` (`Case 050`, 20 long needles + 6 short obturator needles).
  - **PMB 2019 (PMC6726393)**: `/Users/guillaume/Dropbox/2016-paolo/` and `/Users/guillaume/Dropbox/Projects/unet/` (`train.txt`, `valid.txt`, `testing.txt`, `unet.py`, `3d-unet-tf-tiles-1class-multiple-gpus.py`).
- Created [[wiki/research/historical-dataset-splits-2013-2026]].

## [2026-10-21] benchmark | Auto-Needle Detection & Segmentation Quantitative Evaluation
- Executed end-to-end evaluation of `NeedleFinderLogic.autoDetectAllNeedles` across 4 multi-needle 3T T2W MRI datasets (67 ground-truth needles: `2016-RSNA`, `Case016`, `Case030`, `Case038`).
- Saved raw JSON metrics to `sources/2026-10-21-auto-needle-detection-benchmark.json`.
- Created [[wiki/research/2026-10-21-auto-needle-detection-evaluation]].

## [2026-10-21] synthesis | 2026 Paper Roadmap: Explainable Zero-Shot Physics-Informed 3D Detection vs. 3D U-Net
- Synthesized comparison between MICCAI 2013 (semi-automatic), PMB 2019 (3D U-Net + heuristic clustering), and 2026 `autoDetectAllNeedles` (zero-shot 3D radial symmetry + bidirectional Kalman tracking + Rust/C++17 acceleration).
- Created [[wiki/syntheses/2026-paper-roadmap-explainable-zero-shot-vs-3d-cnn]] and [[plans/2026-10-21-new-paper-proposal]].
