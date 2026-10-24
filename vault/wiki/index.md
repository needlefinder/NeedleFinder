# Wiki Index (`OVNF`)

> Master catalog of all wiki pages in the NeedleFinder Obsidian Vault (`OVNF`). Maintained automatically on every ingest, synthesis, or lint pass.

---

## Overview & Navigation
- [[home|Home]] — Entry point, quick links, and publication lineage table
- [[wiki/overview|Overview]] — High-level synthesis of the NeedleFinder project, clinical problem, and algorithmic evolution (2012–2026)
- [[wiki/log|Chronological Log]] — Append-only timeline of ingests, benchmarks, and syntheses

---

## Research & Evaluations (`wiki/research/`)
- [[wiki/research/2026-10-23-clean-nested-multirater-ssl-1dtip-benchmark|2026-10-23 Clean Nested 5x5 Benchmark: Multi-Rater Supervision, External Pelvic SSL, Reference-Free k_lim_auto & Dedicated 1D Axial Tip Localizer]] — Strictly nested $5 \times 5$ patient-level OOF benchmark (`25` inner models, 100% reference-free `k_lim_auto`, 51-volume external pelvic SSL with anatomical hard negatives, 40/40 multi-rater consensus supervision, collinear fragment stitching, and 91-step dilated 1D ResNet + self-attention tip localizer): `89.1%` F1 (`91.4%` recall, `92.3%` inner-OOF cross-fit), `0.59 ± 0.40 mm` (`0.47 mm` med) shaft error, `6.23 ± 10.49 mm` (`2.27 mm` med) 3D tip error, `0.125–0.33 FP/scan` on 48 MSD prostate volumes, and `4.15 s` CPU / `2.67 s` GPU full-volume runtime
- [[wiki/research/2026-10-22-full-54case-unet-transformer-graphnet-benchmark|2026-10-22 Full 54-Case Benchmark (Physics, 3D U-Net, Axial TransUNet, Slice MPNN & Bundle Graph Transformer)]] — 5-fold cross-validation across all 54 BWH clinical cases (`1,063` needles: `759` CY long + `304` HT obturator) and runtime breakdown (`0.61 s` CPU / `0.35 s` GPU)
- [[wiki/research/2026-10-22-literature-clinical-relevance-and-public-datasets|2026-10-22 Literature Survey, Clinical Relevance & Public Datasets (2013–2026)]] — EMBRACE-I/II clinical drivers, MR-only workflow bottleneck (`46 ± 10 min` manual), 2013–2026 SOTA comparison table, and public dataset audit (TCIA, Zenodo, RSNA `Case 050`)
- [[wiki/research/2026-10-21-sota-nn-hybrid-needle-detection|2026-10-21 SOTA Deep Learning & Hybrid 3D CV Survey]] — Architectural survey and root-cause analysis of missed needles
- [[wiki/research/2026-10-21-auto-needle-detection-evaluation|2026-10-21 Auto-Needle Detection Evaluation]] — Initial 4-dataset quantitative evaluation of `NeedleFinderLogic.autoDetectAllNeedles`
- [[wiki/research/historical-dataset-splits-2013-2026|Historical Dataset Splits (2013–2026)]] — Provenance and case-level mapping across MICCAI 2013, MedIA 2017, RSNA 2016, PMB 2019 (`PMC6726393`), and 2026 benchmarks

---

## Syntheses & Paper Roadmaps (`wiki/syntheses/`)
- [[wiki/syntheses/2026-10-22-multi-agent-paper-review-synthesis|2026-10-22 Multi-Agent Paper Review Synthesis (Codex Astra, Claude Opus 5.5, Claude Fable 5.1, AGY Argon)]] — Consolidated paper architecture, 3-scale curve-set formulation, 41-scene (`543`-needle) human inter-observer variability analysis, and 6 reviewer Q&A defenses
- [[wiki/syntheses/2026-paper-roadmap-explainable-zero-shot-vs-3d-cnn|2026 Paper Roadmap: Explainable Zero-Shot Physics-Informed 3D Detection vs. 3D U-Net]] — Initial comparative thesis and venue analysis

---

## Concepts (`wiki/concepts/`)
- [[wiki/concepts/3d-directional-radial-symmetry-filter|3D Directional Radial Symmetry Filter]] — Anisotropic 6-diameter + half-ring asymmetric susceptibility void detector with boundary clamp
- [[wiki/concepts/bidirectional-seed-tracking-and-kalman-smoothing|Bidirectional Seed Tracking & Kalman Smoothing]] — Forward + backward 2D cross-sectional valley tracking with constant-velocity smoothing
- [[wiki/concepts/anatomical-z-zone-priors-and-spatial-deduplication|Anatomical Z-Zone Priors & Spatial Deduplication]] — Pelvic insertion depth priors, obturator vs. interstitial length filtering, and greedy confidence-ranked 3D trajectory NMS
- [[wiki/concepts/short-obturator-vs-long-interstitial-needles|Short Obturator vs. Long Interstitial Needles]] — Geometric and clinical distinction between short vaginal obturator needles (~70–90 mm) and long transperineal interstitial needles (240 mm)

---

## Entities (`wiki/entities/`)
- [[wiki/entities/miccai-2013-pernelle|MICCAI 2013 (Pernelle et al.)]] — *Validation of Catheter Segmentation for MR-Guided Gynecologic Cancer Brachytherapy* (LNCS 8151, pp. 380–387)
- [[wiki/entities/jcmc-2017-mastmeyer-pernelle|Medical Image Analysis 2017 (Mastmeyer, Pernelle et al., PMC5654713)]] — *Accurate model-based segmentation of gynecologic brachytherapy catheter collections in MRI-images* (*Med. Image Anal.* 42:173–188, 2017)
- [[wiki/entities/pmb-2019-zaffino-pernelle-pmc6726393|PMB 2019 (Zaffino, Pernelle et al., PMC6726393)]] — *Fully automatic catheter segmentation in MRI with 3D convolutional neural networks: application to MRI-guided gynecologic brachytherapy* (*Phys. Med. Biol.* 64(16):165008)
- [[wiki/entities/rsna-2016-dataset|RSNA 2016 Dataset (Case 050)]] — Standard 3D Slicer tutorial & calibration case (`Case 050`, 20 interstitial needles + 6 short obturator needles)
- [[wiki/entities/bwh-54-case-gyn-cohort|Brigham & Women's Hospital 54-Case GYN Cohort]] — Complete AMIGO suite 3T T2W MRI dataset (`Cases 002–078`, `1,063` total needles = `759` CY long + `304` HT obturator, plus `543` dual-observer `LB`/`AMe` needles)

---

## Plans & Manuscript Drafts (`plans/` & `paper/`)
- [[paper/manuscript|2026-10-23 NeedleFinder 2026 PMB Manuscript (`needlefinder_2026_pmb.pdf` & `.html`)]] — Publication-ready *Physics in Medicine & Biology* manuscript synchronized with the strictly nested $5 \times 5$ benchmark, Codex Astra v3 audit (`sources/2026-10-23-review-codex-astra-v3-clean.md`), and Experiments 1–4 (TG-43U1 dosimetry, split-conformal intervals & Grubbs/TOST, prostate & multi-sequence transfer, 4D insertion & paired CT), with 6 `booktabs` + `threeparttable` tables (`paper/tables/tab1..6.tex`) and 8 vector PDF/PNG/SVG figures (`reports/figures/fig1..8.pdf`)
- [[plans/2026-10-22-pmb-scientific-design-system|2026-10-22 PMB Scientific Design System & Visual Architecture Spec]] — Okabe–Ito entity palette, booktabs/siunitx table macros (no longtables), 6-figure blueprint with anti-collision rules, and data blockers D1–D10 (adhesion subgroup, nnU-Net+clDice, Xiao modality, few-shot)
- [[plans/2026-10-22-needlefinder-2026-manuscript-draft|2026-10-22 NeedleFinder 2026 Scientific Manuscript Draft & Visual Blueprint]] — Initial paper draft
- [[plans/2026-10-21-new-paper-proposal|2026-10-21 New Paper Proposal]] — Initial execution plan

---

## Raw Sources & Reviews (`sources/` & `reports/`)
- `sources/2026-10-24-peer-reviewed-baselines-and-fusion.json` — Strictly nested $5 \times 5$ reproduction of 3 peer-reviewed MRI brachytherapy baselines (`Dai et al. 2020` [*Medical Physics*]: `93.6%` R, `42.2%` P, `58.2%` F1; `Kostoulas et al. 2025` [*Proc. SPIE*]: `89.5%` R, `53.0%` P, `66.5%` F1; `Aleong et al. 2024 / Kim et al. 2026` [*Medical Physics* / *Brachytherapy*]: `89.2%` R, `56.9%` P, `69.5%` F1) and literature-inspired upgrades inside `NeedleFinder 2026` (`4d: NeedleFinder 2026 + Literature Fusion`: `91.4%` R, `87.4%` P, **`89.4%` F1**, `0.58 mm` mean / `0.47 mm` med shaft error, `6.23 mm` mean / `2.26 mm` med 3D tip error)
- [[sources/2026-10-24-persona-reviews-kapur-clopath-and-dispositions|2026-10-24 Persona Reviews (Tina Kapur, Claudia Clopath) & Dispositions]] — Two fresh-context persona reviews of the PMB V2 manuscript, every numeric finding re-verified against the source JSONs, with applied / left-for-authors dispositions and an 8-item open list (manual tracing time source, Zaffino-2019 ranking, Slicer ONNX timing, baseline code provenance, tandem handling, pair counts, CT adjudication definition, title).
- [[sources/2026-10-23-pmb-v2-peer-review-and-response|2026-10-23 PMB V2 Readability Transformation (Claudia Clopath & Tina Kapur Style) & Three-Reviewer Peer-Review Record]] — Complete record of the V2 readability transformation and point-by-point responses to PMB Reviewers 1 (Clinical Brachytherapy & MRI Physics), 2 (Claudia Clopath Style & ML Pedagogy), and 3 (Biostatistics & Zero-Drift Audit)
- [[sources/2026-10-23-central-claim-verification-ledger|2026-10-23 Central Claim Verification & Reproducibility Ledger (`V-01`–`V-10`, 5-Agent Audit Squad)]] — Complete verification and resolution ledger across Verifiers 1–3 (`flash` subagents), Claude Haiku (`sources/2026-10-23-audit-haiku.md`), and Codex Luna (`sources/2026-10-23-audit-luna.md`), confirming 100% digit-for-digit agreement between code, JSON artifacts, tables, figures, and manuscript prose
- [[sources/2026-10-23-audit-haiku|2026-10-23 Independent Full-Manuscript Audit by Claude Haiku (`claude-haiku`)]] — Internal consistency, statistical unit-of-analysis, and reader-clarity audit
- [[sources/2026-10-23-audit-luna|2026-10-23 Independent JSON-to-Paper Verification Audit by Codex Luna (`gpt-6-luna`)]] — Independent cross-check of all JSON artifacts against manuscript claims and tables
- `sources/2026-10-23-unified-clean-nested-benchmark.json` — Strictly nested $5 \times 5$ OOF benchmark across all 54 patients (`1,063` catheters), 759 long active-dwell shafts, 23-patient Severe Catheter-Adhesion Subgroup (`571` catheters, including paired Wilcoxon $p$-values), and leak-free Stage 2B Few-Shot (`0, 0_synth_zeroshot, 1, 5, 10, 20, 43` shots)
- `sources/2026-10-23-table3-observers-and-nf-clean.json` — Complete 40/40 dual-observer benchmark (`CY+HT` 723 vs. `LB/AMe` 513 vs. union 759 vs. `NeedleFinder 2026`)
- `sources/2026-10-23-public-pelvic-mri-specificity-clean.json` — Clean zero-needle specificity benchmark on 48 public pelvic MRI scans (`MSD Task05_Prostate`: `0.125 FP/scan` with SSL + Stage 0 gate, `0.33 FP/scan` frozen unified NF 2026 + dark-core gate)
- `sources/2026-10-23-failure-analysis.json` — Categorized failure audit of all false positives, false negatives, and $>10\text{ mm}$ tip outliers across the 54 cases
- `sources/2026-10-23-exp1-tg43-dosimetry.json` — AAPM TG-43U1 HDR $^{192}\text{Ir}$ dosimetric sensitivity (Protocol A & Protocol B) and Monte Carlo uncertainty-to-dose propagation across all 54 patients and 40 dual-traced scans
- `sources/2026-10-23-exp2-uncertainty-conformal-grubbs.json` — Longitudinal vs. lateral tip decomposition (`98.76%` along-shaft), strictly nested split-conformal 3D tip intervals (`90.2%` / `95.4%` empirical coverage), clinician review-budget curves ($k=0..10$), and Grubbs' three-cornered-hat / TOST non-inferiority (`n=432`)
- `sources/2026-10-23-exp3-prostate-and-cross-sequence.json` — Zero-shot transfer to 3T prostate brachytherapy MRI (`Case 100 Prostate`, `96.3%–100%` F1 vs. `0%` with Stage 0 physics zeroed) and 297 multi-sequence MRI volumes (`3D wbSSFP` `92.9%` F1, `2D AX TSE T2` `88.4%` recall)
- `sources/2026-10-23-exp4-4d-insertion-and-ct.json` — 4D causal Kalman serial insertion tracking (`Cases 68–71`, `+5.6%–8.2%` F1), in-vivo tissue-push & depth-deflection biomechanics, and paired post-implant Pelvic CT (`40` volumes, `79.2%` recall, `52.0%` confirmation of extra MRI tracks, `4.20 mm` median error on 70 disputed tips)
- [[sources/2026-10-23-review-codex-astra-v3-clean|2026-10-23 Clean Nested Benchmark Audit (v3) by Codex Astra (`gpt-6-astra`)]] — Independent verification of all 5 methodological upgrades (`Items 1–5`), confirming 0 blocking evaluation defects
- `sources/2026-10-22-unified-apples-to-apples-benchmark.json` — Complete 5-fold OOF benchmark across all 54 patients (`1,063` catheters), 23-patient Severe Catheter-Adhesion Subgroup (`233` catheters), and Few-Shot (`0, 1, 5, 10, 20, 43` shots) comparing `PMB 2019 3D U-Net + DBSCAN`, `Native 3D ResUNet + clDice + Skeleton`, `Xiao 2026 Sync-Track (arXiv:2607.11290)`, and `NeedleFinder 2026 + IRGM + Poly3`
- `sources/2026-10-22-public-pelvic-mri-specificity-benchmark.json` — Out-of-distribution zero-needle specificity benchmark on 48 clinical 3T pelvic T2W MRI volumes (`MSD Task05_Prostate`)
- [[sources/2013-miccai-summary|2013 MICCAI Paper Summary]]
- [[sources/2019-pmb-pmc6726393-summary|2019 PMB (PMC6726393) Paper & Codebase Summary]]
- [[sources/2026-10-22-review-codex-astra|2026-10-22 Paper Review by Codex Astra (`gpt-6-astra`)]]
- [[sources/2026-10-22-review-opus-5-5|2026-10-22 Paper Review by Claude Opus 5.5 (`claude-opus-5-5`)]]
- [[sources/2026-10-22-review-fable-5-1|2026-10-22 Paper Review by Claude Fable 5.1 (`claude-fable-5-1`)]]
- `sources/2026-10-22-interobserver-543-benchmark.json` — Complete 41-scene (`543`-needle) `CY` vs. `LB`/`AMe` vs. `NeedleFinder 2026` inter-observer benchmark JSON
- `sources/2026-10-21-54cases-benchmark.json` — Raw per-case 54-case benchmark JSON
- `sources/2026-10-21-auto-needle-detection-benchmark.json` — Raw 4-case benchmark JSON
- `reports/figures/fig1_pipeline_architecture.svg` — Publication architecture diagram
- `reports/figures/fig2_pareto_precision_recall_f1.svg` — 5-Fold OOF Precision-Recall Pareto & confidence sweep figure
- `reports/figures/fig3_shaft_error_and_runtime.svg` — Shaft error vs. inter-observer floor & sub-second runtime breakdown figure

---

## Library (`wiki/library/`) — verified reference pages (`type: library`)

- [[wiki/library/index|Reference Library index]] — 88 one-page records (57 cited in `paper/references.bib` before the audit, 31 added by the 2026-10-22 systematic search), each built from the Crossref/PubMed/arXiv record and retrieved abstract or full text, with a claim-by-claim check of how `paper/manuscript.md` cites it.
