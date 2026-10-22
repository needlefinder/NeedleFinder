---
title: "Plan: Full 54-Case Benchmark & New Paper Execution"
created: 2026-10-21
status: proposed
tags:
  - plan
  - paper
  - benchmark
---

# Plan: Full 54-Case Benchmark & New Paper Execution

## Objective
Extend the 4-dataset quantitative evaluation ([[wiki/research/2026-10-21-auto-needle-detection-evaluation]]) to the entire **54-patient Brigham & Women's Hospital cohort (760–826 catheters)** stored on `mbp-2` (`/Users/guillaume/Dropbox/GYN Cases/`), and prepare a manuscript comparing zero-shot physics-informed 3D detection (`autoDetectAllNeedles`) against the 2019 3D U-Net ([[wiki/entities/pmb-2019-zaffino-pernelle-pmc6726393|PMC6726393]]) and inter-observer human variability (`CY` vs. `LB` vs. `AMe`).

---

## Phase 1: Full Cohort Evaluation on `mbp-2`
- [ ] **Step 1.1**: Write a batch evaluation script (`run_54case_benchmark.py`) that iterates over `/Users/guillaume/Dropbox/GYN Cases/760CYScenes.txt` (54 scenes, 760 long catheters) and `/Users/guillaume/Dropbox/GYN Cases/543LB_AMe_scenes.txt` (41 scenes, 543 catheters with dual-observer annotations).
- [ ] **Step 1.2**: Cross-reference `/Users/guillaume/Dropbox/GYN Cases/short needle segment data of 50 cases.xlsx` to evaluate both **long interstitial catheters** and **short vaginal obturator catheters** separately and jointly.
- [ ] **Step 1.3**: Compute stratified metrics on:
  - Calibration set (`Case 050` / `2016-RSNA`),
  - MICCAI 2013 10-case test set (`Cases 024, 028, 029, 030, 031, 033, 034, 037, 038, 040`),
  - PMB 2019 training/validation/testing splits (`train.txt`, `valid.txt`, `testing.txt`),
  - Full 54-case unseen generalization cohort.

## Phase 2: Algorithmic Refinements for 98%+ Recall
- [ ] **Step 2.1 — Two-Tier Obturator vs. Interstitial Classifier**: Separate candidate trajectories into long interstitial needles and short central obturator needles so users/evaluators can toggle short obturator detection via a checkbox (`Include Short Obturator Needles`).
- [ ] **Step 2.2 — Adaptive Axial Continuity Recovery**: In `Case016`, `Case030`, and `Case038`, exactly 1 needle per case had a mid-shaft partial-volume signal drop where the trajectory split or stopped early. Add a post-NMS **collinear trajectory stitching pass** (merging two fragments if their 3D tangent vectors align within $<8^\circ$, their axial gap is $<12\text{ mm}$, and the extrapolated line passes within $<2.0\text{ mm}$).
- [ ] **Step 2.3 — Sub-Slice Axial Tip Refinement**: Fit a 1D sigmoid/parabolic intensity profile along the needle tangent axis across the distal $\pm 2$ axial slices to refine the 3D tip Z-coordinate to sub-slice precision (reducing 3D tip error from $\sim 1.6\text{–}3.8\text{ mm}$ toward $<1.5\text{ mm}$).

## Phase 3: Manuscript & Open-Source Release
- [ ] **Step 3.1**: Generate publication figures (FROC curves, Bland-Altman plots of shaft error vs. inter-observer variability, 3D rendering of detected vs. manual needles in 3D Slicer, and CPU runtime scaling bar charts).
- [ ] **Step 3.2**: Draft manuscript in `vault/plans/` or LaTeX targeting *Physics in Medicine & Biology* or *Medical Physics*.
