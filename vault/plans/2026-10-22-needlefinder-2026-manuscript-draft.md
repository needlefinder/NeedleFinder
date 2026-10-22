---
title: "2026-10-22 NeedleFinder 2026 Scientific Manuscript Draft & Visual Blueprint"
type: plan
category: publication
created: 2026-10-22
updated: 2026-10-22
sources:
  - 2026-10-22-full-54case-unet-transformer-graphnet-benchmark.md
  - 2026-10-22-literature-clinical-relevance-and-public-datasets.md
  - 2026-10-22-multi-agent-paper-review-synthesis.md
  - 2026-10-22-interobserver-543-benchmark.json
tags: [manuscript, paper-draft, pmb, medical-physics, unet3d, mpnn, graph-transformer, interobserver]
---

# From Voxels to Catheter Bundles: Physics-Guided Multi-Head 3D Networks and Graph Transformers for Sub-Second Catheter Reconstruction in MR-Guided Gynecologic Brachytherapy

**Authors:** Guillaume Pernelle et al.  
**Target Venues:** *Physics in Medicine & Biology (PMB)* (Direct sequel to Zaffino, Pernelle et al., *PMB* 2019, `PMC6726393`) or *Medical Physics (AAPM)*  
**Reviewed & Synthesized By:** Multi-Agent Research Council (`Codex Astra gpt-6-astra`, `Claude Opus 5.5`, `Claude Fable 5.1`, `AGY Argon gemini-4-argon-high`)

---

## Abstract

**Objective:** In Magnetic Resonance-guided adaptive interstitial gynecologic brachytherapy (IGABT), up to 35 plastic catheters and vaginal obturator needles are implanted into pelvic tumors. While transitioning from CT-MR fusion to MR-only treatment planning eliminates inter-scan registration errors, manual digitization of faint $1.5\text{–}2.5\text{ mm}$ susceptibility signal voids on intraoperative T2-weighted MRI requires $46 \pm 10\text{ minutes}$ of expert time while the patient remains under anesthesia. Existing 3D convolutional neural networks (CNNs) treat catheter localization as binary voxel segmentation on isotropically resampled ($1.0\text{ mm}^3$) grids, discarding sub-millimeter in-plane resolution and failing to separate touching, crossing, or boundary-adjacent catheter instances.

**Approach:** We formulate multi-catheter reconstruction not as voxel labeling, but as a **three-scale curve-set inference problem** across local voxel evidence, single-trajectory path continuity, and whole-bundle mutual exclusion. Completing a 13-year research arc on the Brigham and Women's Hospital AMIGO 3T MRI cohort ($54$ patients, $1,063$ total catheters: $759$ long transperineal interstitial catheters and $304$ short vaginal obturator needles, plus $543$ dual-observer annotations across $41$ cases), we introduce **NeedleFinder 2026**:
1. **Scale 1 (Local Native-Resolution Physics & Multi-Head 3D Evidence):** A multi-threaded Rust/C++17 3D radial and half-ring asymmetric dark-tube operator ($60\text{ ms}$) feeds an anisotropic 5-head 3D Residual U-Net (`NeedleUNet3D`) and hybrid Axial/Spatial Conv-Transformer (`AxialTransUNet3D`) operating at native in-plane resolution ($0.23\text{–}0.625\text{ mm}$), predicting full occupancy ($P_{\text{mask}}$), sub-voxel centerline peaks ($P_{\text{peak}}$), 3D tangent slopes ($v_x, v_y$), and short obturator needle probability ($P_{\text{obt}}$).
2. **Scale 2 (Path Continuity via Triplet Collinearity Message-Passing GraphNet):** Candidate 2D slice detections ($68,124$ nodes and $1.24\times 10^6$ directed inter-slice edges across 54 patients) are linked by a 4-layer Message-Passing Neural Network (`NeedleSliceMPNN`) that explicitly scores second-order **triplet collinearity** ($\|\Delta \mathbf{s}_{huv}\|$) across consecutive slices and regresses sub-voxel $(\Delta x, \Delta y)$ centroid refinements.
3. **Scale 3 (Bundle Reasoning via Edge-Biased Graph Attention & Set Transformer):** A lightweight (`257 KB` ONNX) Trajectory-Level Bundle Graph Transformer (`DenseGraphTransformer`) applies 2 layers of Edge-Biased Graph Attention (GAT) and 2 layers of global Set-Transformer self-attention over all candidate 3D trajectories to suppress duplicate/shadow tracks and calibrate 1-to-1 confidence scores.

**Main Results:** Evaluated via 5-fold patient-level cross-validation across all 54 patients ($1,063$ needles):
- On the **759 long interstitial catheters**, our pipeline achieves **`97.63%` closest recall** (`95.65%` strict 1-to-1 Hungarian recall), **`89.19%` F1**, and **`0.62 mm` mean (`0.54 mm` median) in-plane shaft error**.
- Across **all 1,063 catheters (long interstitial + short vaginal obturator)**, adding `NeedleSliceMPNN` + `DenseGraphTransformer` boosts 1-to-1 precision by **`+12.2 percentage points`** at the same **`95.20%` recall** (`93.70%` 1-to-1 recall, `83.38%` F1 at $\tau=25\%$), and reaches **`85.69%` F1 (`83.62%` precision, `90.22%` recall)** at $\tau=60\%$, while pure `NeedleSliceMPNN` achieves **`0.55 mm` mean (`0.43 mm` median) shaft error**.
- Across the **41 dual-observer cases ($543$ catheters annotated independently by `CY` and `LB`/`AMe`)**, our automatic reconstruction achieves **`0.55 ± 0.36 mm` mean (`0.45 mm` median, `1.19 mm` P95) shaft error** against reference observer `CY`—**22% lower mean error and 41% lower 95th-percentile error than human inter-observer disagreement (`0.71 ± 0.58 mm` mean, `0.51 mm` median, `2.02 mm` P95)**—and matches a second human expert zero-shot (`0.73 ± 0.57 mm` mean, `0.51 mm` median against `LB`/`AMe`).
- End-to-end execution requires **`0.61 s` per patient on a standard CPU** (`0.35 s` on GPU) inside 3D Slicer (`10.5 MB` total weights), a **`148×–885×` speedup** over prior 3D U-Net pipelines ($2\text{–}9\text{ min}$ on GPU).

---

## Figures Generated for the Manuscript (`vault/reports/figures/`)
1. **`reports/figures/fig1_pipeline_architecture.svg`**: End-to-end 3-scale architecture diagram (`Stage 0` Rust/C++17 physics filter $\to$ `Stage 1` 5-head `NeedleUNet3D`/`AxialTransUNet3D` $\to$ `Stage 2A` 4-layer Triplet Collinearity `NeedleSliceMPNN` $\to$ `Stage 2B` Bundle `DenseGraphTransformer`).
2. **`reports/figures/fig2_pareto_precision_recall_f1.svg`**: 5-Fold Out-of-Fold 1-to-1 Precision-Recall Pareto frontier across all `1,063` needles (`54` cases) and confidence threshold operating curves ($\tau \in [25\%, 60\%]$).
3. **`reports/figures/fig3_shaft_error_and_runtime.svg`**: Side-by-side comparison of Shaft Localization Error against Human Inter-Observer Variability (`CY` vs. `LB`/`AMe` across `41` scenes / `543` catheters) and Sub-Second Per-Patient Runtime Breakdown (`615 ms` CPU / `352 ms` GPU).

See full review synthesis in [[wiki/syntheses/2026-10-22-multi-agent-paper-review-synthesis]] and literature/dataset survey in [[wiki/research/2026-10-22-literature-clinical-relevance-and-public-datasets]].
