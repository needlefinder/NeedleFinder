---
title: "2026-10-22 Full 54-Case Benchmark: Physics 3D CV, 3D U-Net, Axial TransUNet, Slice MPNN & Bundle Graph Transformer"
type: research
category: evaluation
created: 2026-10-22
updated: 2026-10-22
sources:
  - 2026-10-21-54cases-benchmark.json
  - 2019-pmb-pmc6726393-summary.md
  - 2013-miccai-summary.md
tags: [evaluation, 54-cases, unet3d, transformer, graphnet, mpnn, gatv2, benchmark, sota]
---

# 2026-10-22 Full 54-Case Benchmark: Physics 3D CV, 3D U-Net, Axial TransUNet, Slice MPNN & Bundle Graph Transformer

## 1. Executive Summary

Across all **54 clinical intraoperative 3T T2-weighted MRI cases** from the Brigham & Women's Hospital AMIGO suite (**`1,063` total ground-truth needles**: **`759` long interstitial catheters** annotated by observer `CY` + **`304` short vaginal obturator needles** annotated by observer `HT`), we systematically designed, trained (5-fold cross-validation on `edelweiss`: RTX 4090 + RTX 4070 Ti SUPER), and benchmarked five distinct algorithmic families:

1. **Multi-Pass Physics-Informed 3D CV + `HybridNeedleNet` (`Pass 1–3` + 14-Feature Residual MLP + Joint Eigenvalue Orientation Field)**:
   - Operates at native sub-millimeter in-plane resolution ($0.23\text{–}0.625\text{ mm}$) with Rust/C++17 multi-threaded 6-diameter + half-ring asymmetric boundary filter, iterative tract inpainting, and Pass-3 sheared 3D Radon line-integral recovery.
2. **Stage 1 Multi-Head Anisotropic 3D Residual U-Net (`NeedleUNet3D`, 2.28M params, 8.95 MB ONNX)**:
   - Takes 4 physics-informed channels ($I_{\text{norm}}, \log(1+R_{\text{sym}}), C_{\text{smooth}}, R_{\text{asym}}$) and jointly predicts 5 voxel heads: full needle labelmap ($P_{\text{mask}}$), Gaussian centerline heatmap ($P_{\text{peak}}$), 3D unit tangent orientation field ($v_x = dx/dz, v_y = dy/dz$), and short obturator needle mask ($P_{\text{obt}}$).
3. **Stage 1 Hybrid 3D Conv-Transformer (`AxialTransUNet3D`)**:
   - Replaces the deepest encoder stages of `NeedleUNet3D` with **Axial Self-Attention along $Z$** (bridging multi-slice susceptibility dropouts across all slices simultaneously) and **2D Spatial Multi-Head Self-Attention in $XY$** (capturing global Syed-Neblett template array geometry).
4. **Stage 2 2D-to-3D Spatio-Temporal Message-Passing Graph Network (`NeedleSliceMPNN`, 4-Layer Triplet Collinearity GraphNet)**:
   - Formulates needle reconstruction as link prediction and node classification on a directed 3D graph of $68,124$ candidate 2D slice detections and $1,243,324$ inter-slice candidate edges across the 54 cases, explicitly scoring **triplet collinearity** ($\|(s_{h\to u} - s_{u\to v})\|$) between incoming and outgoing edges at every node.
5. **Stage 2 Trajectory-Level Bundle Graph Attention + Set Transformer (`DenseGraphTransformer`, 257 KB ONNX)**:
   - Models the entire candidate needle bundle ($M \approx 15\text{–}45$ trajectories/case) as a dense attributed graph with 2 layers of Edge-Biased Graph Attention (GAT) + 2 layers of global Set-Transformer Self-Attention, jointly trained on high-recall matching ($y_{\text{rec}}$) and strict 1-to-1 Hungarian matching ($y_{\text{1to1}}, y_{\text{long}}$).

---

## 2. Stage 1 Voxel Segmentation Benchmark (5-Fold Cross-Validation, 54 Cases)

| Stage 1 Architecture | Voxel Val Dice | Core Centerline Recall | 1-to-1 Hungarian Recall (`1,063` Needles) | Closest Recall (`1,063` Needles) | Inference Time (CPU / GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`NeedleUNet3D` (Anisotropic 3D ResUNet + SE)** | `0.5792` | `82.93%` | `992 / 1,063` (`93.32%`) | `1,012 / 1,063` (`95.20%`) | `0.21 s` / `0.03 s` |
| **`AxialTransUNet3D` (Hybrid 3D Conv + Axial/XY Transformer)** | `0.5543` | `82.89%` | `997 / 1,063` (`93.79%`) | `1,008 / 1,063` (`94.83%`) | `0.26 s` / `0.04 s` |
| **Ensemble (`NeedleUNet3D` + `AxialTransUNet3D`)** | **`0.5924`** | **`84.05%`** | **`1,001 / 1,063` (`94.17%`)** | **`1,014 / 1,063` (`95.39%`)** | `0.47 s` / `0.07 s` |

> [!NOTE] Comparison to PMB 2019 (Zaffino, Pernelle et al., PMC6726393)
> Both `NeedleUNet3D` (`0.5792` 5-fold CV across all 54 cases) and the `NeedleUNet3D + AxialTransUNet3D` ensemble (`0.5924`) match the `0.60 ± 0.17` voxel Dice reported by Zaffino et al. (2019) on their 22-case test set, while operating at **native sub-millimeter in-plane resolution** (`0.23–0.625 mm` vs. `1.0 mm³` isotropic resampling) and additionally predicting sub-voxel centerline peaks ($P_{\text{peak}}$) and 3D tangent vectors ($v_x, v_y$).

---

## 3. End-to-End 3D Needle Trajectory Benchmark Across All 54 Cases

### 3.1 Long Interstitial Catheters Benchmark (`759` CY Needles, 54 Cases)

| Cohort Split | Cases | GT Needles | Closest Recall ($\le 3.5\text{ mm}$) | 1-to-1 Hungarian Recall | 1-to-1 Precision | 1-to-1 F1 Score | Mean Shaft Error | Median Shaft Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All 54 BWH Clinical Cases** | 54 | 759 | **`741 / 759` (`97.63%`)** | **`726 / 759` (`95.65%`)** | `83.54%` | **`89.19%`** | **`0.62 mm`** | **`0.54 mm`** |
| **PMB 2019 Held-Out Test Split (`testing.txt`)** | 4 | 72 | **`71 / 72` (`98.61%`)** | **`69 / 72` (`95.83%`)** | **`88.46%`** | **`92.00%`** | **`0.62 mm`** | **`0.52 mm`** |
| **MICCAI 2013 10-Case Cohort** | 10 | 101 | **`99 / 101` (`98.02%`)** | **`97 / 101` (`96.04%`)** | `85.09%` | **`90.23%`** | **`0.58 mm`** | **`0.49 mm`** |
| *Baseline: PMB 2019 3D U-Net (Zaffino et al.)* | *22 test* | *~360* | *`98.5%`* | *—* | *`93.3%`* | *`95.8%`* | *`2.00 ± 3.40 mm`* | *—* |
| *Baseline: MICCAI 2013 Semi-Auto (1 Click/Tip)* | *10 test* | *101* | *`92.1%` ($<2\text{ mm}$)* | *—* | *Manual Seed* | *—* | *`1.05 mm` (correct)* | *`0.58 mm`* |

### 3.2 Complete Multi-Class Cohort (`1,063` Total Needles: `759` Long Interstitial + `304` Short Obturator, 5-Fold OOF)

| Method / Pipeline Configuration | Closest Recall (`1,063` Needles) | 1-to-1 Hungarian Recall | 1-to-1 Precision | 1-to-1 F1 Score | Mean Shaft Error | Median Shaft Error | Total CPU Time / Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Pure 3D CV + `HybridNeedleNet` (Long-needle tuned)** | `788 / 1,063` (`74.13%`) | `770 / 1,063` (`72.44%`) | `88.61%` | `79.71%` | `0.62 mm` | `0.54 mm` | **`0.27 s`** |
| **2. Stage 1 `NeedleUNet3D` + Stage 2 Tangent Tube Tracking** | **`1,012 / 1,063` (`95.20%`)** | `992 / 1,063` (`93.32%`) | `62.90%` | `75.15%` | `0.66 mm` | `0.56 mm` | `0.52 s` |
| **3. Stage 1 `AxialTransUNet3D` + Stage 2 Tangent Tube Tracking** | `1,008 / 1,063` (`94.83%`) | **`997 / 1,063` (`93.79%`)** | `61.62%` | `74.38%` | `0.67 mm` | `0.57 mm` | `0.57 s` |
| **4. Pure `NeedleSliceMPNN` (4-Layer Triplet Collinearity GraphNet)** | `976 / 1,063` (`91.82%`) | `945 / 1,063` (`88.90%`) | `74.41%` | `81.01%` | **`0.55 mm`** | **`0.43 mm`** | `0.42 s` |
| **5. Unified `NeedleUNet3D` + `NeedleSliceMPNN` + `DenseGraphTransformer` (`conf=25%`, Max-Recall)** | **`1,012 / 1,063` (`95.20%`)** | **`996 / 1,063` (`93.70%`)** | `75.11%` | `83.38%` | `0.65 mm` | `0.55 mm` | `0.61 s` |
| **6. Unified `NeedleUNet3D` + `NeedleSliceMPNN` + `DenseGraphTransformer` (`conf=35%`, High-Recall)** | **`1,004 / 1,063` (`94.45%`)** | **`986 / 1,063` (`92.76%`)** | `77.09%` | `84.20%` | `0.64 mm` | `0.54 mm` | `0.61 s` |
| **7. Unified `NeedleUNet3D` + `NeedleSliceMPNN` + `DenseGraphTransformer` (`conf=45%`, Balanced)** | `983 / 1,063` (`92.47%`) | `959 / 1,063` (`90.22%`) | `80.25%` | `84.94%` | `0.63 mm` | `0.53 mm` | `0.61 s` |
| **8. Unified `NeedleUNet3D` + `NeedleSliceMPNN` + `DenseGraphTransformer` (`conf=50%`, Default)** | `969 / 1,063` (`91.16%`) | `944 / 1,063` (`88.81%`) | `81.94%` | **`85.24%`** | `0.63 mm` | `0.53 mm` | `0.61 s` |
| **9. Unified `NeedleUNet3D` + `NeedleSliceMPNN` + `DenseGraphTransformer` (`conf=60%`, High-Precision)** | `959 / 1,063` (`90.22%`) | `934 / 1,063` (`87.86%`) | **`83.62%`** | **`85.69%`** | `0.62 mm` | `0.52 mm` | `0.61 s` |

---

## 4. Per-Stage Computation Time Breakdown (54-Case Full-Volume Uncached Average)

> [!NOTE]
> Earlier benchmark logs (`0.61 s` CPU / `0.35 s` GPU) measured cached-ROI tensor inference (`30×160×160`). In October 2026, we replaced the Python `track_pass` / `trim_trajectory_inside_out` / `dedup_trajs` bottlenecks with native C++17 & Rust kernels (`nf_find_2d_nms_peaks`, `nf_trace_seed_3d`, `nf_trim_trajectory_inside_out`) achieving a **4.1×–4.4× end-to-end speedup** (`8.38 s -> 2.03 s` on `2016-RSNA` `640×416×112`, `1.47 s -> 0.33 s` on `Case016` `320×256×28`) with 100% identical trajectory outputs, and measured true uncached full-volume runtimes across all 54 clinical volumes.

| Pipeline Stage | CPU Runtime (8-core Mean; Std / Hi-Res) | GPU Runtime (`edelweiss` RTX 4090) | Memory / Artifact Size |
| :--- | :---: | :---: | :---: |
| **Stage 0: Rust/C++17 3D Radial & Half-Ring Symmetry Filter** | `0.41 s` (`0.12 s` std / `0.59 s` hi-res) | `0.41 s` (CPU) | `libneedlefinder_rust.so` (`310 KB`) |
| **Stage 1: `NeedleUNet3D` + IRGM Multi-Head 3D Segmentation (ONNX Runtime)** | `0.96 s` (`0.55 s` std / `1.02 s` hi-res) | `0.22 s` | `needle_unet3d.onnx` (`8.95 MB`) |
| **Stage 2A: Native C++17/Rust Seed Tracking + `NeedleSliceMPNN` + Sync-Track** | `1.22 s` (`0.45 s` std / `1.67 s` hi-res) | `0.54 s` | `needle_graphnet.npz` (`1.34 MB`) |
| **Stage 2B: `TipAwareBundleGraphTransformer` + Poly3 + BBox Dedup** | `0.15 s` (`0.08 s` std / `0.19 s` hi-res) | `0.08 s` | `needle_graph_transformer.onnx` (`257 KB`) |
| **Total End-to-End Uncached Pipeline (All Stages Combined)** | **`2.74 s / patient`** (`1.20 s` std / `3.47 s` hi-res) | **`1.25 s / patient`** | **`10.5 MB` total weights** |
