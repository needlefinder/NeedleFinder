---
title: 2026 Auto-Needle Detection Quantitative Evaluation Report
type: research
category: evaluation
created: 2026-10-21
updated: 2026-10-21
sources:
  - 2026-10-21-auto-needle-detection-benchmark.json
  - 2013-miccai-pernelle-catheter-segmentation
  - 2016-rsna-demo-dataset
  - 2019-pmb-zaffino-pernelle-3d-cnn-pmc6726393
tags: [evaluation, hausdorff-distance, tip-error, shaft-error, rust, benchmark, auto-detection]
---

# Quantitative Evaluation Report: Zero-Shot 3D Multi-Needle Detection & Segmentation (`autoDetectAllNeedles`)

## 1. Executive Summary

We evaluated the new fully automatic, zero-seed-click 3D multi-needle detection and segmentation pipeline ([`NeedleFinderLogic.autoDetectAllNeedles`](file:///mnt/SSD/github/needlefinder/NeedleFinder/NeedleFinder/NeedleFinderLib/TrackingAlgorithms.py#L3009) + [`nf_compute_3d_sym_response`](file:///mnt/SSD/github/needlefinder/NeedleFinder/NeedleFinder/NeedleFinderRust/src/lib.rs#L786)) across **4 clinical 3T MRI gynecologic brachytherapy datasets** comprising **67 expert-segmented ground-truth (GT) catheters**.

- **Detection Accuracy**:
  - **Closest-Trajectory Recall**: **$94.0\%$** ($63/67$ clinical needles localized within $\le 3.5\text{ mm}$ mean shaft distance: $19/20$ on `2016-RSNA`, $16/17$ on `Case016`, $14/15$ on `Case030`, $14/15$ on `Case038`).
  - **Strict 1-to-1 Bipartite Recall / Precision / F1**: **$89.6\%$ Recall** ($60/67$), **$89.9\%$ Precision** on fully annotated volumes (`2016-RSNA`, `Case016`, `Case038`), **$87.4\%$ mean F1**.
- **Geometric Segmentation Quality**:
  - **Mean Shaft Error ($\Delta xy_{\text{shaft}}$)**: **$0.65 \pm 0.62\text{ mm}$** overall (**$0.32 \pm 0.14\text{ mm}$** on held-out `Case030` and **$0.39 \pm 0.15\text{ mm}$** on held-out `Case038` — **$3\times$ lower shaft error than the $2.0 \pm 3.4\text{ mm}$ reported for the 2019 3D U-Net CNN in PMC6726393**).
  - **Maximum Slice-Wise Hausdorff Distance ($\text{HD}_{xy,\max}$)**: **$0.67 \pm 0.33\text{ mm}$** (median $0.68\text{ mm}$, max $1.56\text{ mm}$) on `Case030`; **$0.90 \pm 0.52\text{ mm}$** (median $0.75\text{ mm}$, max $2.02\text{ mm}$) on `Case038`; **$1.50 \pm 1.50\text{ mm}$** (median $0.88\text{ mm}$) on `Case016`; and **$2.68 \pm 2.61\text{ mm}$** (median $1.43\text{ mm}$) on `2016-RSNA`.
  - **3D Symmetric Mean Surface/Centerline Distance ($\text{MSD}_{3\text{D}}$)**: **$0.85\text{–}1.83\text{ mm}$** (median **$0.66\text{–}1.09\text{ mm}$**).
  - **Base / Template Entry Error ($\Delta xy_{\text{base}}$)**: **$0.37\text{–}1.66\text{ mm}$** (median **$0.46\text{–}0.72\text{ mm}$**, sub-millimeter across all 4 cases).
  - **In-Plane Tip Error ($\Delta xy_{\text{tip}}$)**: **$1.75\text{–}2.52\text{ mm}$** (median **$0.82\text{–}2.04\text{ mm}$**).
  - **Angular Trajectory Error ($\Delta \theta$)**: **$1.84^\circ\text{–}3.85^\circ$** (median **$1.16^\circ\text{–}2.29^\circ$**).
- **Computational Performance (Speed)**:
  - **3D Symmetric Tubular Filter (`nf_compute_3d_sym_response`)**: **$19.4\text{ ms}$** on 2D T2 FSE ($320 \times 256 \times 28$) and **$295\text{–}377\text{ ms}$** on high-resolution 3D SPACE / T2 volumes ($640 \times 416 \times 112$ and $768 \times 768 \times 64$), achieving a **$19.7\times\text{ to }60.8\times$ speedup** over NumPy/SciPy ($383\text{–}20,096\text{ ms}$).
  - **Total End-to-End Volume Runtime**: **$0.39\text{ s}$** (`Case016`) to **$1.66\text{–}2.76\text{ s}$** (`Case038`, `2016-RSNA`, `Case030`) for **all 15–20 needles combined** ($\approx \mathbf{0.02\text{–}0.14\text{ s}}$ **per needle** on CPU, with zero GPU requirement).

---

## 2. Dataset Partitioning: Design Calibration, Validation, and Held-Out Testing

Because `autoDetectAllNeedles` is a physics-informed, zero-shot geometric algorithm (no neural network weights), we partitioned the clinical benchmark cohorts into three strict roles to prevent overfitting of geometric priors:

| Dataset ID | BWH Cohort ID | Split Role | MRI Sequence & Resolution | Volume Dimensions | Voxel Spacing ($\text{mm}^3$) | GT Needles ($N_{\text{GT}}$) | Why Assigned to This Split |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`2016-RSNA`** | `Case 050` | **1. Design / Calibration** | 3D Siemens SPACE T2 | $640 \times 416 \times 112$ | $0.469 \times 0.469 \times 1.60$ | **20** (long) | Used to design the 6-angle symmetric diameter-pair dark-tube filter ($r \in \{1.7, 2.5, 3.4\}\text{ mm}$), anterior muscle tissue floor ($0.45 \times T_{\text{body}}$), and Syed-Neblett bounding-box midpoint filter. |
| **`Case016`** | `Case 016` | **2. Validation (Hyperparameter Tuning)** | 2D Axial T2 TSE | $320 \times 256 \times 28$ | $0.875 \times 0.875 \times 3.00$ | **17** | Used to validate invariance to thick-slice anisotropic spacing ($3.0\text{ mm}$ Z slices) and tune template-hole base deduplication ($2.25\text{ mm}$). |
| **`Case030`** | `Case 030` (MICCAI13 #4) | **3. Held-Out Test Set A** | 2D Axial T2 FSE | $640 \times 416 \times 112$ | $0.469 \times 0.469 \times 1.60$ | **15** (long; +7 obturator unlabelled in folder) | Held-out MICCAI 2013 benchmark case with Vitamin E template markers and 15 long + 7 short obturator catheters. |
| **`Case038`** | `Case 038` (MICCAI13 #9) | **4. Held-Out Test Set B** | 2D Axial T2 FSE | $768 \times 768 \times 64$ | $0.417 \times 0.417 \times 2.00$ | **15** (9 long + 6 short obturator) | Held-out MICCAI 2013 benchmark case testing simultaneous detection of 9 long interstitial needles (`Manual CY`) and 6 short obturator needles (`Manual-shortOneedle`). |

---

## 3. Detection Performance (Recall, Precision, F1)

We report detection accuracy under two standard matching protocols ($\tau_{\text{shaft}} = 3.5\text{ mm}$ mean in-plane shaft distance):
1. **Closest-Trajectory Protocol** (used in Zaffino et al. 2019): A GT needle is counted as detected if at least one predicted needle trajectory lies within $3.5\text{ mm}$ mean shaft distance.
2. **Strict 1-to-1 Bipartite Protocol**: Greedy minimum-distance 1-to-1 assignment where each predicted trajectory can match at most one GT needle.

| Dataset | Split | $N_{\text{GT}}$ | $N_{\text{det}}$ | Closest-Traj Recall | Strict 1-to-1 TP | Strict 1-to-1 Recall | Strict 1-to-1 Precision | Strict 1-to-1 F1 | Notes on Unmatched / Extra Detections |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`2016-RSNA`** | Design / Calibration | 20 | 19 | **19 / 20 (95.0%)** | 18 | **90.0%** | **94.7%** | **92.3%** | 2 converging needles share 1 track near tip; `manual-seg_14` is at extreme lateral edge. |
| **`Case016`** | Validation | 17 | 17 | **16 / 17 (94.1%)** | 15 | **88.2%** | **88.2%** | **88.2%** | `manual-seg_14` & `15` converge within $1.5\text{ mm}$ in 3 mm thick slices. |
| **`Case030`** | Held-Out Test A | 15* | 19 | **14 / 15 (93.3%)** | 14 | **93.3%** | **73.7%** *(94.7% incl. obt.)* | **82.4%** | *`Case030` has 7 additional short obturator needles in `Manual-shortOneedle`; 4 of the 5 "FPs" are real obturator needles. |
| **`Case038`** | Held-Out Test B | 15 | 15 | **14 / 15 (93.3%)** | 13 | **86.7%** | **86.7%** | **86.7%** | Detects 9/9 long needles ($100\%$) + 4–5/6 short obturator needles at $k_{\text{lim}}=37$. |
| **Overall** | **All 4 Cases** | **67** | **70** | **63 / 67 (94.0%)** | **60** | **89.6%** | **85.7%** *(91.4% adj.)* | **87.6%** | Sub-millimeter shaft localization across both 2D T2 FSE and 3D SPACE. |

---

## 4. Geometric Segmentation Accuracy (Hausdorff, Shaft, Tip, Base, Angle)

All distance metrics are computed in physical 3D RAS coordinates ($\text{mm}$) across matched needle pairs (using template limit $k_{\text{lim}}$):

### 4.1 In-Plane Shaft Error & Slice-Wise Maximum Hausdorff Distance ($\text{HD}_{xy}$)

| Dataset | Mean Shaft Error $\Delta xy$ ($\text{mm}$) | Median Shaft Error (IQR) ($\text{mm}$) | Slice-Wise Max HD $\text{HD}_{xy,\max}$ ($\text{mm}$) | Median $\text{HD}_{xy,\max}$ (IQR) ($\text{mm}$) | 95th Percentile $\text{HD}_{xy,\text{p95}}$ ($\text{mm}$) | Min–Max $\text{HD}_{xy,\max}$ ($\text{mm}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2016-RSNA`** | $1.06 \pm 0.96$ | $0.60$ ($1.14$) | $2.68 \pm 2.61$ | $1.43$ ($2.53$) | $2.29 \pm 2.31$ (med $0.99$) | $0.68\text{ – }8.20$ |
| **`Case016`** | $0.72 \pm 0.45$ | $0.53$ ($0.30$) | $1.50 \pm 1.50$ | $0.88$ ($0.48$) | $1.33 \pm 1.35$ (med $0.79$) | $0.60\text{ – }5.33$ |
| **`Case030` (Test)** | **$0.32 \pm 0.14$** | **$0.26$ ($0.16$)** | **$0.67 \pm 0.33$** | **$0.68$ ($0.46$)** | **$0.59 \pm 0.27$ (med $0.58$)** | **$0.26\text{ – }1.56$** |
| **`Case038` (Test)** | **$0.39 \pm 0.15$** | **$0.38$ ($0.24$)** | **$0.90 \pm 0.52$** | **$0.75$ ($0.65$)** | **$0.78 \pm 0.45$ (med $0.65$)** | **$0.30\text{ – }2.02$** |
| **Held-Out Test Mean** | **$0.35 \pm 0.15$** | **$0.31$** | **$0.78 \pm 0.44$** | **$0.71$** | **$0.68 \pm 0.37$** | **$0.26\text{ – }2.02$** |

> [!IMPORTANT]
> On the two held-out MICCAI 2013 clinical test cases (`Case030` and `Case038`), **every single detected long needle has a slice-wise Maximum Hausdorff Distance $\text{HD}_{xy,\max} \le 1.56\text{ mm}$** (well below the $2.0\text{ mm}$ clinical acceptability threshold defined in MICCAI 2013) and a **mean shaft error of $0.32\text{–}0.39\text{ mm}$** (less than 1 pixel width!).

### 4.2 Tip Error, Base Entry Error, 3D Symmetric Distance ($\text{MSD}_{3\text{D}}$), and Angular Error

| Dataset | Base Entry Error $\Delta xy_{\text{base}}$ ($\text{mm}$) | In-Plane Tip Error $\Delta xy_{\text{tip}}$ ($\text{mm}$) | Axial Tip Depth Error $|\Delta z_{\text{tip}}|$ ($\text{mm}$) | 3D Tip Error $\|\Delta \mathbf{p}_{\text{tip}}\|_2$ ($\text{mm}$) | 3D Symmetric MSD ($\text{mm}$) | Trajectory Angle Error $\Delta \theta$ ($^\circ$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2016-RSNA`** | $1.66 \pm 2.42$ (med **$0.72$**) | $2.52 \pm 2.20$ (med **$1.98$**) | $9.96 \pm 7.46$ (med $10.40$) | $10.64 \pm 7.26$ (med $10.85$) | $1.51 \pm 1.01$ (med **$1.09$**) | $1.84^\circ \pm 1.45^\circ$ (med **$1.16^\circ$**) |
| **`Case016`** | $0.97 \pm 1.23$ (med **$0.69$**) | $1.75 \pm 1.75$ (med **$0.82$**) | $6.70 \pm 8.16$ (med **$1.50$**) | $7.04 \pm 8.24$ (med **$2.08$**) | $1.40 \pm 1.09$ (med **$0.80$**) | $2.73^\circ \pm 2.01^\circ$ (med **$2.29^\circ$**) |
| **`Case030` (Test)** | **$0.37 \pm 0.16$** (med **$0.35$**) | $2.08 \pm 1.77$ (med **$2.04$**) | $12.80 \pm 12.56$ (med $8.00$) | $13.10 \pm 12.42$ (med $8.47$) | $1.83 \pm 2.26$ (med **$0.78$**) | $1.91^\circ \pm 1.35^\circ$ (med **$1.58^\circ$**) |
| **`Case038` (Test)** | **$0.41 \pm 0.18$** (med **$0.47$**) | $1.95 \pm 1.59$ (med **$1.94$**) | $5.48 \pm 5.11$ (med **$3.00$**) | $5.90 \pm 5.25$ (med **$3.67$**) | **$0.85 \pm 0.63$** (med **$0.66$**) | $3.85^\circ \pm 3.57^\circ$ (med **$2.01^\circ$**) |

*Note on Axial Tip Depth Error ($|\Delta z_{\text{tip}}|$)*: In T2 MRI without exogenous contrast, the distal catheter tip transitions gradually into dark pelvic fascia/bowel loops or susceptibility "figure-8" blooming artifacts (whichDiMaio et al. 2006 showed can shift apparent tip void extent by $4\text{–}5\text{ mm}$). While in-plane tip localization ($\Delta xy_{\text{tip}}$) is **$0.82\text{–}2.04\text{ mm}$ median**, the 3D bidirectional Hausdorff distance ($d_{H,3\text{D}}$) is dominated by the axial termination difference $|\Delta z_{\text{tip}}|$ at the distal tip (`Case016` median $d_{H,3\text{D}} = 5.26\text{ mm}$, `Case038` median $d_{H,3\text{D}} = 3.67\text{ mm}$). In NeedleFinder's clinical workflow, users can refine any distal tip in 1 click using the existing `findTip` /Assisted mode if needed.

---

## 5. Automatic vs. Manual Template Limit Slice ($k_{\text{lim}}$)

[`NeedleFinderLogic.estimateAxialLimitSlice`](file:///mnt/SSD/github/needlefinder/NeedleFinder/NeedleFinder/NeedleFinderLib/TrackingAlgorithms.py#L2929) automatically estimates the template plane when the user does not set a limit marker:

| Dataset | Expert Marker $k_{\text{lim,GT}}$ | Auto-Estimated $k_{\text{lim,auto}}$ | Slice Difference $|\Delta k|$ | Physical Difference $|\Delta z|$ ($\text{mm}$) | Auto-$k_{\text{lim}}$ Mean Shaft Error ($\text{mm}$) | Auto-$k_{\text{lim}}$ Median $\text{HD}_{xy,\max}$ ($\text{mm}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`2016-RSNA`** | 49 | 55 | $+6$ slices | $9.6\text{ mm}$ | $0.92 \pm 0.75$ | $1.26\text{ mm}$ |
| **`Case016`** | 10 | 9 | $-1$ slice | **$3.0\text{ mm}$** | $0.66 \pm 0.41$ | $0.89\text{ mm}$ |
| **`Case030`** | 39 | 42 | $+3$ slices | **$4.8\text{ mm}$** | $0.33 \pm 0.14$ | $0.69\text{ mm}$ |
| **`Case038`** | 37 (long) / 27 (short) | 27 | $0$ slices vs. short base | **$0.0\text{ mm}$** (at obturator base) | $0.40 \pm 0.13$ | $0.58\text{ mm}$ |

---

## 6. Computational Speed & Multi-Threaded Rust/C++ Acceleration

Benchmarked on Linux (`think`, 24 hardware threads) and verified on Apple Silicon (`mbp-2`):

| Dataset | Volume Size ($I \times J \times K$) | Active ROI Slices ($K - k_0$) | NumPy/SciPy 3D Filter Time ($\text{ms}$) | **Rust/C++17 3D Filter Time ($\text{ms}$)** | **Filter Speedup** | Seed + 3D Momentum Tracking + Polyfit ($\text{ms}$) | **Total End-to-End Runtime ($\text{s}$)** | **Runtime per Detected Needle ($\text{s}$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Case016`** | $320 \times 256 \times 28$ | 19 | $382.9\text{ ms}$ | **$19.4\text{ ms}$** | **$19.7\times$** | $371.3\text{ ms}$ | **$0.39\text{ s}$** | **$0.023\text{ s}$** |
| **`Case038`** | $768 \times 768 \times 64$ | 28 | $15,805.9\text{ ms}$ | **$327.8\text{ ms}$** | **$48.2\times$** | $1,330.0\text{ ms}$ | **$1.66\text{ s}$** | **$0.111\text{ s}$** |
| **`2016-RSNA`** | $640 \times 416 \times 112$ | 64 | $17,938.0\text{ ms}$ | **$295.2\text{ ms}$** | **$60.8\times$** | $2,135.1\text{ ms}$ | **$2.43\text{ s}$** | **$0.128\text{ s}$** |
| **`Case030`** | $640 \times 416 \times 112$ | 74 | $20,096.1\text{ ms}$ | **$377.4\text{ ms}$** | **$53.3\times$** | $2,378.1\text{ ms}$ | **$2.76\text{ s}$** | **$0.145\text{ s}$** |

---

## 7. Comparison Across Generations (2013 $\to$ 2017 $\to$ 2019 $\to$ 2026)

| Metric / Capability | MICCAI 2013 (Pernelle et al.) | JCMC 2017 (Mastmeyer, Pernelle et al.) | PMB 2019 3D U-Net (Zaffino, Pernelle et al., PMC6726393) | **2026 NeedleFinder Auto-Detect (`autoDetectAllNeedles`)** |
| :--- | :--- | :--- | :--- | :--- |
| **Manual Tip Clicks Required?** | **Yes** (1 click per needle = 15–35 clicks/case) | **Yes** (1 click per needle) | **No** (Fully automatic) | **No** (Fully automatic, 0 clicks) |
| **Training Data / GPU Required?** | No (5 cases for $\sigma, p$ tuning) | No (GA parameter tuning) | **Yes** (25–37 training MRI cases, $148^3$ 3D U-Net, GPU + TensorFlow) | **No** (Zero-shot physics-informed tubular filter, CPU-only) |
| **Catheter Detection Recall** | N/A (tips given manually) | N/A (tips given manually) | $98.5\%$ ($1.5\%$ FN) | **$94.0\%$** ($63/67$ closest) / **$89.6\%$** (1-to-1) |
| **Mean Shaft Distance Error** | $1.05\text{ mm}$ (on $92.1\%$ correct needles) | $0.95\text{–}1.15\text{ mm}$ | $2.0 \pm 3.4\text{ mm}$ | **$0.32\text{–}1.06\text{ mm}$** (Test mean: **$0.35 \pm 0.15\text{ mm}$**) |
| **Max Slice-Wise HD ($\text{HD}_{xy}$)** | $1.00\text{–}1.92\text{ mm}$ (up to $17.6\text{ mm}$ on failed) | $0.99\text{–}1.67\text{ mm}$ (correct) | Not reported (DSC = $0.60 \pm 0.17$) | **$0.67\text{–}0.90\text{ mm}$** on test set ($100\%$ of test long needles $< 1.6\text{ mm}$) |
| **Runtime per Patient Volume** | $\approx 5\text{–}10\text{ min}$ (manual tip picking) + $0.3\text{ s}$/needle | $\approx 5\text{–}10\text{ min}$ (manual tip picking) + $0.2\text{ s}$/needle | $\approx 60\text{–}120\text{ s}$ (resample to $1\text{ mm}^3$ + 3D U-Net tiling + postproc) | **$0.39\text{–}2.76\text{ s}$ total** (native resolution, zero preprocessing) |
| **3D Slicer Extension Integration** | Slicer 4.2 (Python 2, legacy Annotation) | Slicer 4.5 (Python 2) | External Python/TensorFlow scripts (never in Slicer Catalog) | **Slicer 5.10+ Native (`Markups`, Rust+C++17, zero external dependencies)** |
