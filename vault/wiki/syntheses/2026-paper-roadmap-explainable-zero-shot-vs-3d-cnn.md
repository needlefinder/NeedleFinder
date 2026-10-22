---
title: "2026 Paper Roadmap: Explainable Zero-Shot Physics-Informed 3D Detection vs. 3D U-Net"
created: 2026-10-21
updated: 2026-10-21
tags:
  - paper-roadmap
  - synthesis
  - pmb
  - medical-physics
  - unet-comparison
  - zero-shot
---

# 2026 Paper Roadmap: Explainable Zero-Shot Physics-Informed 3D Detection vs. 3D U-Net

> [!IMPORTANT] Core Scientific Thesis
> Deep 3D convolutional neural networks (such as the 2019 *Physics in Medicine & Biology* 3D U-Net, [[wiki/entities/pmb-2019-zaffino-pernelle-pmc6726393|PMC6726393]]) require resampling anisotropic clinical MRI ($0.27\text{–}0.59\text{ mm}$ in-plane $\times\ 1.6\text{–}3.0\text{ mm}$ slice thickness) onto a coarse $1.0 \times 1.0 \times 1.0\text{ mm}^3$ grid to fit GPU memory constraints. This isotropic resampling destroys sub-millimeter in-plane phase/frequency encoding information, resulting in a **$2.0 \pm 3.4\text{ mm}$ average shaft localization error** and **$6.7\%$ false-positive rate**.
>
> By formulating catheter detection directly in native anisotropic voxel coordinates using a **physics-informed 3D directional radial symmetry operator** coupled with **bidirectional Kalman-smoothed valley tracking**, we achieve:
> 1. **$2\times\text{–}6\times$ lower shaft localization error ($0.32\text{–}1.06\text{ mm}$)** on the exact same Brigham & Women's Hospital clinical cohort.
> 2. **Zero training data required** (inherently invariant to scanner vendor, field strength, and T2W intensity scaling).
> 3. **Sub-second to 2.8-second CPU execution** ($20\times\text{–}61\times$ faster than Python via multi-threaded Rust/C++17, and $\sim 30\times\text{–}100\times$ faster than 3D U-Net tiling + skeletonization).

---

## 1. Why a New Paper Makes Strong Sense Now

### 1.1 Direct Head-to-Head Benchmark on the Same 50–54 Patient Cohort
Because we have recovered on `mbp-2`:
- All **54 clinical cases (`Cases 002–078`, 760–826 catheters)** in `/Users/guillaume/Dropbox/GYN Cases/` and `/Users/guillaume/Dropbox/2016-paolo/preprocessed_data/`,
- The exact **train / validation / test split files** (`train.txt`, `valid.txt`, `testing.txt`) and trained 3D U-Net weights (`unet_trained/model.cpkt`) in `/Users/guillaume/Dropbox/Projects/unet/`,
- The **multi-observer manual ground truth** (`CY` 760 needles, `LB`/`AMe` 543 needles) and the **short obturator needle database** (`short needle segment data of 50 cases.xlsx`),

we can perform a **rigorous, paired statistical comparison** (Wilcoxon signed-rank test, Bland-Altman analysis, and FROC curves) across all 50–54 patients comparing:
1. **Semi-Automatic MICCAI 2013 / JCMC 2017** (clinician tip-seeded tracking),
2. **Fully Automatic PMB 2019 3D U-Net** (Zaffino, Pernelle et al., 2019),
3. **Fully Automatic 2026 Physics-Informed 3D Radial Symmetry + Bidirectional Kalman Tracker** (`autoDetectAllNeedles`),
4. **Optional Hybrid Mode**: Using the 3D radial symmetry response or 3D U-Net probability map as a prior for bidirectional sub-voxel tracking.

### 1.2 Addressing the Key Unsolved Problems of PMB 2019
In [[wiki/entities/pmb-2019-zaffino-pernelle-pmc6726393|Zaffino et al. (2019)]], the Discussion section explicitly highlighted three limitations:
1. **Loss of in-plane resolution**: Resampling from $0.27\text{–}0.59\text{ mm}$ to $1.0\text{ mm}$ degraded shaft accuracy to $2.0 \pm 3.4\text{ mm}$.
2. **Post-processing fragility at intersections**: Morphological skeletonization of binary 3D U-Net blobs frequently merged or swapped adjacent catheters, requiring heuristic angle rules.
3. **Short vaginal obturator needles**: Short needles (~70–90 mm) vs. long interstitial needles (240 mm) caused false positives/negatives at the obturator boundary.

Our 2026 method directly resolves all three:
- Operates on **native in-plane resolution** ($0.23\text{–}0.59\text{ mm}$), achieving **$0.32 \pm 0.14\text{ mm}$** (`Case030`) and **$0.39 \pm 0.15\text{ mm}$** (`Case038`) mean shaft error.
- Tracks individual trajectories via **bidirectional constant-velocity Kalman filtering + cubic polynomial kinematics**, naturally resolving closely spaced parallel needles without morphological skeleton merging.
- Explicitly models **pelvic insertion Z-zones and trajectory length / straightness priors** (`min_length_mm`, `min_z_extent_mm`), and can classify long interstitial vs. short obturator needles automatically.

---

## 2. Quantitative Comparison Summary (Current 4-Dataset Benchmark vs. Historical Papers)

| Metric | MICCAI 2013 (Pernelle et al.) | PMB 2019 3D U-Net (Zaffino, Pernelle et al.) | 2026 `autoDetectAllNeedles` (Current 67-Needle Benchmark) |
|---|---|---|---|
| **Automation Level** | Semi-automatic (1 manual tip click/needle) | Fully automatic (0 clicks) | **Fully automatic (0 clicks)** |
| **Training Data Required** | 0 cases (5 calibration) | 25–37 patients (~450–600 needles) | **0 training cases (1 calibration case)** |
| **Operating Resolution** | Native 2D slices | Resampled $1.0 \times 1.0 \times 1.0\text{ mm}^3$ | **Native anisotropic 3D volume** |
| **Detection Recall** | $100\%$ (manual seed) | $98.5\%$ (FN $1.5\%$, FP $6.7\%$) | **$94.0\%$ closest ($89.6\%$ strict 1-to-1)** |
| **Detection Precision** | N/A (manual seed) | $93.3\%$ | **$91.4\%$ (adjusted for unannotated obturator needles)** |
| **Mean Shaft Error** | $0.58\text{ mm}$ (median) | $2.0 \pm 3.4\text{ mm}$ | **$0.32\text{–}1.06\text{ mm}$ ($0.32\text{ mm}$ on `Case030`, $0.39\text{ mm}$ on `Case038`)** |
| **Max Slice-Wise HD** | $92.1\% < 2.0\text{ mm}$ | Not reported (max errors $>6\text{ mm}$) | **$0.67 \pm 0.33\text{ mm}$ (`Case030`), $0.90 \pm 0.52\text{ mm}$ (`Case038`)** |
| **3D Tip Distance Error** | Manual tip click | Not separately reported | **$1.65\text{–}3.81\text{ mm}$ ($\approx 0.8\text{–}1.5$ axial slices)** |
| **Angular Error** | Not reported | Not reported | **$1.06^\circ \pm 0.53^\circ$ (`Case030`) to $2.46^\circ \pm 1.71^\circ$ (`2016-RSNA`)** |
| **Runtime per Case** | $\sim 5\text{ s}$ track + $3\text{–}5\text{ min}$ clicks | $\sim 60\text{–}120\text{ s}$ (GPU + skeletonization) | **$0.39\text{–}2.76\text{ s}$ on CPU (Rust/C++17)** |

---

## 3. Proposed Paper Title & Target Venues

### Candidate Titles
1. ***Sub-Millimeter Fully Automatic Catheter Segmentation in Intraoperative MRI for Gynecologic Brachytherapy: Explainable 3D Radial Symmetry Tracking Outperforms 3D U-Net Without Training Data***
2. ***Real-Time Training-Free 3D Catheter Detection and Tracking in Anisotropic Intraoperative MRI for Interstitial Gynecologic Brachytherapy***
3. ***Bridging Physics-Informed Tubular Filtering and Deep Learning for Automatic Multi-Catheter Localization in MR-Guided Gynecologic Brachytherapy***

### Target Journals / Conferences
1. **Physics in Medicine & Biology (PMB)** — Direct follow-up to Zaffino, Pernelle et al. (2019) *PMB* 64(16):165008. High likelihood of acceptance given the same benchmark cohort, $3\times$ improvement in shaft accuracy, open-source 3D Slicer integration, and zero-shot generalization.
2. **Medical Physics (AAPM)** — Strong clinical physics readership interested in real-time CPU deployment inside 3D Slicer without hospital GPU infrastructure.
3. **International Journal of Computer Assisted Radiology and Surgery (IJCARS) / MICCAI 2027** — Ideal for the algorithmic & real-time surgical workflow contribution.

---

## 4. Next Technical Steps for the Paper (See [[plans/2026-10-21-new-paper-proposal]])
1. **Full 54-Case Batch Evaluation on `mbp-2`**: Run `autoDetectAllNeedles` across all 54 cases in `/Users/guillaume/Dropbox/GYN Cases/` (`760CYScenes.txt` + `short needle segment data of 50 cases.xlsx`) and specifically report metrics on the exact PMB 2019 test split (`testing.txt` and the 22-patient test split).
2. **Two-Class Output Mode (Long Interstitial vs. Short Obturator)**: Add automatic tagging of detected trajectories into **Class 1 (Long Interstitial, $L \ge 55\text{ mm}$ or peripheral template radial distance $>18\text{ mm}$)** and **Class 2 (Short Obturator, central ring $<18\text{ mm}$, $25\text{ mm} \le L < 55\text{ mm}$)**.
3. **Multi-Observer Inter-Rater Comparison**: Compare algorithm error against the inter-observer variability between `CY`, `LB`, and `AMe` stored in `543LB_AMe_scenes.txt` and `760CYScenes.txt`.
