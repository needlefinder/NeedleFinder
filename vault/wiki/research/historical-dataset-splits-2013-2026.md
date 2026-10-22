---
title: Historical Dataset Splits — MICCAI 2013, JCMC 2017, RSNA 2016, and PMB 2019 (PMC6726393)
type: research
category: dataset
created: 2026-10-21
updated: 2026-10-21
sources:
  - 2013-miccai-pernelle-catheter-segmentation
  - 2016-rsna-demo-dataset
  - 2017-jcmc-mastmeyer-pernelle-model-based-catheters
  - 2019-pmb-zaffino-pernelle-3d-cnn-pmc6726393
tags: [datasets, miccai-2013, rsna-2016, pmc6726393, train-val-test, bwh-amigo]
---

# Historical Dataset Splits (2013–2026)

This document catalogs the exact clinical datasets, file locations in `/Users/guillaume/Dropbox/`, and training/validation/test splits used across all four generations of the BWH/Harvard Medical School gynecologic interstitial brachytherapy catheter segmentation studies.

---

## 1. MICCAI 2013 Study (Pernelle et al., LNCS 8150, 2013)

- **Paper**: *Validation of Catheter Segmentation for MR-guided Gynecologic Cancer Brachytherapy* (`Dropbox/2013 MICCAI/Latex/miccai_submission.tex`)
- **Algorithm**: Interactive distal tip initialization + axial template limit plane + Fibonacci-spaced conic ray search (`algoVers = 1, 2`) + 5th-degree Bézier curve fitting.
- ** MRI Sequences (3T Siemens)**:
  - 2D T2-weighted Fast Spin Echo (FSE): $\text{TR/TE} = 3000/120\text{ ms}$, $0.2 \times 0.3 \times 2.0\text{ mm}^3$
  - 3D FSE (Siemens SPACE): $\text{TR/TE} = 3000/160\text{ ms}$, $0.4 \times 0.4 \times 1.0\text{ mm}^3$
  - 3D balanced SSFP (3D bSSFP / 3D WB-bSSFP): $\text{TR/TE} = 6.8/3.4\text{ ms}$, $0.6 \times 0.6 \times 1.6\text{ mm}^3$
- **Dataset Split**:
  1. **Parameter Learning / Calibration Set ($N = 5$ patients)**: 5 retrospective patient MRIs not included in the validation set, used to optimize linear pixel/gradient weights $p$ and Gaussian attenuation $\sigma$.
  2. **Phantom Validation Set ($N = 1$ phantom, $12$ catheters)**: Custom transparent gel-wax phantom with a Syed-Neblett template, central vaginal obturator, and 12 MR-compatible plastic catheters (straight, bent, and touching), imaged with 3D bSSFP MRI ($0.6 \times 0.6 \times 1.6\text{ mm}^3$) and gold-standard CT.
     - **MR–MR Max Hausdorff Distance**: $1.5\text{ mm}$
     - **MR–CT Max Hausdorff Distance**: $3.5\text{ mm}$ (mean $2.3\text{ mm}$, variance $0.5\text{ mm}^2$)
  3. **Clinical Patient Validation Set ($N = 10$ patients, $101$ catheters)**: Listed in `Dropbox/GYN Cases/MICCAI13scenes.txt` and `GynCase Data Overview.xlsx`:

| MICCAI13 Paper ID | BWH Cohort ID | Catheters (Non-Obturator) | Obturator Needles | Sequence | MICCAI 2013 Max HD (All) | MICCAI 2013 Mean HD (All) | Incorrect ($>2\text{ mm}$) | Mean HD* (Correct) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Case 1** | `Case 024` | 7 | 6 | 2D AX T2 TSE | $1.08\text{ mm}$ | $0.99\text{ mm}$ | 0 | $0.99\text{ mm}$ |
| **Case 2** | `Case 028` | 14 | 3 | 2D AX T2 TSE | $6.50\text{ mm}$ | $1.81\text{ mm}$ | 2 | $1.15\text{ mm}$ |
| **Case 3** | `Case 029` | 8 | 6 | 2D AX T2 TSE | $8.93\text{ mm}$ | $2.74\text{ mm}$ | 3 | $0.93\text{ mm}$ |
| **Case 4** | `Case 030` | 15 | 7 | 2D AX T2 TSE | $1.00\text{ mm}$ | $0.96\text{ mm}$ | 0 | $0.96\text{ mm}$ |
| **Case 5** | `Case 031` | 16 | 7 | 3D AX SPACE | $10.43\text{ mm}$ | $2.19\text{ mm}$ | 2 | $1.01\text{ mm}$ |
| **Case 6** | `Case 033` | 5 | 6 | 2D AX T2 TSE | $1.00\text{ mm}$ | $0.97\text{ mm}$ | 0 | $0.97\text{ mm}$ |
| **Case 7** | `Case 034` | 7 | 6 | 2D AX T2 TSE | $1.20\text{ mm}$ | $1.06\text{ mm}$ | 0 | $1.06\text{ mm}$ |
| **Case 8** | `Case 037` | 6 | 6 | 2D AX T2 TSE | $1.92\text{ mm}$ | $1.33\text{ mm}$ | 0 | $1.33\text{ mm}$ |
| **Case 9** | `Case 038` | 9 | 6 | 2D AX T2 TSE | $1.22\text{ mm}$ | $1.05\text{ mm}$ | 0 | $1.05\text{ mm}$ |
| **Case 10** | `Case 040` | 14 | 7 | 3D AX SPACE | $17.56\text{ mm}$ | $3.28\text{ mm}$ | 2 | $1.19\text{ mm}$ |
| **Total / Summary** | **10 Cases** | **101** | **60** | **T2 / SPACE** | — | — | **8 / 101** ($92.1\%$ correct) | **$\approx 1.05\text{ mm}$** |

---

## 2. MICCAI 2015 / JCMC 2017 Re-Evaluation & Expansion (Mastmeyer, Pernelle et al.)

- **Files**: `Dropbox/2015-AndreGuillaume-NeedleFinder/iGyne Needle Re-Evaluation Study2015.xlsx`, `Dropbox/GYN Cases/543LB_AMe_scenes.txt` (41 cases, 543 needles), `Dropbox/GYN Cases/760CYScenes.txt` (54 cases, 760 needles).
- **Algorithmic Additions**:
  - Mechanical deflection cone search (`algoVers = 3`)
  - Multi-needle rationality check (`algoVers = 4, 5`) penalizing trajectories that cross or merge with neighboring needles
  - Dedicated obturator needle tracking (`obturatorNeedle`) for the 6–8 short central channels around the vaginal obturator (`short needle segment data of 50 cases.xlsx`, evaluating 139 obturator needles across 22 cases).

---

## 3. RSNA 2016 Demo Dataset (`Dropbox/2016-RSNA` = `Case 050`)

- **Volume**: `12 AX space trial 1.6 SLICE COVER TEMPLATE.6 SLICE COVER TEMPLATE.MR._.18.1.2013.07.29.09.48.42.911.59151707.nrrd`
  - Dimensions: $320 \times 320 \times 128$, Voxel spacing: $0.5859 \times 0.5859 \times 1.3000\text{ mm}^3$ (3D T2 SPACE).
- **Ground Truth**:
  - **20 long interstitial needles** (`manual-seg_0.vtk` .. `manual-seg_19.vtk`)
  - **6 short obturator needles** (`Case 050/NRRD/Manual-shortOneedle/manual-seg_1..6.vtk`)
  - Template slice marker: `template slice position.acsv` ($z_{\text{RAS}} = -48.5\text{ mm}$, $k_{\text{lim}} = 49$).

---

## 4. PMB 2019 3D U-Net CNN Study with Paolo Zaffino (PMC6726393)

- **Citation**: P. Zaffino, G. Pernelle, A. Mastmeyer, A. Mehrtash, H. Zhang, R. Kikinis, T. Kapur, M. F. Spadea, *"Fully automatic catheter segmentation in MRI with 3D convolutional neural networks: application to MRI-guided gynecologic brachytherapy"*, *Physics in Medicine & Biology*, 64(16):165008, 2019. PMID: 31272095, PMCID: **PMC6726393**.
- **Code & Preprocessed Data Locations on `mbp-2`**:
  - Preprocessing script: `Dropbox/2016-paolo/join_needles.py`
  - Resampled volumes & labelmaps ($1.0 \times 1.0 \times 1.0\text{ mm}^3$): `Dropbox/2016-paolo/preprocessed_data/LabelMapsNEW2_1.00-1.00-1.00/` (54 cases: `002` through `078`)
  - 3D U-Net TensorFlow training/inference code & checkpoint: `Dropbox/Projects/unet/` (`3d-unet-tf-tiles-1class-multiple-gpus.py`, `unet.py`, `functions.py`, `unet_trained/model.cpkt`, tile size $148^3 \to 60^3$ output).
- **Dataset Splits**:
  - **In the TensorFlow codebase (`Dropbox/Projects/unet/`)**:
    - `train.txt` (**37 cases**): `007, 008, 012, 014, 016, 021, 022, 023, 024, 025, 026, 027, 029, 030, 031, 033, 034, 035, 037, 038, 039, 040, 041, 042, 044, 045, 048, 049, 050, 053, 056, 058, 059, 060, 063, 064, 069`
    - `valid.txt` (**7 cases** + rotations `rot1`, `rot2` = 21 volumes): `070, 071, 072, 074, 075, 076, 077`
    - `testing.txt` (**4 held-out cases**): `002, 028, 051, 054`
  - **In the published PMB 2019 paper (50 patients, 826 catheters, 10–35 catheters/patient)**:
    - **Training set**: 25 patients
    - **Validation set**: 3 patients
    - **Testing set**: 22 patients
- **Published PMB 2019 Results (on held-out test set)**:
  - **Voxel-wise Dice Similarity Coefficient (DSC)**: $0.60 \pm 0.17$
  - **Average Shaft Distance Error**: $2.0 \pm 3.4\text{ mm}$
  - **False Positive (FP) Catheter Rate**: $6.7\%$ (Precision $\approx 93.3\%$)
  - **False Negative (FN) Catheter Rate**: $1.5\%$ (Recall $\approx 98.5\%$)

---

## 5. 2026 Zero-Shot 3D Tubular + Momentum Tracker Split (`auto-needle-detection`)

Because the 2026 `autoDetectAllNeedles` algorithm is a **physics-informed, zero-shot geometric and tubular filtering algorithm** (requiring no neural network weights), datasets are partitioned into:
1. **Algorithm Design & Calibration Case (`2016-RSNA` / `Case 050`, 20 GT needles)**: Used to design the 6-angle symmetric diameter-pair dark-tube filter (`r = 1.7, 2.5, 3.4 mm`), tissue floor (`0.45 * body_thresh`), and template-cluster bounding-box midpoint filter.
2. **Hyperparameter Validation Case (`Case 016`, 17 GT needles)**: Used to validate spacing invariance across anisotropic 2D T2 FSE ($0.43 \times 0.43 \times 3.0\text{ mm}^3$) vs. 3D SPACE ($0.59 \times 0.59 \times 1.3\text{ mm}^3$) and tune the base deduplication threshold ($2.25\text{ mm}$).
3. **Held-Out Clinical Test Cases (`Case 030` with 15 GT needles; `Case 038` with 15 GT needles = 9 long + 6 short obturator needles)**: Two original MICCAI 2013 benchmark cases (`Case 4` and `Case 9`) used to evaluate generalization across both standard long interstitial needles and mixed long + short obturator needle configurations.
