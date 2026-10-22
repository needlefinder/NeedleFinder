---
title: MICCAI 2013 (Pernelle et al.)
created: 2026-10-21
tags:
  - entity
  - publication
  - miccai
---

# MICCAI 2013 (Pernelle et al.)

## Citation
- **Title**: *Validation of Catheter Segmentation for MR-Guided Gynecologic Cancer Brachytherapy*
- **Authors**: Guillaume Pernelle, Alireza Mehrtash, Lauren Barber, Antonio Damato, Wang, Tina Kapur, et al.
- **Venue**: *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2013)*, LNCS vol. 8151, pp. 380–387, 2013.
- **Source Files on `mbp-2`**:
  - LaTeX source: `/Users/guillaume/Dropbox/2013 MICCAI/Latex/miccai_submission.tex`
  - Scene list: `/Users/guillaume/Dropbox/GYN Cases/MICCAI13scenes.txt`

## Dataset Split
- **Calibration**: 5 patient cases (used for tuning multi-scale elliptical Hough & Rayleigh PDF parameters; excluded from test set).
- **Phantom Validation**: 1 gel-wax phantom with 12 catheters (MR–MR max HD $1.5\text{ mm}$; MR–CT max HD $3.5\text{ mm}$).
- **Clinical Test Cohort**: **10 patients, 101 non-obturator interstitial catheters**:
  - `Case 024` (7), `Case 028` (14), `Case 029` (8), `Case 030` (15), `Case 031` (16), `Case 033` (5), `Case 034` (7), `Case 037` (6), `Case 038` (9), `Case 040` (14).

## Key Results
- **Interaction**: Semi-automatic (1 manual tip click per catheter).
- **Accuracy**: $93/101$ catheters ($92.1\%$) segmented with Maximum Hausdorff Distance $< 2.0\text{ mm}$ (median $0.58\text{ mm}$).
- **Runtime**: $0.3\text{ s}$ per catheter after tip selection.

## Related Pages
- [[wiki/research/historical-dataset-splits-2013-2026]]
- [[wiki/entities/jcmc-2017-mastmeyer-pernelle]]
- [[wiki/entities/pmb-2019-zaffino-pernelle-pmc6726393]]
