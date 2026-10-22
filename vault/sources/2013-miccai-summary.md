---
title: 2013 MICCAI Paper Summary
created: 2026-10-21
tags:
  - source
  - miccai-2013
---

# Source Summary: MICCAI 2013 (`miccai_submission.tex`)

- **Path on `mbp-2`**: `/Users/guillaume/Dropbox/2013 MICCAI/Latex/miccai_submission.tex`
- **Scene Index on `mbp-2`**: `/Users/guillaume/Dropbox/GYN Cases/MICCAI13scenes.txt`

## Extracted Facts
1. **Clinical Problem**: Intraoperative 3T MRI catheter segmentation for interstitial gynecologic brachytherapy in the AMIGO suite.
2. **Method**:
   - Clinician selects needle tip on axial T2W MRI.
   - Search cone initialized toward the template base.
   - Multi-scale 2D elliptical Hough transform + Rayleigh PDF intensity weighting evaluates candidate cross-sections on each slice.
   - 3D cubic Bézier curve smoothed across slices.
3. **Evaluation**:
   - **Calibration**: 5 patient datasets.
   - **Phantom**: 1 gel-wax phantom, 12 catheters (MR–MR HD $\le 1.5\text{ mm}$, MR–CT HD $\le 3.5\text{ mm}$).
   - **Patients**: 10 clinical cases (`024, 028, 029, 030, 031, 033, 034, 037, 038, 040`), 101 non-obturator catheters.
   - **Results**: $93/101$ ($92.1\%$) catheters had Max Hausdorff Distance $< 2.0\text{ mm}$ compared to expert manual segmentation; median HD $0.58\text{ mm}$; $0.3\text{ s}$ per catheter.
