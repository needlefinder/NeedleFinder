---
title: 2019 PMB (PMC6726393) Paper & Codebase Summary
created: 2026-10-21
tags:
  - source
  - pmb-2019
  - unet
---

# Source Summary: PMB 2019 (PMC6726393) & TensorFlow Codebase

- **PubMed Central**: https://pmc.ncbi.nlm.nih.gov/articles/PMC6726393/
- **Preprocessing Path on `mbp-2`**: `/Users/guillaume/Dropbox/2016-paolo/`
- **3D U-Net Codebase Path on `mbp-2`**: `/Users/guillaume/Dropbox/Projects/unet/`

## Extracted Facts from Code & Manuscript
1. **Preprocessing (`join_needles.py`)**:
   - Reads manual fiducial/model trajectories from 54 BWH cases (`002`–`078`).
   - Rasterizes each catheter as a $1.5\text{ mm}$ radius binary cylinder (`LabelMapsNEW2_1.00-1.00-1.00/`).
   - Resamples all T2W MRI volumes and labelmaps to **$1.0 \times 1.0 \times 1.0\text{ mm}^3$ isotropic spacing**.
2. **Network Architecture (`unet.py`, `3d-unet-tf-tiles-1class-multiple-gpus.py`)**:
   - 3D U-Net operating on $148 \times 148 \times 148$ voxel tiles.
   - Trained with Dice similarity coefficient loss across 4 GPUs.
   - Data augmentation: 3D rotations ($\pm 10^\circ$).
3. **Codebase Split Files (`/Users/guillaume/Dropbox/Projects/unet/`)**:
   - `train.txt`: 37 cases (`007, 009, 010, 012, 015, 017, 019, 021, 024, 025, 026, 027, 029, 030, 031, 033, 034, 037, 038, 040, 042, 046, 047, 049, 050, 052, 055, 056, 059, 060, 061, 062, 064, 065, 066, 067, 069`)
   - `valid.txt`: 7 cases (`070, 071, 072, 073, 074, 076, 077` $\times 3$ rotations)
   - `testing.txt`: 4 cases (`002, 028, 051, 054`)
   - Manuscript Experiment Split: 50 patients (826 catheters) split into **25 training, 3 validation, 22 testing**.
4. **Reported Metrics in PMB 2019**:
   - Voxel DSC: **$0.60 \pm 0.17$**
   - Average shaft distance error: **$2.0 \pm 3.4\text{ mm}$**
   - False positive rate: **$6.7\%$**
   - False negative rate: **$1.5\%$**
