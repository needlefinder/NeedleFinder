---
title: RSNA 2016 Dataset (Case 050)
created: 2026-10-21
tags:
  - entity
  - dataset
  - rsna
---

# RSNA 2016 Dataset (`Case 050`)

## Overview
- **Source Case**: Brigham & Women's Hospital `Case 050`
- **Paths**:
  - Local (`edelweiss`): `/mnt/SSD/github/needlefinder/2016-RSNA/`
  - `mbp-2`: `/Users/guillaume/Dropbox/2016-RSNA/` and `/Users/guillaume/Dropbox/GYN Cases/Case  050/`
- **Volume Dimensions & Spacing**: $320 \times 320 \times 78$ voxels, $0.4688 \times 0.4688 \times 2.0\text{ mm}^3$ ($150 \times 150 \times 156\text{ mm}$ FOV).
- **Ground Truth Annotations**:
  - **20 long interstitial needles** in `2016-RSNA.mrml` (`vtkMRMLMarkupsFiducialNode1`–`21`, excluding empty `#14`).
  - **6 short vaginal obturator needles** in `/Users/guillaume/Dropbox/GYN Cases/Case  050/NRRD/Manual-shortOneedle/`.

## Role in 2026 Benchmark
Used as the **single calibration/development case** for setting default dimensionless multipliers in `NeedleFinderLogic.autoDetectAllNeedles`. Achieves **$95.0\%$ recall ($19/20$)**, **$100\%$ precision ($19/19$)**, **$1.06 \pm 0.96\text{ mm}$ mean shaft error**, and **$0.39\text{ s}$ total runtime** ($19.4\text{ ms}$ Rust filter).

## Related Pages
- [[wiki/research/2026-10-21-auto-needle-detection-evaluation]]
- [[wiki/research/historical-dataset-splits-2013-2026]]
