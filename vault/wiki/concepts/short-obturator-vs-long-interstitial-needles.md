---
title: Short Obturator vs. Long Interstitial Needles
created: 2026-10-21
tags:
  - concept
  - clinical
  - brachytherapy
---

# Short Obturator vs. Long Interstitial Needles

## Clinical Distinction
In Syed-Neblett template-guided gynecologic brachytherapy at Brigham & Women's Hospital:
1. **Long Interstitial Needles ($240\text{ mm}$ length)**: Inserted through the outer rings of the perineal template deep into the parametrium/cervix. In the MRI field of view, they typically span **$65\text{–}135\text{ mm}$** across $35\text{–}65$ axial slices.
2. **Short Vaginal Obturator Needles ($\sim 70\text{–}90\text{ mm}$ length)**: Inserted through the central grooves of the plastic vaginal obturator (typically 6 needles arranged radially around the central tandem/obturator axis) to cover the vaginal cuff and lower cervix. In the MRI field of view, they span **$25\text{–}55\text{ mm}$** and terminate near the upper end of the vaginal obturator.

## Impact on Historical Datasets & Evaluations
- **MICCAI 2013 & `760CYScenes.txt`**: Only the **long interstitial needles** were annotated in the primary `.mrml` scenes (e.g., 15 long needles in `Case030`, 9 long needles in `Case038`, 20 long needles in `Case 050` / `2016-RSNA`).
- **Separate Obturator Ground Truth**: The short obturator needles were annotated separately in `/Users/guillaume/Dropbox/GYN Cases/short needle segment data of 50 cases.xlsx` and in `Manual-shortOneedle/` folders (e.g., 6 short needles in `Case 050` and 6 short needles `#10`–`#15` in `Case 038`).
- **Discovery in 2026 Evaluation**: When running `autoDetectAllNeedles` on `Case030` (where only 15 long needles were in the MRML scene), the detector found 14/15 long needles plus 5 additional trajectories in the central obturator ring ($X \in [-6, +12]\text{ mm}, Y \in [-24, -4]\text{ mm}$). Cross-checking `short needle segment data of 50 cases.xlsx` confirmed that `Case030` actually contains **4 physical short obturator needles** that our zero-shot algorithm detected automatically.

## Related Pages
- [[wiki/research/historical-dataset-splits-2013-2026]]
- [[wiki/research/2026-10-21-auto-needle-detection-evaluation]]
