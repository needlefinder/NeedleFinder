---
title: Anatomical Z-Zone Priors & Spatial Deduplication
created: 2026-10-21
tags:
  - concept
  - nms
  - priors
---

# Anatomical Z-Zone Priors & Spatial Deduplication

## 1. Anatomical Z-Zone & Kinematic Priors
To separate real brachytherapy catheters from pelvic blood vessels, bowel gas, and template edge artifacts without deep learning training data, each candidate 3D trajectory $\mathcal{T}_i = \{(x_z, y_z, z)\}_{z=z_{\text{base}}}^{z_{\text{tip}}}$ is filtered and scored by physical priors:
- **Minimum 3D Length ($L_{\min}$)** and **Axial Z-Span ($\Delta Z_{\min}$)**: Rejects short punctate voids.
- **Polynomial Straightness Residual ($\sigma_{\text{poly}}$)**: Plastic catheters deflected by tissue follow smooth cubic curves ($\sigma_{\text{poly}} < 0.8\text{ mm}$), whereas tortuous iliac/uterine vessels exhibit high cubic residuals.
- **Insertion Depth Prior ($z_{\text{base}}$)**: Interstitial needles enter from the inferior/caudal perineal template ($z_{\text{base}}$ in the lower $55\%$ of the FOV).
- **Maximum Tilt Angle ($\theta_{\max} \le 35^\circ$ from the cranio-caudal Z-axis)**: Needles inserted through the Syed-Neblett template are nearly parallel to the scanner bore axis.

## 2. Confidence Scoring
Each valid trajectory receives a composite confidence score $C(\mathcal{T}_i) \in [0, 1]$ combining:
- Normalized mean 3D radial symmetry response along the shaft ($45\%$),
- Normalized trajectory length ($25\%$),
- Straightness factor $\exp(-\sigma_{\text{poly}} / 1.5\text{ mm})$ ($15\%$),
- Caudal base entry bonus ($15\%$).

## 3. Greedy 3D Trajectory Non-Maximum Suppression (NMS)
Multiple seeds along the same catheter shaft generate overlapping trajectories. Trajectories are sorted in descending order of $C(\mathcal{T}_i)$ and greedily accepted if their mean shared-slice 2D Euclidean distance to all previously accepted trajectories exceeds $d_{\text{NMS}} = 3.5\text{ mm}$ (or if their overlap fraction is $<30\%$).

## Related Pages
- [[wiki/concepts/3d-directional-radial-symmetry-filter]]
- [[wiki/concepts/bidirectional-seed-tracking-and-kalman-smoothing]]
- [[wiki/concepts/short-obturator-vs-long-interstitial-needles]]
