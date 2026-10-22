---
title: Bidirectional Seed Tracking & Kalman Smoothing
created: 2026-10-21
tags:
  - concept
  - tracking
  - kalman-filter
---

# Bidirectional Seed Tracking & Kalman Smoothing

## Overview
Classical semi-automatic needle tracking ([[wiki/entities/miccai-2013-pernelle|MICCAI 2013]]) requires the user to click the distal needle tip and tracks strictly in one direction (tip $\to$ base). In fully automatic detection (`autoDetectAllNeedles`), candidate seeds are discovered in the mid-zone of the volume where the catheter shaft contrast is highest.

## Algorithm Stages
1. **3D Local Maxima Seed Extraction**: Extracts top-scoring 3D peaks in $R_{3\text{D}}(z, y, x)$ within the middle axial zone ($15\%\text{–}85\%$ of slice range), enforcing a minimum 2D in-plane separation of $3.5\text{ mm}$ per slice.
2. **Bidirectional Propagation**:
   - **Forward Pass ($+z$, toward Needle Tip)**: Steps slice-by-slice from $z_{\text{seed}}$ toward $z_{\max}$, predicting the next $(x, y)$ position using linear velocity extrapolation over the last 4 slices, extracting a local ROI around the prediction, and computing a sub-pixel intensity/symmetry centroid.
   - **Backward Pass ($-z$, toward Obturator/Perineal Base)**: Steps from $z_{\text{seed}}$ toward $z_{\min}$ using the same predictive valley tracker.
3. **Adaptive Contrast Termination**: Tracking stops when local radial symmetry drops below $30\%$ of the trajectory's median contrast for `max_missed_slices` consecutive slices, or when a sudden lateral jump exceeds `max_step_mm`.
4. **4-State Constant-Velocity Kalman Smoother & Degree-3 Polynomial Fit**:
   - State vector $\mathbf{x}_k = [x_k, y_k, v_{x,k}, v_{y,k}]^\top$ filters slice-to-slice jitter.
   - A degree-3 polynomial $x(z), y(z)$ is fit to the smoothed trajectory to enforce physical bending stiffness of plastic catheters with tungsten stylets.

## Related Pages
- [[wiki/concepts/3d-directional-radial-symmetry-filter]]
- [[wiki/concepts/anatomical-z-zone-priors-and-spatial-deduplication]]
- [[wiki/research/2026-10-21-auto-needle-detection-evaluation]]
