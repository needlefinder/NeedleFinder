---
title: 3D Directional Radial Symmetry Filter
created: 2026-10-21
tags:
  - concept
  - algorithm
  - rust
  - cpp
---

# 3D Directional Radial Symmetry Filter

## Mathematical Formulation
In T2-weighted turbo spin-echo (TSE) pelvic MRI, plastic brachytherapy catheters appear on axial slices as small, dark circular/elliptical susceptibility voids ($1.5\text{–}2.5\text{ mm}$ diameter) surrounded by brighter tissue or template gel.

Unlike standard isotropic blob detectors (Laplacian of Gaussian or Frangi vesselness) which fire strongly on flat tissue boundaries (e.g., rectum/vaginal wall or bladder wall), `NeedleFinderLogic` computes an **anisotropic 8-direction ring-minus-center radial symmetry response** $R(z, y, x)$ in native physical coordinates $(s_x, s_y, s_z)$:

$$\Delta_d(z, y, x) = \frac{1}{|\mathcal{R}_d|} \sum_{r \in \mathcal{R}_d} I\bigl(z,\, y + \Delta y(r, \theta_d),\, x + \Delta x(r, \theta_d)\bigr) - \bar{I}_{\text{center}}(z, y, x)$$

for 8 compass directions $\theta_d = d \cdot \frac{\pi}{4}$ ($d \in \{0,\dots,7\}$) and physical ring radii $r \in [1.5\text{ mm}, 3.0\text{ mm}]$.

### Boundary-Suppression Clamp
To reject dark-to-bright step edges (where 4 directions are bright and 4 directions are dark), each directional contrast is clamped above zero and penalized if the minimum opposing directional contrast is negative:

$$R_{2\text{D}}(z, y, x) = \left(\frac{1}{8}\sum_{d=0}^{7} \max(0, \Delta_d)\right) \cdot \mathbb{I}\!\left(\text{count}(\Delta_d > 0) \ge 6\right)$$

### 3D Axial Persistence
Because true catheters traverse multiple contiguous axial slices along a smooth 3D tangent vector whereas random k-space noise and punctate vessels do not, $R_{2\text{D}}$ is accumulated across adjacent slices $(z-1, z, z+1)$ within a physical displacement cone $\Delta r_{\max} \approx 2.0\text{ mm}$:

$$R_{3\text{D}}(z, y, x) = \frac{1}{3}\left(R_{2\text{D}}(z, y, x) + \max_{\|\delta\| \le \Delta r} R_{2\text{D}}(z-1, y+\delta_y, x+\delta_x) + \max_{\|\delta\| \le \Delta r} R_{2\text{D}}(z+1, y+\delta_y, x+\delta_x)\right)$$

## Native Rust (`rayon`) and C++17 (`std::thread`) Acceleration
Implemented in `rust_core/src/lib.rs` (`nf_compute_3d_sym_response`) and `cpp_core/needlefinder_cpp.cpp`, parallelized across axial slices $z$. On `Case038` ($512 \times 512 \times 92$, 24.1M voxels), the native filter executes in **$183.7\text{ ms}$** vs. **$11,164.6\text{ ms}$** in NumPy/SciPy (**$60.8\times$ speedup**).

## Related Pages
- [[wiki/research/2026-10-21-auto-needle-detection-evaluation]]
- [[wiki/concepts/bidirectional-seed-tracking-and-kalman-smoothing]]
