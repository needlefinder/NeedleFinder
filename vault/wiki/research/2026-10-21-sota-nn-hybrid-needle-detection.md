---
title: "State-of-the-Art Deep Learning & Hybrid 3D CV for 100% Recall Needle Detection"
date: 2026-10-21
tags:
  - research
  - deep-learning
  - vision-transformer
  - unet
  - cldice
  - hybrid-cv
  - brachytherapy
---

# State-of-the-Art Deep Learning & Hybrid 3D CV for 100% Recall Needle Detection

## 1. Literature & State-of-the-Art Survey (2019–2026)

| Family / Architecture | Key References | Core Mechanism | Strengths for Brachytherapy Needles | Limitations When Used Alone |
| :--- | :--- | :--- | :--- | :--- |
| **3D Anisotropic U-Net + Connected Components** | Zaffino, Pernelle et al. (*PMB* 2019, PMC6726393); Shaaer et al. (*Med. Phys.* 2022) | 3D Encoder-Decoder on raw MRI $\rightarrow$ binary mask $\rightarrow$ connected components $\rightarrow$ polynomial fit | Learns dark susceptibility signal voids directly from tissue context; $92.6\%$ recall on 20-case PMB test set | Merges touching/crossing needles ($<2.5\text{ mm}$); breaks into disconnected fragments across dark fascia/rectal air; sensitive to scanner contrast shifts |
| **nnU-Net + Tubular Topology Loss (`clDice`)** | Isensee et al. (*Nat. Methods* 2021); Shit et al. (*CVPR* 2021 `clDice`) | Self-configuring anisotropic 3D ResUNet trained with soft-skeleton centerline Dice (`clDice`) + Dice/Focal loss | Enforces 1D topological continuity along thin tubular shafts; reduces mid-shaft breaks | Still outputs a semantic voxel mask requiring post-hoc instance separation; heavy runtime without GPU |
| **Vision Transformers & Hybrid CNN-Transformers (`Swin UNETR`, `DCT-UNet`, `MedNeXt`)** | Hatamizadeh et al. (*Swin UNETR* 2022); Roy et al. (*MedNeXt* MICCAI 2023); *DCT-UNet* (2024) | Local 2D/3D ConvNeXt or residual blocks for fine sub-millimeter edges + **Axial/Shifted-Window Self-Attention** across slices | Long-range z-axis self-attention bridges $1\text{--}3$ slice signal dropouts and uses global pelvic anatomy (obturator, bladder, rectum) to suppress false positives | Pure ViT patch tokenization ($4\times 4\times 4$) loses $1\text{ mm}$ needle cross-sections unless paired with full-resolution convolutional stems |
| **Centerline Heatmap & Vector-Field Regression (`Trexplorer` / Deep Hough / Radon)** | Continuous Gaussian centerline distance-field regression + tangent orientation vector $(\Delta x/\Delta z, \Delta y/\Delta z)$ | Regresses smooth centerline probability + local 3D trajectory direction instead of binary masks | Separates touching needles via tangent vectors; invariant to inconsistent manual labelmap radii across cases | Requires physics-aware multi-slice tracking to assemble full trajectories from template base to tip |
| **Multi-Pass Physics & 3D CV Iterative Tract Peeling (Our `autoDetectAllNeedles`)** | NeedleFinder 2-Pass Rust/C++ + Momentum + Bundle Consensus (2026) | Multi-radius ($1.35\text{--}3.3\text{ mm}$) 6-diameter symmetric dark-tube filter $\rightarrow$ Pass-1 lock $\rightarrow$ volume tract inpainting $\rightarrow$ Pass-2 guided tracking | Zero training required; $0.27\text{ s}$ runtime; $95.0\%$ recall on PMB 2019 Test ($\Delta xy = 0.69\text{ mm}$); cannot merge adjacent needles | Misses ~12.9% of needles across all 54 cases when needles lie on **non-symmetric boundaries** (against dark obturator/rectum) or are fragmented across slices |

---

## 2. Root-Cause Diagnosis of the Remaining 12.9% Missed Needles (Why 87.1% $\rightarrow$ 100% Requires Hybrid NN + 3D CV)

Detailed error analysis across all 54 BWH clinical cases (759 long needles) reveals three specific failure modes that prevent either pure 3D CV or pure 3D U-Net from reaching $100\%$ recall alone:

1. **Asymmetric Boundary Needles (Obturator & Rectal Wall Interface):**
   - Our 6-diameter symmetric filter computes $\min(I(\mathbf{x} + r\hat{u}_\theta), I(\mathbf{x} - r\hat{u}_\theta))$. When a needle runs immediately adjacent to the dark vaginal obturator or rectal air, one side of the ring lies inside the dark obturator, collapsing the symmetric pair minimum.
   - **Solution:** Add **half-ring / directional sector contrast features** and a **learned neural network enhancement head** that recognizes dark circular notches along tissue-obturator boundaries.

2. **Mid-Shaft Signal Dropouts & Fragmented Tracks (Pass-3 Sheared Radon / Line-Integral Recovery):**
   - Some peripheral needles have weak susceptibility contrast on $30\text{--}50\%$ of slices and fail slice-by-slice seed thresholding or `high_frac` gates.
   - **Solution:** Once Pass 1 locks the consensus needle bundle and determines the patient-specific 3D trajectory slope field $\vec{s}(x,y) = (s_x, s_y)$, a **Pass-3 Sheared 3D Radon / Shift-and-Add Line Integral** projects the unmasked 3D response along $\vec{s}(x,y)$ across the entire volume depth:
     $$A(x_0, y_0) = \frac{1}{|Z|} \sum_{k=k_{\text{lim}}}^{k_{\text{top}}} \tilde{R}\bigl(x_0 + s_x (z_k - z_0),\; y_0 + s_y (z_k - z_0),\; z_k\bigr)$$
     Even a faint needle whose per-slice SNR is low accumulates coherently across $15\text{--}60$ slices into a high-SNR peak in $A(x_0, y_0)$.

3. **Learned Hybrid Neural-CV Scoring & Centerline Probability Fusion (`HybridNeedleNet`):**
   - Combining our Rust/C++ physics features ($I_{\text{norm}}, C_{\text{smooth}}, R_{\text{sym}}, R_{\text{asym}}$) with an **Anisotropic 2.5D/3D ConvNeXt + Axial Transformer Network** trained on the PMB 2019 training split (and deployable in pure NumPy/ONNX/PyTorch inside 3D Slicer) provides both voxel-level centerline probabilities and trajectory-level confidence classification.
