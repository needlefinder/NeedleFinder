# NeedleFinder — Fast Interactive Needle Segmentation for 3D Slicer

![NeedleFinder Screenshot](screenshot.png)

**NeedleFinder** is a [3D Slicer](https://www.slicer.org/) extension for fast interactive and automated needle/catheter segmentation in 3D MR and CT volumes (originally developed for MR-guided interstitial gynecologic brachytherapy).

## Features

- **4 Automated Tracking Algorithms**:
  - **Kalman Filter (Algo 3 — Recommended)**: Fast 3D cone search with Kalman filtering and polynomial deflection modeling.
  - **Particle Filter (Algo 2)**: SIR particle filter tracking along the needle shaft.
  - **iGyne (Algo 1)**: Center-of-mass iterative slice tracking.
  - **Minimum-Cost Path (Algo 4)**: Multi-hypothesis minimum-cost path needle tracking.
- **Native Rust / C++17 Acceleration (`NeedleFinderRust`)**:
  - High-performance native kernels for 3D trilinear cone search, dichotomy tip localization, Bezier/Cardinal spline evaluation, and 2D center-of-mass tracking (~125x–240x faster than legacy Python/VTK).
- **Manual & Assisted Segmentation**:
  - Interactive control point placement, assisted centerline control-point estimation, and Hausdorff / tip-to-tip validation against ground-truth needles.

---

## Installation & Setup in 3D Slicer (5.x)

### Option 1: Install from the Slicer Extension Manager
1. Open **3D Slicer** (5.10 / 5.12+).
2. Go to **View → Extension Manager** (`Ctrl+4` / `Cmd+4`).
3. Search for **NeedleFinder** under the **IGT** category and click **Install**.
4. Restart Slicer and select **IGT → NeedleFinder** from the module selector.

### Option 2: Load Directly from Source (Developer Mode)
1. Clone this repository:
   ```bash
   git clone https://github.com/needlefinder/NeedleFinder.git
   ```
2. Launch Slicer with `--additional-module-paths`:
   ```bash
   Slicer --additional-module-paths /path/to/NeedleFinder/NeedleFinder
   ```
   Or in Slicer, open **Edit → Application Settings → Modules**, add `/path/to/NeedleFinder/NeedleFinder` to **Additional module paths**, and restart Slicer.

---

## Quick Start Workflow

1. **Load a Volume**: Load your 3D MR or CT dataset (`File → Add Data`).
2. **Select Axial Limit (Step 1)**: Scroll the Red (Axial) slice view to the superior limit of the needles and click **`1. Select Current Axial Slice as Segmentation Limit`** (or click **`Skip Limit`** to segment across the full volume).
3. **Seed Needle Tips (Step 2)**: Click **`2. Start Giving Needle Seed Points`** (or press `Ctrl+Return`), then click on a needle in the axial slice view to automatically track and render its 3D trajectory.
4. **Manual / Assisted Segmentation**: Use **`Start Giving Control Points`** or **`Assisted Manual Segmentation`** in section **2. Manual Needle Segmentation & Validation** to create or refine manual reference needles and compute Hausdorff / tip distances.

---

## Citation

If you use NeedleFinder in your research, please cite:

- Pernelle G., Mehrtash A., Barber L., Damato A., Wang W., Seethamraju R.T., Schmidt E., Cormack R.A., Wells W., Viswanathan A., Kapur T. *"Validation of Catheter Segmentation for MR-Guided Gynecologic Cancer Brachytherapy."* MICCAI 2013, LNCS vol. 8151, pp. 380–387 (2013).
- Mastmeyer A., Pernelle G., et al. *"Accurate Model-based Segmentation of Gynecologic Brachytherapy Catheter Collections in MRI-Images."* Medical Image Analysis (2017).
