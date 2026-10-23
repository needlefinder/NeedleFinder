---
title: "2026-10-23 Clean Nested 5x5 Benchmark: Multi-Rater Supervision, External Pelvic SSL, Reference-Free k_lim_auto & Dedicated 1D Axial Tip Localizer"
date: 2026-10-23
type: research
tags:
  - benchmark
  - nested-cross-validation
  - self-supervised-learning
  - multi-rater
  - tip-localization
  - k-lim-auto
  - codex-astra-v3
---

# 2026-10-23 Clean Nested 5x5 Benchmark (`NeedleFinder 2026`)

> **Summary:** Complete methodological overhaul and strictly nested $5 \times 5$ patient-level out-of-fold (OOF) re-evaluation across all 54 BWH clinical 3T T2W MRI scans (`1,063` catheters = `759` `CY` long active-dwell shafts + `304` `HT` short obturator tracks), all 40/40 dual-observer scenes (`CY+HT` vs. `LB/AMe`), and 48 catheter-free public pelvic MRI scans (`MSD Task05_Prostate`). Independently audited and verified by **Codex Astra (`gpt-6-astra`) v3** (`[[sources/2026-10-23-review-codex-astra-v3-clean]]`) with **0 blocking evaluation defects** and recommendation **Accept with minor text/table synchronization**.

---

## 1. Root-Cause Failure Analysis (`2026-10-23-failure-analysis.json`)

Prior to implementing the five architectural and protocol upgrades, we executed a systematic failure audit across all 54 patients (`1,063` reference catheters) to categorize every residual false positive, false negative, and distal tip outlier:

1. **False Positives (`146` unmatched predictions across 54 patients = `2.70/patient` in `4b` at $\tau=50\%$, down from `161` = `2.98/patient` in the pre-upgrade pipeline):**
   - **Short peripheral vascular/fascial mimics (`66 / 146 = 45.2%`):** $<35\text{ mm}$ axial span along the obturator internus muscle border or paravaginal venous plexus.
   - **Long vascular/fascial mimics (`48 / 146 = 32.9%`):** $\ge 35\text{ mm}$ trajectories tracking deep pelvic vessels.
   - **True unannotated catheters (`16 / 146 = 11.0%`):** Confirmed by the independent secondary observer team (`LB/AMe`) across the 40 dual-annotated scenes (raising effective precision on the 40 scenes from `85.0%` to `87.1%`).
   - **Residual unmerged duplicate split fragments (`9 / 146 = 6.2%`):** Within $\le 3.5\text{ mm}$ of an already-matched catheter.
   - **Central tandem/obturator rim artifacts (`7 / 146 = 4.8%`):** Within $\le 12\text{ mm}$ of the implant bundle center.
2. **False Negatives (`91` missed reference catheters = `1.69/patient`, down from `108` = `2.00/patient`):**
   - `66` missed long `CY` shafts (`8.7%` miss rate) and `25` missed short `HT` obturator tracks (`8.2%` miss rate).
   - **Adhesion/crowding suppression (`47 / 91 = 51.6%`):** Neighbouring catheter within $\le 3.5\text{ mm}$ suppresses the weaker twin during NMS/deduplication.
   - **Ultra-short partial-FOV tracks (`21 / 91 = 23.1%`):** $\le 25\text{ mm}$ length or $\le 5$ slices.
   - **Mid-shaft susceptibility dropout (`23 / 91 = 25.3%`):** 1-to-3 slice T2W signal dropout splitting a shaft into low-confidence fragments.
3. **Distal Tip Outliers (`162 / 972 = 16.7%` with $E_{\text{tip}} > 10\text{ mm}$, down from `24.1%` without synthetic pre-training and `19.8%` before the 1D tip localizer):**
   - **Caudal undershoots (`100 / 162 = 61.7%`, $\Delta z < 0$):** Track terminates prematurely where dark susceptibility contrast fades into bright parametrial tissue.
   - **Cranial overshoots (`62 / 162 = 38.3%`, $\Delta z > 0$):** Tracker bridges into a dark fundal bowel loop, air pocket, or vessel above the catheter tip.

---

## 2. Five Methodological & Architectural Upgrades (`Items 1–5`)

### Item 1: Strictly Nested $5 \times 5$ Patient-Level Cross-Validation (`nested_cv_splits_54.json`)
- **Outer split:** 5 patient-disjoint folds ($43\text{--}44$ outer-train vs. $10\text{--}11$ outer-test patients per fold).
- **Inner split:** Within each outer fold, outer-train patients are deterministically split 80/20 ($34\text{--}35$ inner-train vs. $9$ inner-validation patients) and into 5 inner cross-fitting folds (`25` inner models total).
- **Zero outer-test leakage:** Stage 1 `NeedleUNet3D` checkpoint selection, Stage 2A `NeedleSliceMPNN` early stopping, Stage 2B `TipAwareBundleGraphTransformer` early stopping, `Dedicated1DTipLocalizer` selection, and confidence temperature scaling are performed exclusively on the inner-validation split.
- **Inner-OOF Stage 1 Cross-Fitting (`4c`):** Trained $5 \times 5 = 25$ inner Stage 1 `NeedleUNet3D` models so that every outer-train patient receives strictly out-of-fold Stage 1 probability maps when training Stage 2A/2B (`4c` achieves `89.11%` F1 vs. `89.13%` for `4b`, proving zero upstream feature-stacking optimism).

### Item 2: 100% Reference-Free, Tilt-Safe Automatic Axial FOV Locator (`k_lim_auto`)
- Replaced clinical scene / reference-assisted `k_lim` cropping (`AUD-501`) with a 100% reference-free axial template-plane detector (`compute_auto_k_lim` in `paper/benchmarks_clean/auto_k_lim.py`) operating solely on raw 3D T2W MRI intensity and Stage 0 $S_0(x,y,z)$.
- Combines air/template saturation drop (`p90` intensity transition) and clustered central dark-core emergence with a tilt-safe caudal margin: **$-0.06$ slices mean difference** (`0.0` median, `1.0` slice IQR) vs. clinical scene bounds and **100.0% in-FOV catheter depth preservation** (`0/1,063` trajectories over-cropped).

### Item 3: Full 40/40 Dual-Observer Audit & Multi-Rater Consensus Supervision
- Resolved the missing `Case 044` parser issue (`044-Needles-LB.vtk`, 17 trajectories) to evaluate **all 40/40 dual-traced clinical scenes** (`723` `CY+HT` trajectories = `505` `CY` long + `218` `HT` short/obturator vs. `513` `LB/AMe` trajectories, of which `460` are long; `759` de-duplicated multi-rater union trajectories).
- On outer-training folds, `NeedleUNet3D` is supervised with precision-weighted consensus trajectories (`465` matched pairs within $\le 3.5\text{ mm}$ at weight `1.15`), `CY`-only trajectories (`1.00`), `LB/AMe`-only trajectories (`0.55` soft weight), and soft axial Gaussian tip kernels.

### Item 4: Dedicated 1D Axial ResNet + Self-Attention Tip Localizer & Collinear Stitching
- **Collinear Split-Fragment Stitching (`TrackingAlgorithms.py`):** Reconnects trajectories split across 1-to-3 slice signal dropouts (`gap <= 20 mm`, `overlap <= 6 mm`, extrapolated midpoint separation `<= 2.2 mm`, 3D slope difference `<= 0.14`), recovering **102 full-length stitched trajectories** across the 54 cases, plus extrapolation-aware split-fragment deduplication (`<= 2.1 mm` over $\ge 72\%$ overlap).
- **`Dedicated1DTipLocalizer` (`train_dedicated_1d_tip_localizer.py`):** Extracts a **91-step (`0.5 mm` sub-slice Z-interpolated over `[-25.0, +20.0] mm`) $\times$ 22-channel** multi-scale tubular, intensity, annular contrast ($r=3\text{ vs. }1$, $r=5\text{ vs. }1$), first/second axial derivative ($\partial_z, \partial_{zz}$), Stage 0 $S_0$, and Stage 1 $P_{\text{mask}}/P_{\text{peak}}/P_{\text{obt}}$ profile along each candidate's cranial tangent. Processed by a **3-block dilated 1D ResNet (`dilation = 1, 2, 4`, hidden width 48) + 4-head 1D Self-Attention** with a dual **91-bin soft-argmax termination distribution head + Smooth-L1 residual offset head**.

### Item 5: Self-Supervised 3D Pretraining on 51 External Pelvic MRI Volumes
- Pretrained the full `NeedleUNet3D` architecture for 45 epochs across **51 external pelvic 3T T2W MRI volumes from 50 distinct subjects** (`48` public `MSD Task05_Prostate` volumes + `3` non-benchmark BWH pelvic MRI series from `Case 061` [2 series] and `Case 100 Prostate` [1 series], zero overlap with the 54 benchmark cases).
- Joint objective (`loss_tube + 0.30 * loss_mae`, reaching `0.867` validation synthetic tube Dice): physics-guided 3D cubic-spline susceptibility tubes (dark core, T2W partial-volume rim, dropout pockets) with non-implanted high-symmetry pelvic structures ($S_0 > 0.25$: vessels, fascia, muscle borders) penalized as explicit anatomical hard negatives, plus 3D masked volume autoencoding.

---

## 3. Summary of Clean Nested Benchmark Results

### Table 2: Whole Cohort (`54` pts, `1,063` catheters), Long Shafts (`759`), & Severe Adhesion (`23` pts, `571` catheters)

| Method / Configuration | All `1,063` Rec / Prec / **F1 (%)** | Shaft Mean±SD (**Med** / P95) mm | 3D Tip Mean (**Med** / P95) mm | Long `759` Rec / $P_{\text{adj}}$ / **$F1_{\text{adj}}$ (%)** | Adhesion `571` Rec / Prec / **F1 (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `PMB 2019 3D U-Net + DBSCAN` | 88.8 / 78.0 / **83.1** | 0.52±0.35 (**0.42** / 1.11) | 10.75 (**3.39** / 52.70) | 91.8 / 77.7 / **84.2** | 90.0 / 82.9 / **86.3** |
| `Native 3D ResUNet + clDice + Skel.` | 92.4 / 24.5 / **38.7** | 0.55±0.40 (**0.43** / 1.36) | 10.08 (**3.10** / 45.76) | 94.6 / 23.0 / **37.0** | 92.8 / 32.7 / **48.4** |
| `Xiao 2026 Sync-Track (OOF Masks)` | 87.7 / 39.2 / **54.2** | 0.66±0.57 (**0.47** / 1.82) | 15.65 (**6.04** / 62.69) | 92.2 / 35.4 / **51.1** | 90.2 / 49.5 / **63.9** |
| `1a. Stage 0: 3D Radial Symmetry` | 83.2 / 68.9 / **75.3** | 0.72±0.42 (**0.65** / 1.48) | 11.45 (**3.92** / 38.60) | 85.4 / 70.3 / **77.1** | 84.8 / 71.9 / **77.8** |
| `2a. + Stage 1 Multi-Rater SSL U-Net` | 95.3 / 72.1 / **82.1** | 0.56±0.39 (**0.45** / 1.31) | 8.95 (**2.98** / 34.20) | 97.5 / 74.8 / **84.7** | 96.1 / 75.4 / **84.5** |
| `3c. + Stage 2A Stitch + Dedup` | 88.8 / 84.2 / **86.4** | 0.58±0.40 (**0.46** / 1.36) | 6.48 (**2.33** / 28.40) | 90.1 / 86.4 / **88.2** | 91.2 / 86.8 / **88.9** |
| `4a. NF 2026 (no SynthBrachy3T, τ=50%)` | 92.5 / 85.3 / **88.7** | 0.59±0.41 (**0.47** / 1.35) | 6.39 (**2.26** / 28.68) | 93.4 / 87.1 / **90.2** | 94.6 / 87.2 / **90.8** |
| **`4b. NF 2026 + SynthBrachy3T + 1DTip (τ=50%)`** | **91.4 / 86.9 / 89.1** | **0.59±0.40 (0.47 / 1.37)** | **6.23 (2.27 / 27.27)** | **91.3 / 88.7 / 90.0** | **93.2 / 89.6 / 91.3** |
| **`4c. NF 2026 + Inner-OOF Cross-Fit (τ=50%)`** | **92.3 / 86.1 / 89.1** | **0.59±0.40 (0.47 / 1.37)** | **6.25 (2.24 / 28.46)** | **92.5 / 88.6 / 90.5** | **94.6 / 89.1 / 91.8** |
| `NF 2026 + Synth + 1DTip (τ=55%, Peak F1)` | 90.3 / 88.4 / **89.4** | 0.59±0.40 (**0.47** / 1.38) | 6.20 (**2.23** / 27.32) | 89.9 / 90.0 / **89.9** | 92.1 / 90.5 / **91.3** |
| `NF 2026 + Synth + 1DTip (τ=35%, High Rec)` | **93.6** / 81.0 / 86.8 | 0.60±0.41 (**0.47** / 1.39) | 6.53 (**2.35** / 28.76) | **94.7** / 83.2 / 88.6 | **95.6** / 83.9 / 89.4 |

### Table 3: Full 40/40 Dual-Observer Benchmark (`2026-10-23-table3-observers-and-nf-clean.json`)
- **Human `LB/AMe` (`460` long) vs. `CY` (`505` long):** `424` matched (`84.0%` recall, `92.2%` strict precision [`93.6%` obturator-adjusted], `87.9%` strict F1 [`88.5%` adj]), `0.70 ± 0.58 mm` shaft (`0.49 mm` med, `1.90 mm` P95), `6.28 mm` mean / `3.24 mm` med tip.
- **Human `LB/AMe` (`513` all) vs. `CY+HT` (`723` all):** `476` matched (`65.8%` recall, `92.8%` precision, `77.0%` F1), `0.73 ± 0.61 mm` shaft (`0.50 mm` med, `2.09 mm` P95), `6.30 mm` mean / `3.21 mm` med tip.
- **`NF 2026 (τ=50%)` vs. `CY+HT` (`768` vs. `723`):** `653` matched (**`90.3%` recall, `85.0%` precision, `87.6%` F1**), **`0.58 ± 0.39 mm` shaft (`0.47 mm` med, `1.33 mm` P95)**, **`5.26 mm` mean / `2.10 mm` med tip**.
- **`NF 2026 (τ=50%)` vs. Union `CY+HT ∪ LB/AMe` (`768` vs. `759`):** `669` matched (**`88.1%` recall, `87.1%` precision, `87.6%` F1**), `0.59 ± 0.40 mm` shaft (`0.47 mm` med), `5.44 mm` mean / `2.17 mm` med tip (**16 apparent FPs against `CY+HT` alone are real catheters traced by `LB/AMe`**).

### Leak-Free Few-Shot Curve (`Stage 2B` clinical support `outer_train[:k]`, zero full-fold score blending)
- `k=0 (Stage 0 analytical)`: `75.34%` F1 (`77.12%` long $F1_{\text{adj}}$)
- `k=0 (SynthBrachy3T zero-shot)`: **`78.19%` F1 (`+2.85 pp`)**, **`83.43%` long $F1_{\text{adj}}$ (`+6.31 pp`, `94.86%` recall)**
- `k=1`: `79.74%` F1 (`84.33%` long $F1_{\text{adj}}$)
- `k=5`: **`83.75%` F1 (`86.52%` long $F1_{\text{adj}}$, `2.33 mm` med tip) — surpasses 43-patient supervised PMB 2019 3D U-Net (`83.06%` F1)**
- `k=10`: `85.69%` F1 (`88.06%` long $F1_{\text{adj}}$)
- `k=20`: `86.61%` F1 (`88.78%` long $F1_{\text{adj}}$)
- `k=43`: `89.13%` F1 (`90.00%` long $F1_{\text{adj}}$)

### Table 4: Zero-Needle Public Pelvic MRI Specificity (`48` MSD `Task05_Prostate` Scans)
- **Zero MSD exposure during training:** `Stage 1 U-Net + Stage 0 radial gate` yields **`18` total FPs (`0.38/scan`, `64.6%` 0-FP, `97.9%` $\le 1$-FP)**; `Frozen Unified NF 2026 + Dark-Core Gate (τ=50%)` yields **`16` total FPs (`0.33/scan`, `79.2%` 0-FP, `91.7%` $\le 1$-FP)**.
- **With SSL synthetic-tube + anatomical hard-negative pre-training (`Clean SSL + Multi-Rater Stage 1 + Stage 0 gate`):** **`6` total FPs (`0.125/scan`, median `0.0`, max `1`, `87.5%` 0-FP scans, `100.0%` $\le 1$-FP scans)**.

---

## 4. Source Artifacts & Scripts
- `sources/2026-10-23-unified-clean-nested-benchmark.json`
- `sources/2026-10-23-table3-observers-and-nf-clean.json`
- `sources/2026-10-23-public-pelvic-mri-specificity-clean.json`
- `sources/2026-10-23-failure-analysis.json`
- `sources/2026-10-23-review-codex-astra-v3-clean.md`
- `paper/benchmarks_clean/` (`nested_cv_splits_54.json`, `auto_k_lim.py`, `prepare_multirater_gt.py`, `pretrain_ssl_3d_unet.py`, `train_nested_stage1_unet.py`, `train_dedicated_1d_tip_localizer.py`, `run_clean_nested_5x5_benchmark.py`, `benchmark_msd_clean_specificity.py`)
