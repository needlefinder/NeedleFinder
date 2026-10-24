---
title: "2026-10-24 Medical Image Analysis (MedIA) Venue Upgrade, True 3D Figure Suite & Appendix Restructuring Plan"
date: 2026-10-24
type: plan
status: active
tags:
  - paper
  - media
  - 3d-visualization
  - appendix
  - brachytherapy
  - geometric-deep-learning
  - image-guided-therapy
---

# 2026-10-24 *Medical Image Analysis* (*MedIA*) Venue Upgrade, True 3D Figure Suite & Appendix Restructuring Plan

## 1. Executive Summary & Target Venue Decision

We are upgrading the target venue of **NeedleFinder 2026** from *Physics in Medicine & Biology* (*PMB*) to ***Medical Image Analysis* (*MedIA*, Elsevier, Impact Factor ~10.7–11.8)**.

### 1.1 Venue Comparison & Rationale

| Venue | Impact Factor | Lineage & Fit | Strengths | Trade-offs |
| :--- | :---: | :--- | :--- | :--- |
| **1. *Medical Image Analysis* (*MedIA*, Elsevier)** | **~10.7–11.8** | **Direct sequel** to our group's **Mastmeyer, Pernelle *et al.* 2017 (*MedIA* 42:173–188)** on the BWH AMIGO gynecologic cohort | **Selected Target.** Premier journal for physics-informed geometric deep learning, multi-rater Grubbs adjudication, split-conformal uncertainty, dosimetric validation, and open-source 3D Slicer translation. Supports in-paper Appendices (`Appendix A–D`) and Graphical Abstract + Highlights. | ~3–4 month first-decision cycle. |
| **2. *IEEE Transactions on Medical Imaging* (*IEEE TMI*)** | **~8.9–10.6** | Algorithmic medical imaging flagship | Strong visibility in ML/computer vision; values K-space susceptibility simulation (`SynthBrachy3T`) and graph transformers. | Strict 10–12 page double-column limit forces most clinical translation into a separate supplementary PDF. |
| **3. *Physics in Medicine & Biology* (*PMB*, IOP)** | **~3.3–3.6** | Direct sequel to **Zaffino, Pernelle *et al.* 2019 (*PMB* 64:165008)** | Fast turnaround (~6–8 weeks); read by clinical brachytherapy physicists. | Lower visibility in the broader computer vision and surgical AI communities. |
| **4. *Medical Physics* (AAPM / Wiley)** | **~3.2–3.8** | Home of reproduced baselines (`Dai 2020`, `Aleong 2024`) | Core North American radiation-oncology physics readership. | Less emphasis on geometric deep learning and relational graph architectures. |
| **5. *Nature Biomedical Engineering* / *Nat. Commun.*** | **~26 / ~14.7** | High-impact translational / multidisciplinary | Broadest scientific reach. | Typically requires prospective multi-institutional clinical implant validation or live operating-room timing trials. |

---

## 2. How NeedleFinder 2026 Answers Open Challenges Across Three Fields

Positioning the manuscript for *Medical Image Analysis* requires foregrounding how a single unified architecture resolves tightly coupled challenges across **Brachytherapy & Radiation Oncology**, **Computer Vision & Geometric Deep Learning**, and **Image-Guided Therapy (IGT)**:

### 2.1 Brachytherapy & Radiation Oncology
1. **Eliminating the 45–50 Minute MR-Only Planning Bottleneck:** Manual slice-by-slice catheter digitization while the patient remains under anesthesia is the primary operational barrier to routine MR-only interstitial gynecologic HDR brachytherapy (EMBRACE-II). NeedleFinder 2026 compresses full-bundle 3D reconstruction to **`4.15 s` on a standard workstation CPU (`2.67 s` GPU)**.
2. **Resolving the 3T T2W Tip-Depth & Dosimetric Safety Paradox:** On T2W MRI, `98.8%` of squared 3D tip error lies longitudinally along the catheter shaft, causing a `4.24%` median (`8.21%` mean) target dose error $|\Delta D_{90}|$ if dwell positions are optimized blindly on unreviewed automated tips. We prove that **split-conformal tip uncertainty ($\hat{\sigma}_{\text{tip}}$, `90.2%` empirical coverage at 90% nominal)** converts this physical MRI limit into a **modeled `18 s` clinician-in-the-loop verification**: checking only the **top 2 flagged tips per patient** reduces median $|\Delta D_{90}|$ to **`2.83%` ($p = 2.4 \times 10^{-5}$)**, matching human inter-observer dosimetric agreement (`2.12%`).

### 2.2 Computer Vision & Geometric Deep Learning
1. **Preventing Topology Collapse in Dense, Touching 3D Curvilinear Bundles:** Voxelwise 3D CNNs (`3D U-Net`, `ResUNet + clDice`, `3D Attention U-Net`) merge touching catheters (`<3 mm` clearance) into single blobs and fragment shafts across signal dropouts (`38.7%–69.5%` whole-cohort F1 on native T2W MRI). Our **multi-representation progression**—Stage 0 analytical circular dark-core symmetry $\to$ Stage 1 sub-pixel Gaussian peak & 3D tangent regression $\to$ Stage 2A triplet message-passing graph & collinear gap stitcher $\to$ Stage 2B bundle-level graph transformer—achieves **`91.3%–91.8%` F1 in the 23-patient severe catheter-adhesion subgroup** (`+9.6` to `+43.4` F1 points over peer-reviewed baselines).
2. **Learning Without Large Annotated Cohorts (`SynthBrachy3T` + Pelvic SSL):** Annotated interstitial brachytherapy cohorts are small worldwide. By pre-training Stage 1 on **51 external pelvic MRI scans (50 needle-free)** where real pelvic veins and fascia act as hard negatives against synthetic catheter voids, and pre-training Stage 2B on **`SynthBrachy3T` (`83.4%` zero-shot long-shaft $F1_{\text{adj}}$ at $k=0$)**, fine-tuning on just **$k = 5$ clinical patients (`83.8%` F1)** already matches a 43-patient fully supervised 3D U-Net (`83.1%` F1).
3. **Reference-Free Evaluation Beyond Noisy Single-Rater Labels:** Using **Grubbs' (1948) three-cornered-hat estimator** on $n = 432$ three-way matched catheters across 40 dual-traced scans and **paired post-implant Pelvic CT (`38` paired patients)**, we decouple intrinsic algorithm error (`0.39 mm` MAD shaft SD) from human observer jitter (`0.35 mm` Team 1, `0.48 mm` Team 2) and show that **`52.0%` of apparent "false alarms"** against a single human team are physical catheters confirmed on CT.

### 2.3 Image-Guided Therapy (IGT) & Surgical AI Translation
1. **Auditable Reason Codes & 1-Click Candidate Restoration in 3D Slicer:** Clinicians distrust black-box AI filters that silently discard trajectories. NeedleFinder 2026 runs in **3D Slicer** on a CPU without a GPU, tags every candidate with one of **5 deterministic reason codes**, renders **calibrated 3D conformal tip spheres**, and supports **1-click candidate restoration**.
2. **In-Vivo 4D Needle–Tissue Biomechanics:** Across 23 serial intra-procedural 3T MRI scans (`Cases 68–71`), coupling NeedleFinder 2026 with a causal 4D Kalman filter (`1.31 s`/scan) boosts precision by `+12.4` points (`76.0% -> 88.4%`) and quantifies in-vivo lateral "tissue push" (`1.44 mm` median shift within $\le 10\text{ mm}$ of a newly inserted needle vs. `1.00 mm` beyond $20\text{ mm}$, $p = 9.97 \times 10^{-4}$) and non-linear depth deflection (`3.29 mm` median beyond $90\text{ mm}$ depth).

---

## 3. True 3D Anatomical & Geometric Figure Upgrades

Following best practices in scientific visualization, statistical charts (`Figs. 3, 4, 6, 7`) remain crisp 2D plots, while **spatial, physical, and anatomical panels (`Figs. 1b, 2d, 5, 8d/e`) are upgraded to true 3D perspective/isometric renderings**:

1. **Figure 1b (`fig1_pipeline_overview.pdf`) — 3D Isometric Cutaway of Susceptibility & Bundle Geometry:**
   - Upgrade `Clue 1` and `Clue 3` to include a 3D isometric MRI slab cutaway illustrating an oblique 6F catheter ($\alpha \le 30^\circ$) traversing axial slices with its 3D dark-core ring sampling and 3D bundle surface.
2. **Figure 2d (`fig2_architecture_modules.pdf`) — 3D Perspective Conformal Tip Sphere & 91-Step Ray:**
   - Render a true 3D perspective sub-volume cutaway around the distal catheter tip showing the 3D tubular shaft, the 91-step ($0.5\text{ mm}$) sampling ray along $\hat{\mathbf{t}}$, and the translucent 3D split-conformal sphere ($\hat{r}_{90} = \hat{q}_{90}\hat{\sigma}_{\text{tip}}$) enclosing the true tip.
3. **Figure 5 (`fig5_qualitative_mri_3d.pdf`) — True 3D Anatomical & Catheter Bundle Perspective Renderings (`Case 017` & `Case 031`):**
   - Replace flat 2D orthographic `(X, Z)` plots in Panels `a4` and `b4` with **true 3D perspective volumetric/surface scenes** (`mpl_toolkits.mplot3d` with custom depth-sorted Phong-like shading, 3D pelvic template plane $z = k_{\text{lim}}$, translucent 3D `HR-CTV` target envelope and bladder/rectum OAR shells, 3D tubular catheters, 3D distal tip spheres, and a 3D zoomed inset of the touching catheter pair in `Case 031`).
   - **Interactive 3D HTML Viewer (`build_html.py`):** Embed an interactive, mouse-rotatable 3D canvas right below Figure 5 in the HTML companion (`needlefinder_2026_media.html` / `needlefinder_2026_pmb.html`) displaying the real 3D coordinates of `Case 017` and `Case 031` (`CY+HT` reference vs. `NeedleFinder 2026` out-of-fold tracks + 90% conformal tip spheres + template plane) so readers can rotate, pan, and inspect the 3D implant bundles interactively!
4. **Figure 8d–e (`fig8_clinical_translation_and_generalization.pdf`) — 3D 4D-Insertion Biomechanics Schematic Inset:**
   - Add a 3D depth-deflection and lateral tissue-push ribbon inset showing a newly inserted needle deflecting at depth $>90\text{ mm}$ and pushing a neighboring `<10 mm` catheter sideways by `1.44 mm`.

---

## 4. Main-Text vs. Appendix Restructuring (`Appendix A–D`)

To give the *Medical Image Analysis* manuscript a crisp, high-impact reading flow (~32–36 pages of main text instead of 54 uninterrupted pages) while retaining 100% of our audited tables and derivations inside the same PDF:

### 4.1 Main-Text Tables (4 Focused Tables)
- **Table 1 (`tables/tab1_cohorts.tex`):** Clinical imaging cohorts and evaluation protocols (`10` rows).
- **Table 2 (`tables/tab2_main_results.tex`):** End-to-end comparison against the 6 reproduced literature baselines across all 54 patients (`1,063` catheters), the 759 long shafts, and the 23-patient severe catheter-adhesion subgroup (`571` catheters) (`Part A` only in main text; the 11-row stage ablation ladder `Part B` moves to `Appendix B, Table B.1` since `Figure 4a` already visualizes the stage progression in the main text).
- **Table 3 (`tables/tab3_interobserver.tex`):** Human inter-observer agreement, Grubbs' three-cornered-hat intrinsic error SD, non-inferiority tests, and paired CT adjudication across the 40 dual-traced scans.
- **Table 4 (`tables/tab4_clinical_summary.tex` — New Compact Main-Text Synthesis Table):** A single, high-impact 14-row summary table of Clinical Translation & Generalization covering:
  - *Block 1:* AAPM TG-43U1 $^{192}\text{Ir}$ dosimetry (`PMB 2019`, `Human Team 2`, `NeedleFinder 2026 unreviewed`, and `NeedleFinder 2026 + Top-2 Tip Review [18 s]`),
  - *Block 2:* Split-conformal 90% / 95% 3D tip interval coverage and $k=2, 5, 8$ clinician tip triage,
  - *Block 3:* Zero-needle specificity (`48` MSD prostate scans: `0.33` and `0.125 FP/scan`) and cross-domain generalization (`Case 100` 3T prostate implant, `3D wbSSFP`, `2D AX T2`, `Post-contrast T1 VIBE`, `4D Causal Kalman`, and `Paired Pelvic CT`).

### 4.2 In-Paper Appendices (`Appendix A` – `Appendix D`, before References)
- **Appendix A — Physical Properties, Tilt Invariance, and Network Hyperparameters:**
  - Explicit geometric derivation of Stage 0 oblique tilt tolerance ($\sec\alpha \le 1.15$ for $\alpha \le 30^\circ$), `SynthBrachy3T` K-space susceptibility parameters, and architectural layer/training specifications for `NeedleUNet3D`, `NeedleSliceMPNN`, `TipAwareBundleGraphTransformer`, and `Dedicated1DTipLocalizer`.
- **Appendix B — Complete Stage-by-Stage Ablation Ladder (`Table B.1`):**
  - Full 11-row controlled stage progression (`tables/tab_app_b_ablation.tex`, formerly `Table 2 Part B`) + oracle ground-truth mask upper bound (`Xiao 2026 on GT masks`).
- **Appendix C — Out-of-Distribution Specificity Breakdown and Published Literature Comparison (`Tables C.1` & `C.2`):**
  - Full 7-row `MSD Task05_Prostate` zero-needle specificity ablation (`tables/tab_app_c1_specificity.tex`, formerly `Table 4`) and the 22-row multi-modality literature survey across CT, Ultrasound, and MRI (`tables/tab_app_c2_literature.tex`, formerly `Table 5`).
- **Appendix D — Extended Dosimetric, Split-Conformal, and Cross-Domain Generalization Tables (`Tables D.1` & `D.2`):**
  - Complete TG-43U1 dosimetry, full $k = 0 \dots 10$ tip-triage table, and $80\%/85\%/90\%/95\%$ split-conformal table (`tables/tab_app_d1_dosimetry_conformal.tex`, formerly `Table 6 Parts A & B`) plus the full 12-row cross-organ, multi-sequence, 4D insertion, and paired CT table (`tables/tab_app_d2_generalization.tex`, formerly `Table 6 Part C`).

### 4.3 *Medical Image Analysis* Front-Matter Enhancements
- **Highlights Box:** 5 concise bullet points ($\le 85$ characters each) immediately preceding the Abstract, as required by *Medical Image Analysis* (Elsevier).
- **Keywords Block:** 6 indexing keywords (`Interstitial brachytherapy; Catheter reconstruction; Susceptibility MRI; Graph transformer; Conformal prediction; 3D Slicer`).
- **Dual Artifact Naming:** `build_paper.sh` and `build_html.py` will output both `needlefinder_2026_media.{pdf,html,tex}` and `needlefinder_2026_pmb.{pdf,html,tex}` (plus `_v2`) so all downstream references work seamlessly.
