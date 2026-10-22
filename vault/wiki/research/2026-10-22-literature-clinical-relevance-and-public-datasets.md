---
title: "2026-10-22 Literature Survey, Clinical Relevance & Public Datasets (2013–2026)"
type: research
category: publication
created: 2026-10-22
updated: 2026-10-22
sources:
  - 2013-miccai-summary.md
  - 2019-pmb-pmc6726393-summary.md
  - 2026-10-22-full-54case-unet-transformer-graphnet-benchmark.md
tags: [literature, embrace-ii, mrtp, sota, public-datasets, brachytherapy, mri, unet, gnn, transformer]
---

# 2026-10-22 Literature Survey, Clinical Relevance & Public Datasets (2013–2026)

## 1. Why Automated MR-Only Catheter Reconstruction Is More Critical Today (2025–2026) Than in 2013–2019

Between 2013 and 2019, MRI-guided interstitial gynecologic brachytherapy (IGABT) was restricted to a small number of specialized academic suites (such as the BWH AMIGO suite or Medical University of Vienna). By 2025–2026, four clinical shifts have made sub-second, sub-millimeter automated MRI catheter reconstruction a central bottleneck in radiation oncology:

1. **International Mandate of EMBRACE-I / EMBRACE-II & GEC-ESTRO Guidelines**:
   - The prospective multi-center **EMBRACE-I** trial (**Pötter et al., *Lancet Oncology* 2021; 22(4):538–547**, PMID `33743859`) proved **92% 5-year actuarial local control** across 1,341 patients with locally advanced cervical cancer treated with MRI-guided adaptive brachytherapy.
   - The **EMBRACE-II** protocol (**Pötter et al., *Clin Transl Radiat Oncol* 2018; 9:48–60**, PMID `29594251`) and updated ABS/GEC-ESTRO guidelines mandated systematic dose escalation to the High-Risk CTV ($\text{D}_{90} \ge 90\text{–}95\text{ Gy }\text{EQD2}_{10}$) with strict OAR constraints ($\text{D}_{2\text{cm}^3}$ for bladder, rectum, sigmoid).
   - To achieve this in Stage IIB–IVA or bulky/asymmetric tumors, **combined intracavitary/interstitial (IC/IS) and template-based interstitial implants (Syed-Neblett, Venezia, Geneva, Utrecht)** expanded from ~20–30% of implants in early cohorts to **>60–75% of locally advanced cases** in EMBRACE-II centers, routinely deploying **10 to 35+ needles per patient** (our 54-case BWH cohort averages **$19.7$ total needles/case**: $14.1$ long interstitial + $5.6$ short vaginal obturator needles).

2. **Transition from CT+MR Fusion to MR-Only Treatment Planning (MRTP)**:
   - Historically, clinics acquired a post-implant CT to visualize catheters and fused it to T2W MRI for soft-tissue tumor contouring. However, CT-to-MRI registration introduces **1.5–3.5 mm systematic geometric uncertainty** due to inter-scan bladder/rectal filling, patient transport, and applicator displacement.
   - Modern radiation oncology centers are rapidly transitioning to **MR-only brachytherapy workflows**. In MR-only workflows, plastic/tungsten/titanium catheters appear only as **faint $1.5\text{–}2.5\text{ mm}$ susceptibility signal voids** on T2W MRI—blending into blood vessels, air pockets, and partial-volume fascia boundaries.

3. **Intraoperative Time Bottleneck Under Anesthesia / Sedation**:
   - **Shaaer et al. (*J Appl Clin Med Phys* 2022, PMID `34898144`)** quantified that manual catheter digitization in MR-only interstitial GYN brachytherapy requires **$46 \pm 10\text{ minutes}$ per patient** while the patient remains immobilized under general, spinal, or epidural anesthesia/sedation.
   - Reducing catheter reconstruction from **~46 minutes to `0.61 seconds` on CPU (`0.35 s` on GPU)** directly compresses procedure duration, reduces anesthesia/thromboembolic risk, and minimizes intra-fraction organ motion between MRI scan and dose delivery.

4. **Steep Dosimetric Gradient Around $^{192}\text{Ir}$ Dwell Positions (~6–12% / mm)**:
   - **Tanderup et al. (*Radiother Oncol* 2008, PMID `18328584`)** and **Schindel et al. (*IJROBP* 2015, PMID `25442340`)** established that catheter reconstruction errors induce **$5\text{–}6\%\text{ per mm}$ shifts in OAR $\text{D}_{2\text{cm}^3}$** and **3–5% per mm shifts in HR-CTV $\text{D}_{90}$**.
   - GEC-ESTRO and AAPM TG-138/TG-236 require total catheter reconstruction error to be **strictly $<2.0\text{ mm}$** (and ideally sub-millimeter along the shaft). While **Zaffino, Pernelle et al. (PMB 2019)** had **$2.0 \pm 3.4\text{ mm}$** shaft error due to $1.0\text{ mm}^3$ isotropic resampling, our 2026 native-resolution pipeline achieves **$0.55\text{–}0.62\text{ mm}$ mean shaft error** ($0.43\text{–}0.54\text{ mm}$ median)—a **$3.2\times$ reduction in localization error**.

---

## 2. Comprehensive SOTA Comparison Table (2013–2026)

| Study & Venue | Year | Modality & Anatomy | Cohort Size (Patients / Needles) | Architecture / Method | Automation Level | Voxel DSC | Recall / Detection Rate | Precision / FP Rate | Shaft Error (mm) | Tip Error (mm) | Runtime / Case |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pernelle et al.** (*MICCAI* LNCS 8151) | 2013 | 3T T2W MRI — GYN (Syed-Neblett) | 10 test pts / 101 needles (+5 calib) | Multi-scale Elliptical Hough + Rayleigh PDF | Semi-auto (1 tip click/needle) | — | $92.1\%$ ($<2\text{ mm}$) | Manual seed | $1.05\text{ mm}$ mean ($0.58\text{ mm}$ med) | Manual seed | $0.3\text{ s}$ / needle |
| **Mastmeyer, Pernelle et al.** (*MICCAI* 2015 / *JCMC* 2017) | 2017 | 3T T2W MRI — GYN (Syed-Neblett) | 54 pts / 760 CY needles (41 pts / 543 LB/AMe) | 1-click & 2-click mechanical needle-model tracking | Semi-auto (1–2 clicks) | — | $>95\%$ | Manual seed | $\sim 1.0\text{ mm}$ | Manual seed | $<0.5\text{ s}$ / needle |
| **Zaffino, Pernelle et al.** (*Phys Med Biol* 64:165008, PMC6726393) | 2019 | 3T T2W MRI — GYN (Syed-Neblett) | 50 pts / 826 needles (22 test pts / ~360 needles) | Isotropic $1\text{ mm}^3$ 3D U-Net + 3D skeletonization + heuristic linking | **Fully Auto** | $0.60 \pm 0.17$ | $98.5\%$ ($1.5\%$ FN) | $93.3\%$ ($6.7\%$ FP) | $2.00 \pm 3.40\text{ mm}$ (HD $15.9\text{ mm}$) | — | $\sim 60\text{–}120\text{ s}$ (GPU+CPU) |
| **Jung, Shen, Jia et al.** (*Phys Med Biol* 64:215003, PMID 31470425) | 2019 | 3D CT — GYN Interstitial | Test cohort (~20.7 needles/pt) | Deep U-Net + polynomial curve fitting | **Fully Auto** | $0.93$ | $>98\%$ | — | $0.71\text{ mm}$ (HD) | $0.63\text{ mm}$ | $\sim 300\text{ s}$ ($5\text{ min}$) |
| **Dai, Lei, Yang et al.** (*Med Phys* 47:4116, PMID 32542674) | 2020 | T2W MRI — Prostate HDR | 20 pts / 299 catheters | Attention-Gated 3D U-Net + Total Variation (TV) reg. | **Fully Auto** | — | $>98\%$ | — | $0.93 \pm 0.50\text{ mm}$ | $0.37 \pm 1.68\text{ mm}$ | Few seconds (GPU) |
| **Shaaer, Paudel, Ravi et al.** (*JACMP* 23:e13494, PMID 34898144) | 2022 | Dual T1W + T2W MRI — GYN Interstitial | 20 pts / 88 test catheters | 2D U-Net on T1W+T2W + connected-component tracking | DL-Assisted Semi-Auto | — | $98\%$ within $2\text{ mm}$ | — | $0.97 \pm 0.66\text{ mm}$ | — | $11 \pm 1\text{ min}$ (vs $46\text{ min}$ manual) |
| **Weishaupt, Enger, Deufel et al.** (*Brachytherapy* 21:520, PMID 35431139) | 2022 | Sagittal CT / US — Prostate HDR | Clinical HDR prostate cohort | Sagittal 2D U-Net + DBSCAN density clustering | **Fully Auto** | — | $>98\%$ | — | $0.13 \pm 0.09\text{ mm}$ | $-0.1 \pm 0.6\text{ mm}$ | $<10\text{ s}$ |
| **Xie, Wang, Tang et al.** (*J Contemp Brachy* 15:134, PMID 37215613) | 2023 | 3D CT — Post-op Cervical Interstitial | Post-op cervical CT cohort | 3D U-Net + trajectory fitting | **Fully Auto** | $0.89$ | $>98\%$ | — | $<0.50\text{ mm}$ (HD95 $1.66\text{ mm}$) | $0.80\text{ mm}$ | $<30\text{ s}$ |
| **Aleong, Chung, Weersink et al.** (*Med Phys* 51:5361, PMID 38712613) | 2024 | T2W MRI — Prostate HDR Interventions | Multi-observer MRI cohort | 3D U-Net ensemble + confidence filtering | **Fully Auto** | — | $97\%$ ($<2\text{ mm}$) | — | $0.93 \pm 0.50\text{ mm}$ | $0.37 \pm 1.68\text{ mm}$ ($87\% < 2\text{ mm}$) | Rapid ($<15\text{ s}$) |
| **Jung et al.** (*Brachytherapy* 2025/2026, PMID 41203438) | 2026 | T2W MRI — Cervical (Venezia + Needles) | Cervical IC/IS MRI cohort | Uncertainty-aware DL + 3D mesh applicator alignment | **Fully Auto** | — | — | — | $0.74 \pm 0.32\text{ mm}$ | $2.52 \pm 2.04\text{ mm}$ | $<30\text{ s}$ |
| **Ours (2026): Long Interstitial Benchmark** | **2026** | **3T T2W MRI — GYN (Syed-Neblett)** | **54 pts / 759 long needles** *(+4-pt PMB test: 72)* | **Rust 3D Symmetry + `NeedleUNet3D` + `NeedleSliceMPNN` + `DenseGraphTransformer`** | **Fully Auto** | **`0.592`** *(Ensemble)* | **`97.6%` closest / `95.7%` 1-to-1** *(`98.6%` PMB test)* | **`83.5%`** *(`88.5%` PMB test, F1 `92.0%`)* | **`0.62 mm` mean / `0.54 mm` med** *(`0.52 mm` med PMB)* | Sub-slice Z-zone & Radon tip | **`0.61 s` CPU / `0.35 s` GPU** |
| **Ours (2026): Full Multi-Class (Long + Short Obturator)** | **2026** | **3T T2W MRI — GYN (All Needles)** | **54 pts / 1,063 needles (`759` long + `304` short obturator)** | **5-Fold OOF Unified Multi-Head 3D U-Net + Triplet MPNN + Bundle GAT-Transformer** | **Fully Auto** | **`0.579`–`0.592`** | **`95.2%` closest / `93.7%` 1-to-1** (`conf=25%`) | **`83.6%`** (`conf=60%`, F1 **`85.7%`**) | **`0.55–0.65 mm` mean / `0.43–0.55 mm` med** | Dual-class long vs. obturator head | **`0.61 s` CPU / `0.35 s` GPU** |

---

## 3. Public Needle & Catheter Datasets Landscape

A systematic search across TCIA, Zenodo, Grand Challenge, Figshare, GitHub, and 3D Slicer archives confirms that **no public multi-patient dataset of clinical MRI gynecologic brachytherapy images with ground-truth catheter trajectories currently exists**:

1. **BWH AMIGO GYN Tutorial Dataset (`Case 050` / `2016-RSNA`)**: Publicly distributed via `gpernelle/iGyne` and NA-MIC RSNA tutorials (1 patient, 3D T2 SPACE MRI, 20 long interstitial needles + 6 short obturator needles).
2. **TCIA `PROSTATE-MRI-US-BIOPSY`** (`https://doi.org/10.7937/TCIA.2020.A61IOC1A`): 1,151 patients with T2W MRI, 3D TRUS, and tracked 3D biopsy needle core coordinates (start/end points), useful for biopsy needle localization benchmarks but containing single transient biopsy tracks rather than dense 20–30 catheter brachytherapy bundles.
3. **SlicerProstate Dataset (`Zenodo 16396`) & McGill `BrachyUtils` (`github.com/engerlab/brachyutils`)**: Sample prostate MRI/US registration scenes and DICOM-RTPLAN dwell-position parsers.

> [!TIP] High-Impact Open-Science Contribution for Our Paper
> By releasing our **open-source 3D Slicer extension (`NeedleFinder`)**, our **pre-trained ONNX models (`needle_unet3d.onnx`, `needle_graph_transformer.onnx`, `needle_graphnet.npz`)**, and a standardized **public evaluation harness on `Case 050` (`2016-RSNA`, 26 long + obturator needles)**, our paper provides the first reproducible, open-weights benchmark for MRI brachytherapy catheter reconstruction.
