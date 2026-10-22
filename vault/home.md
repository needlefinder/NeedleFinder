---
title: NeedleFinder Knowledge Base (OVNF)
type: overview
category: architecture
created: 2026-10-21
updated: 2026-10-21
tags: [needlefinder, slicer, brachytherapy, mri, obsidian-vault]
---

# NeedleFinder Knowledge Base (OVNF)

Welcome to **OVNF**, the persistent research and engineering knowledge base for **NeedleFinder** (3D Slicer extension for automated interstitial gynecologic brachytherapy needle/catheter detection and segmentation in 3T MRI and CT).

## Quick Navigation

- **Catalog & Changelog**
  - [[wiki/index|Wiki Content Index (Read First)]]
  - [[wiki/log|Chronological Activity Log]]
  - [[wiki/overview|NeedleFinder Project & Research Overview]]

- **Core Evaluations & Historical Datasets (`wiki/research/`)**
  - [[wiki/research/2026-10-21-auto-needle-detection-evaluation|2026 Auto-Needle Detection Quantitative Evaluation Report (Accuracy, Hausdorff, Tip Error & Speed)]]
  - [[wiki/research/historical-dataset-splits-2013-2026|Historical Dataset Splits: MICCAI 2013, JCMC 2017, RSNA 2016, and PMB 2019 (PMC6726393)]]

- **Paper Strategy & Syntheses (`wiki/syntheses/` & `plans/`)**
  - [[wiki/syntheses/2026-paper-roadmap-explainable-zero-shot-vs-3d-cnn|Synthesis: Zero-Shot 3D Tubular + Momentum Tracking vs. 3D U-Net CNN (PMC6726393)]]
  - [[plans/2026-10-21-new-paper-proposal|New Paper Proposal: Real-Time Training-Free 3D Multi-Catheter Detection and Segmentation in Gynecologic Brachytherapy MRI]]

- **Algorithmic Concepts (`wiki/concepts/`)**
  - [[wiki/concepts/3d-symmetric-tubular-filter|3D Symmetric Diameter-Pair Dark-Tube Filter (Rust/C++17)]]
  - [[wiki/concepts/bidirectional-momentum-tracking|Multi-Slice Seeding & Bidirectional Momentum Tracking]]
  - [[wiki/concepts/template-cluster-geometry-filtering|Syed-Neblett Template-Cluster Geometry & Base Deduplication]]
  - [[wiki/concepts/fibonacci-cone-ray-tracking|MICCAI 2013/2015 Fibonacci Cone-Ray & Mechanical Deflection Tracking]]

- **Entities (`wiki/entities/`)**
  - [[wiki/entities/needlefinder-slicer-extension|NeedleFinder 3D Slicer Extension (Slicer 5+ / Python 3 / Rust+C++)]]
  - [[wiki/entities/bwh-gyn-brachytherapy-cohort|Brigham and Women's Hospital (BWH) AMIGO GYN Brachytherapy MRI Cohort (Cases 001–078)]]
  - [[wiki/entities/syed-neblett-template-obturator|Syed-Neblett Perineal Template & Vaginal Obturator Assembly]]

- **Primary Sources (`sources/`)**
  - [[sources/2013-miccai-pernelle-catheter-segmentation|Pernelle et al. (MICCAI 2013) — Validation of Catheter Segmentation for MR-Guided Gynecologic Cancer Brachytherapy]]
  - [[sources/2016-rsna-demo-dataset|RSNA 2016 Quantitative Imaging Demo Dataset (Case 050)]]
  - [[sources/2017-jcmc-mastmeyer-pernelle-model-based-catheters|Mastmeyer, Pernelle et al. (JCMC 2017) — Model-Based Segmentation of Brachytherapy Catheter Collections]]
  - [[sources/2019-pmb-zaffino-pernelle-3d-cnn-pmc6726393|Zaffino, Pernelle et al. (PMB 2019, PMC6726393) — Fully Automatic Catheter Segmentation in MRI with 3D CNNs]]
