---
title: "Dealing with segmentation errors in needle reconstruction for MRI-guided brachytherapy"
key: kostoulas2025
type: library
authors: "Kostoulas V, Guijt A, Kerkhof E, Pieters BR, Bosman PAN, Alderliesten T"
year: 2025
venue: "Medical Imaging 2025: Image-Guided Procedures, Robotic Interventions, and Modeling (Proc. SPIE 13408)"
volume: "13408"
pages: "1340826 (not independently verified; Crossref lists page 75)"
doi: "10.1117/12.3043274"
pmid: "none"
arxiv: "2507.18895"
url: "https://doi.org/10.1117/12.3043274"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON + Semantic Scholar record (abstract, arXiv link) + arXiv HTML full text (2507.18895); SPIE page itself not retrievable"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Compares post-processing techniques that turn deep-learning needle segmentations into needle reconstructions on MRI for prostate brachytherapy, proposing adaptations that handle segmentation errors (false needle parts, discontinuous needles). Two clinical datasets from Amsterdam UMC: 20 and 14 turbo spin echo 3T MRI scans of prostate cancer patients with needle annotations (first dataset 0.6 x 0.7 mm in-plane, 3.0 mm slices with 0.3 mm gap). nnU-Net (2D and 3D) was trained on the 20-scan dataset (5-fold, 16 train / 4 validation per fold) and tested on the 14-scan dataset. The 3D nnU-Net produced far fewer false positives and discontinuities than the 2D model. The best adapted post-processing achieved median needle-tip error 1.07 mm (IQR +/- 1.04), needle-bottom error 0.43 mm (IQR +/- 0.46) and median shaft error 0.75 mm (IQR +/- 0.69), with 0 false positive and 0 false negative needles on the 261-needle test set. The authors state that a clinical validation study with medical professionals is still needed.

## Key facts for our paper
- Modality/site: 3T T2W turbo spin echo MRI, prostate HDR brachytherapy (Methods 2.1).
- Cohort: 20 + 14 scans (34 scans); test set 14 scans, 261 needles (Methods 2.1, Results Table 1).
- Method: nnU-Net segmentation (2D and 3D) + adapted post-processing (Methods 2.3-2.4).
- Median shaft error 0.75 mm (IQR +/- 0.69); median tip error 1.07 mm (IQR +/- 1.04); median bottom-point error 0.43 mm; 0 FP, 0 FN on 261 needles (abstract, Table 1).
- No run time reported in the abstract.

## How our manuscript cites it
- Manuscript line 57, "... to MRI-guided prostate implants [@dai2020; @aleong2024; @kostoulas2025]": SUPPORTED (nnU-Net on prostate MRI).
- Manuscript line 223, "the best published MRI results are ... 0.75 to 0.93 mm on prostate implants [@dai2020; @kostoulas2025]": SUPPORTED (median shaft error 0.75 mm).
- Table 5 row "Kostoulas et al. 2025 | MRI | Prostate | 34 scans | 261 | 100 / 100 | 0.75 (j: median) | 1.07 (j: median) | --": SUPPORTED (34 scans total, 261 test needles, 0 FP / 0 FN, medians as reported).

## Bib corrections
None required. Note: the SPIE article number 1340826 could not be confirmed from the SPIE page (blocked); Crossref records the proceedings page field as "75". Optionally add `note = {arXiv:2507.18895}`.
