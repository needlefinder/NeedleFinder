---
title: "Automatic catheter digitization in breast brachytherapy"
key: quetin2025
type: library
authors: "Quetin S, Jafarzadeh H, Kalinowski J, Bekerat H, Bahoric B, Maleki F, Enger SA"
year: 2025
venue: "Medical Physics"
volume: "52(9)"
pages: "e18107"
doi: "10.1002/mp.18107"
pmid: "40940288"
arxiv: "none"
url: "https://doi.org/10.1002/mp.18107"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON + PubMed abstract (PMID 40940288) + PMC full text (PMC12431940)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
First automatic catheter digitization pipeline for breast HDR brachytherapy on planning CT. Treatment data from 117 breast cancer patients; pseudo-contours (catheter body, head, tip classes) were built from clinical digitization points and used to train an nnU-Net (v2.5.2) on 88 patients; the remaining 29 patients formed the test set. Predicted contours are split into connected components, colliding catheters are separated with a quality-of-fit criterion, a spline is fitted and the tip located, and dwell positions, TG-43 dose and AI-predicted dose-to-medium are generated. On the test set of 29 patients with 408 catheters, shaft distance averaged 0.70 +/- 3.91 mm (95% of 13350 clinical dwell positions within 1.05 mm) and tip distance 1.37 +/- 5.25 mm (median 0.86 mm; 99% below 2.19 mm; one outlier of 72.70 mm from three colliding catheters). Dosimetric errors averaged below 3% for PTV V100/V150/V200 and OAR D2cc/D1cc; execution time averaged 118 s (63 to 294 s). The pipeline flagged all failed digitizations.

## Key facts for our paper
- Modality/site: planning CT, breast HDR brachytherapy (abstract).
- Cohort: 117 patients; 88 training/validation, 29 test; 408 test-set catheters (abstract; Methods 2.2).
- Method: nnU-Net multi-class segmentation + connected-component digitization + spline fit (abstract).
- Shaft 0.70 +/- 3.91 mm; tip 1.37 +/- 5.25 mm (mean +/- SD), median tip 0.86 mm (abstract; Results).
- Pipeline time 118 s average (range 63-294 s), of which AI contouring ~70 s and digitization ~10 s (abstract; Table "Pipeline Duration").

## How our manuscript cites it
- Manuscript line 57, "Convolutional segmentation, often nnU-Net based ... CT brachytherapy of the cervix, prostate, and breast [... @quetin2025]": SUPPORTED (nnU-Net on breast CT).
- Manuscript line 223, "shaft errors of 0.1 to 0.7 mm on CT [@weishaupt2022; @xiao2026; @quetin2025]": SUPPORTED (0.70 mm mean shaft distance).
- Table 5 row "Quetin et al. 2025 | CT | Breast | 117 | 408 (a: test-set catheters) | -- | 0.70 | 1.37 | 118 s": SUPPORTED.

## Bib corrections
None.
