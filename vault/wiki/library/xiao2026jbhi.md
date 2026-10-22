---
title: "Multi-needle Localization for Pelvic Seed Implant Brachytherapy based on Tip-handle Detection and Matching"
key: xiao2026jbhi
type: library
authors: "Xiao Z, Zhou F, Wang J, He C, Liu B, Sun H, Ji Z, Jiang Y, Wang J, Wu Q"
year: 2026
venue: "IEEE Journal of Biomedical and Health Informatics"
volume: "early access (PP)"
pages: "1-14"
doi: "10.1109/JBHI.2026.3694701"
pmid: "42154720"
arxiv: "none"
url: "https://doi.org/10.1109/JBHI.2026.3694701"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON + PubMed abstract (PMID 42154720) + Europe PMC record + GitHub README (no results); IEEE Xplore full text not retrievable"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Addresses multi-needle localization on intraoperative CT for pelvic seed-implant brachytherapy, where poor contrast and needle adhesion make segmentation hard. The problem is reframed as tip-handle detection and matching: an anchor-free HRNet-based network detects needle tips and handles by predicting their centres and orientations with decoupled heatmap-regression and polar-angle branches; a greedy matching and merging (GMM) method solves the resulting unbalanced assignment problem with constraints to pair tips with handles and merge them into 3D needle paths. Evaluated on a dataset of 100 patients, the method achieved higher precision and F1 score than nnU-Net segmentation-based baselines. Source code: https://github.com/BUAAXZzz/CTNeedleLocalization. Exact numbers (precision, F1, tip error, run time) are not in the abstract and the full text could not be retrieved.

## Key facts for our paper
- Modality/site: intraoperative CT, pelvic seed implant brachytherapy (abstract).
- Cohort: 100 patients (abstract).
- Method: HRNet-based anchor-free tip/handle detector + greedy matching and merging (GMM) for the unbalanced assignment problem with constraints (abstract).
- Result statement: "higher precision and F1 score compared to segmentation-based baselines utilizing the nnUNet model" (abstract); no figures given.
- Bibliographic: early-access article, PubMed lists volume PP, 2026 May; Crossref pages 1-14.

## How our manuscript cites it
- Manuscript line 59, "Their companion work handles needle adhesion in dense CT seed implants by detecting tips and handles [@xiao2026jbhi]": SUPPORTED (abstract: needle adhesion on CT in pelvic seed implant brachytherapy; tip-handle detection and matching).
- Table 5 row "Xiao et al. 2026 (JBHI) | CT | Pelvic seeds | 100 | -- | F1 90.3 | -- | 1.05 | 9.4 s": modality, site and 100 patients SUPPORTED; the F1 of 90.3, tip error of 1.05 mm and 9.4 s run time are NOT CHECKABLE (abstract only; IEEE full text not retrievable). Keep only if the authors have the PDF at hand to confirm.

## Bib corrections
None (volume/issue not yet assigned; pages 1-14 per Crossref are early-access placeholders).
