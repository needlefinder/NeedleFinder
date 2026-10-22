---
title: "Automatic Needle Segmentation and Localization in MRI With 3-D Convolutional Neural Networks: Application to MRI-Targeted Prostate Biopsy"
key: mehrtash2019
type: library
authors: "Mehrtash A, Ghafoorian M, Pernelle G, Ziaei A, Heslinga FG, Tuncali K, Fedorov A, Kikinis R, Tempany CM, Wells WM, Abolmaesumi P, Kapur T"
year: 2019
venue: "IEEE Transactions on Medical Imaging"
volume: "38(4)"
pages: "1026-1036"
doi: "10.1109/TMI.2018.2876796"
pmid: "30334789"
arxiv: "none"
url: "https://doi.org/10.1109/TMI.2018.2876796"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON (doi.org) + PubMed abstract (efetch) + PMC author manuscript (PMC6450731 via efetch db=pmc, metadata only used)"
bib_status: ok
cited_in_manuscript: no
---

## Summary
System for fast automatic needle tip and trajectory localisation in MRI, developed and tested in an active clinical research programme of transperineal MRI-targeted prostate biopsy; described by the authors as the first reported system for this application and the first using deep neural networks for needle segmentation in MRI. Needle tip and trajectory were annotated on 583 T2-weighted intra-procedural MRI scans acquired after needle insertion in 71 patients, split at the patient level into training-validation and test sets. A deep 3D fully convolutional network was trained and deployed. On unseen test data the average needle-tip error was 2.80 mm and the trajectory-angle error 0.98 degrees. In an observer study, a second blinded annotator's disagreement with the first was comparable to the method's error, which the authors take as supporting clinical acceptability.

## Key facts for our paper
- 583 T2W intra-procedural MRI scans, 71 patients, transperineal MRI-targeted prostate biopsy (Abstract).
- Deep 3D fully convolutional network; patient-level split (Abstract).
- Tip error 2.80 mm average; trajectory angle error 0.98 degrees (Abstract).
- Error comparable to inter-observer concordance (Abstract).

## How our manuscript cites it
- Not cited anywhere in manuscript.md (no [@mehrtash2019]) and not in Table 5. The key is present in references.bib only. If it is meant to support the 3D-CNN needle-in-MRI lineage, it could be added to the related-work sentence on MRI needle segmentation; otherwise it can be removed from the bib.

## Bib corrections
None. (Journal, volume 38, number 4, pages 1026-1036, year 2019, DOI verified. Full first names per Crossref: Ziaei Alireza, Heslinga Friso G.)
