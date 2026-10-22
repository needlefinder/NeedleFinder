---
title: "Automating implant reconstruction in interstitial brachytherapy of the breast: A hybrid approach combining electromagnetic tracking and image segmentation"
key: durrbeck2022
type: library
authors: "Dürrbeck C, Sauer BC, Lotter M, Kreppner S, Strnad V, Fietkau R, Bert C"
year: 2022
venue: "Radiotherapy and Oncology"
volume: "176"
pages: "172-178"
doi: "10.1016/j.radonc.2022.09.019"
pmid: "36181920"
arxiv: "none"
url: "https://doi.org/10.1016/j.radonc.2022.09.019"
verified: 2026-10-22
verified_via: "Crossref + PubMed abstract (full text paywalled)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Dürrbeck et al. developed an automatic catheter implant reconstruction for interstitial breast brachytherapy based on electromagnetic tracking (EMT). The implant geometry of 64 patients was recorded with an afterloader prototype with EMT functionality immediately after the planning CT, and EMT data were mapped into CT space by rigid registration of the catheter fixation buttons as landmarks. To refine accuracy, the EMT points seeded small regions of interest in the CT within which the catheter track was segmented by thresholding and blob detection. The perpendicular distance between the refined EMT points and the manual reconstruction of an experienced planner had a median of 0.25 mm and a 95th percentile of 1 mm across all patients; refinement took about 0.5 s per point and at most about 3 min per implant without user interaction. The authors conclude the hybrid approach is accurate and fast and transferable to clinical routine; the work was partially funded by Elekta.

## Key facts for our paper
- 64 breast patients; EMT-capable afterloader prototype; recorded immediately after planning CT (abstract).
- Median distance to manual reconstruction 0.25 mm, 95th percentile 1 mm (abstract).
- About 0.5 s per point, ≤3 min per implant, no user interaction (abstract).

## How our manuscript cites it
- "Electromagnetic tracking avoids the imaging problem altogether but requires dedicated hardware in the operating room [@durrbeck2022; ...]": SUPPORTED for "requires dedicated hardware" (an afterloader prototype with EMT functionality). Caveat: Dürrbeck's method is a hybrid that still uses CT segmentation to refine the EMT trace, so it does not itself "avoid imaging altogether"; and the recording took place after the planning CT rather than explicitly in the operating room. Softer wording: "Electromagnetic tracking sidesteps image-based catheter detection but requires dedicated tracking hardware".

## Bib corrections
None.
