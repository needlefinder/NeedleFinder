---
title: "Automatic needle detection using improved random sample consensus in CT image-guided lung interstitial brachytherapy"
key: zheng2021
type: library
authors: "Zheng Y, Jiang S, Yang Z, Wei L"
year: 2021
venue: "Journal of Applied Clinical Medical Physics"
volume: "22(4)"
pages: "121-131"
doi: "10.1002/acm2.13231"
pmid: "33764659"
arxiv: "none"
url: "https://doi.org/10.1002/acm2.13231"
verified: 2026-10-22
verified_via: "Crossref + PubMed abstract + PMC full text (PMC8035571 via Europe PMC)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Zheng et al. developed an automatic needle detection method for CT image-guided lung interstitial brachytherapy to assist needle placement assessment and dose optimization. Candidate needle voxels are thresholded, and the needle shaft is fitted as a straight line with a modified random sample consensus (RANSAC) that combines preview model-parameter evaluation, local RANSAC optimization and principal component analysis; the tip is found by tracing intensities along the axis, and multiple needles in one slice are segmented successively by inlier deletion. On simulated point sets the method was much faster than standard RANSAC (0.287 s vs 0.553 s at 80% outliers) while keeping submillimeter accuracy; on a physical phantom with 20 needles accuracy depended on insertion depth; and on CT images of 14 lung brachytherapy patients results were compared with manual segmentation by a medical physicist. Mean errors in needle orientation and endpoint were within 2° and 1 mm, with an average segmentation time of 0.238 s per needle.

## Key facts for our paper
- CT, lung interstitial (permanent seed) brachytherapy, rigid straight needles; 14 patients, 20-needle phantom, simulations (Methods, Table 1).
- Improved RANSAC line fitting + PCA; tip by intensity tracing along the axis (abstract).
- Mean orientation error < 2°, endpoint error < 1 mm; 0.238 s per needle (abstract).

## How our manuscript cites it
- "line fitting by RANSAC digitizes straight needles on CT [@zheng2021]": SUPPORTED (modified RANSAC straight-line fitting of needles on CT in lung brachytherapy).

## Bib corrections
None.
