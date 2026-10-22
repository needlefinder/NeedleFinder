---
title: "Automated applicator digitization for high-dose-rate cervix brachytherapy using image thresholding and density-based clustering"
key: deufel2020
type: library
authors: "Deufel CL, Tian S, Yan BB, Vaishnav BD, Haddock MG, Petersen IA"
year: 2020
venue: "Brachytherapy"
volume: "19(1)"
pages: "111-118"
doi: "10.1016/j.brachy.2019.09.002"
pmid: "31594729"
arxiv: "none"
url: "https://doi.org/10.1016/j.brachy.2019.09.002"
verified: 2026-10-22
verified_via: "PubMed efetch (PMID 31594729); Crossref DOI 10.1016/j.brachy.2019.09.002"
bib_status: new
cited_in_manuscript: yes
---

## Summary
Deufel et al. developed an automated digitization algorithm for titanium tandem-and-ovoid applicators in CT-based cervix HDR brachytherapy. Without human interaction, CT images are thresholded to isolate high-density voxels, which are assigned to applicator and non-applicator structures with HDBSCAN, a density-based linkage clustering algorithm; applicator contours are taken as the per-slice centroids of the clustered voxels and written to the plan file. On 10 patients the mean contour disagreement with manual digitization was 0.3 mm, Hausdorff distances were at most 1.0 mm, tip coordinates agreed to sub-millimeter precision, and median and mean DVH differences were at most 1% for HR-CTV D90/D95 and bladder, rectum, large and small bowel D2cc. Processing took about 30 s.

## Key facts for our paper
- Fully automatic CT T&O digitization by thresholding + HDBSCAN; mean contour disagreement 0.3 mm, Hausdorff <= 1.0 mm, DVH differences <= 1%, ~30 s.
- Same density-based clustering family (DBSCAN/HDBSCAN) used as post-processing in [@zaffino2019] and [@weishaupt2022].
- Relies on high-density voxels, a signal absent on T2W MRI.

## Suggested use in our manuscript
Introduction, 'Previous Work', second generation paragraph where heuristic post-processing (DBSCAN) is introduced. Suggested sentence: 'On CT the problem is often solved by thresholding high-density voxels and density-based clustering, which reaches 0.3 mm mean disagreement for rigid tandem-and-ovoid applicators [@deufel2020]; MRI offers no such intensity threshold.'

## BibTeX
```bibtex
@article{deufel2020,
  author  = {Deufel, C. L. and Tian, S. and Yan, B. B. and Vaishnav, B. D. and Haddock, M. G. and Petersen, I. A.},
  title   = {Automated applicator digitization for high-dose-rate cervix brachytherapy using image thresholding and density-based clustering},
  journal = {Brachytherapy},
  volume  = {19},
  number  = {1},
  pages   = {111--118},
  year    = {2020},
  doi     = {10.1016/j.brachy.2019.09.002},
  pmid    = {31594729}
}
```
