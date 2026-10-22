---
title: "Keypoint detection network for needle localization on intra-procedural MRI in MRI-guided liver interventions"
key: zhouw2026
type: library
authors: "Zhou W, Dai Q, Curiel O, Tsao TC, Chiang J, Lu DS, Wu HH"
year: 2026
venue: "International journal of computer assisted radiology and surgery"
volume: "21(8)"
pages: "1775-1788"
doi: "10.1007/s11548-026-03592-5"
pmid: "41910913"
arxiv: "none"
url: "https://doi.org/10.1007/s11548-026-03592-5"
verified: 2026-10-22
verified_via: "PubMed efetch (PMID 41910913); Crossref DOI 10.1007/s11548-026-03592-5"
bib_status: new
cited_in_manuscript: yes
---

## Summary
Zhou et al. developed 2D and 3D keypoint detection networks (enhanced stacked hourglass with multi-task prediction of part affinity fields connecting the keypoints) to localize the needle entry point and tip directly on intra-procedural liver MRI, avoiding the large annotated datasets and post-processing steps that segmentation networks require. On pre-clinical single-slice and multislice controlled-breathing MRI from thirteen in vivo pigs, the networks achieved median tip and axis errors of 1.56 mm (1 pixel) and 1.1 degrees (single-slice) and 2.21 mm (about 1.5 pixels) and 1.45 degrees (multislice), with computation times of 10 ms (2D) and 30 ms (3D). Accuracy was significantly higher (p < 0.001) than segmentation-based UNet and Swin Transformer baselines and comparable to human intra-reader variation.

## Key facts for our paper
- Keypoint (heatmap) regression of needle tip/entry on MRI outperforms voxel segmentation (UNet, Swin Transformer) for needle localization; p < 0.001.
- Median tip error 1.56 mm single-slice, 2.21 mm multislice; 10 to 30 ms inference.
- Pre-clinical (13 pigs), single needle, liver; closest published analogue to our Stage 1 centerline-peak heads.

## Suggested use in our manuscript
Introduction, 'Previous Work', second generation paragraph, or Discussion when explaining why Stage 1 regresses centerline peaks rather than masks. Suggested sentence: 'Keypoint heatmap regression has recently been shown to localize an interventional needle on MRI more accurately than UNet or Swin-Transformer segmentation of the same images [@zhouw2026], which is the same design choice as the centerline-peak head of Stage 1.'

## BibTeX
```bibtex
@article{zhouw2026,
  author  = {Zhou, W. and Dai, Q. and Curiel, O. and Tsao, T. C. and Chiang, J. and Lu, D. S. and Wu, H. H.},
  title   = {Keypoint detection network for needle localization on intra-procedural MRI in MRI-guided liver interventions},
  journal = {International journal of computer assisted radiology and surgery},
  volume  = {21},
  number  = {8},
  pages   = {1775--1788},
  year    = {2026},
  doi     = {10.1007/s11548-026-03592-5},
  pmid    = {41910913}
}
```
