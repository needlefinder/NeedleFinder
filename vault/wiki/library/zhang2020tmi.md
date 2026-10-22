---
title: "Multi-Needle Detection in 3D Ultrasound Images Using Unsupervised Order-Graph Regularized Sparse Dictionary Learning"
key: zhang2020tmi
type: library
authors: "Zhang Y, He X, Tian Z, Jeong JJ, Lei Y, Wang T, Zeng Q, Jani AB, Curran WJ, Patel P, Liu T, Yang X"
year: 2020
venue: "IEEE transactions on medical imaging"
volume: "39(7)"
pages: "2302-2315"
doi: "10.1109/TMI.2020.2968770"
pmid: "31985414"
arxiv: "none"
url: "https://doi.org/10.1109/TMI.2020.2968770"
verified: 2026-10-22
verified_via: "PubMed efetch (PMID 31985414); Crossref DOI 10.1109/TMI.2020.2968770"
bib_status: new
cited_in_manuscript: yes
---

## Summary
Zhang et al. proposed an unsupervised multi-needle detection workflow for 3D transrectal ultrasound in HDR prostate brachytherapy that exploits the many needle-free images as auxiliary data. Position-specific dictionaries are learned on 3D overlapping patches of needle-free images with an order-graph regularized sparse dictionary learning method that encodes the spatial continuity of 3D ultrasound; target images are reconstructed to obtain residual pixels, which are clustered per slice into centers, grouped into cylindrical regions of interest, fitted per ROI with RANSAC, and the tips are located at sharp intensity drops along each detected axis. On a prostate dataset of 70 patients without and 21 patients with needles, the method correctly detected 95% of needles with a tip location error of 1.01 mm.

## Key facts for our paper
- Multi-needle detection in 3D TRUS using needle-free images as auxiliary training data (70 without / 21 with needles).
- Per-slice clustering -> cylinder ROI -> RANSAC -> tip at intensity drop: a slice-detection-then-linking design related to our Stage 2.
- 95% needle detection, 1.01 mm tip error.

## Suggested use in our manuscript
Introduction, 'Previous Work', ultrasound sentence of the second generation paragraph. Suggested sentence: 'Slice-wise detection followed by cylinder grouping and RANSAC fitting detected 95% of needles on 3D TRUS with a 1.01 mm tip error [@zhang2020tmi].'

## BibTeX
```bibtex
@article{zhang2020tmi,
  author  = {Zhang, Y. and He, X. and Tian, Z. and Jeong, J. J. and Lei, Y. and Wang, T. and Zeng, Q. and Jani, A. B. and Curran, W. J. and Patel, P. and Liu, T. and Yang, X.},
  title   = {Multi-Needle Detection in 3D Ultrasound Images Using Unsupervised Order-Graph Regularized Sparse Dictionary Learning},
  journal = {IEEE transactions on medical imaging},
  volume  = {39},
  number  = {7},
  pages   = {2302--2315},
  year    = {2020},
  doi     = {10.1109/TMI.2020.2968770},
  pmid    = {31985414}
}
```
