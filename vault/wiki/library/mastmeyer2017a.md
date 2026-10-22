---
title: "Model-based Catheter Segmentation in MRI-images"
key: mastmeyer2017a
type: library
authors: "Mastmeyer A, Pernelle G, Barber L, Pieper S, Fortmeier D, Wells S, Handels H, Kapur T"
year: 2017
venue: "arXiv preprint arXiv:1705.06712"
volume: "none"
pages: "none"
doi: "none"
pmid: "none"
arxiv: "1705.06712"
url: "https://arxiv.org/abs/1705.06712"
verified: 2026-10-22
verified_via: "arXiv API (id 1705.06712v2, comment: 'MICCAI 2015')"
bib_status: new
cited_in_manuscript: yes
---

## Summary
This preprint (arXiv comment: MICCAI 2015) describes the model-based catheter segmentation method of the BWH/Lübeck collaboration that preceded the Medical Image Analysis paper [@mastmeyer2017]. Given manually clicked distal tips, the search towards the proximal ends is guided by per-catheter mechanical models whose bending characteristics constrain the image-feature candidate points, and the final trajectories are hybrid sequences of model-derived and image-derived points. On 10 patient MRI scans with 101 manually segmented catheters the mean error was 1.40 mm and the median 1.05 mm, with 7 catheters deviating more than 2 mm and 2 more than 3 mm from the gold standard; correct catheter identification exceeded 98% (error 2.88 mm) and outliers were reduced to one-fourth compared with the state of the art.

## Key facts for our paper
- Same 10-patient / 101-catheter BWH set as [@pernelle2013]; mean error 1.40 mm, median 1.05 mm; >98% correct identification.
- Semi-automatic: requires the distal tip of each catheter.
- Intermediate step in the lineage pernelle2013 -> mastmeyer2017 -> zaffino2019 -> this work.

## Suggested use in our manuscript
Introduction, 'Previous Work', model-based tracking paragraph, next to [@mastmeyer2017]. Suggested sentence: 'A first version of the mechanical model, presented at MICCAI 2015, already reached 1.40 mm mean error on the 101 catheters of the 2013 set from a manual tip click [@mastmeyer2017a].'

## BibTeX
```bibtex
@misc{mastmeyer2017a,
  author        = {Mastmeyer, A. and Pernelle, G. and Barber, L. and Pieper, S. and Fortmeier, D. and Wells, S. and Handels, H. and Kapur, T.},
  title         = {Model-based Catheter Segmentation in MRI-images},
  howpublished  = {arXiv preprint arXiv:1705.06712},
  year          = {2017},
  eprint        = {1705.06712},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/1705.06712}
}
```
