---
title: "Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures"
key: kirchhoff2024
type: library
authors: "Kirchhoff Y, Rokuss MR, Roy S, Kovacs B, Ulrich C, Wald T, Zenk M, Vollmuth P, Kleesiek J, Isensee F, Maier-Hein K"
year: 2024
venue: "Computer Vision -- ECCV 2024, Lecture Notes in Computer Science"
volume: "none"
pages: "218-234"
doi: "10.1007/978-3-031-72980-5_13"
pmid: "none"
arxiv: "2404.03010"
url: "https://doi.org/10.1007/978-3-031-72980-5_13"
verified: 2026-10-22
verified_via: "arXiv API (2404.03010v2, 'Accepted at ECCV 2024'); Crossref DOI 10.1007/978-3-031-72980-5_13 (Computer Vision -- ECCV 2024, LNCS, pp. 218-234)"
bib_status: new
cited_in_manuscript: yes
---

## Summary
Kirchhoff et al. (the nnU-Net group) propose the Skeleton Recall Loss for segmenting thin tubular structures such as vessels and nerves. Standard Dice or cross-entropy losses focus on volumetric overlap at the expense of structural connectivity, and existing topology-focused losses introduce significant computational and memory overheads that make them infeasible for large 3D volumes and multi-class problems. The proposed loss replaces GPU-intensive calculations with inexpensive CPU operations, shows overall superior performance to current topology-preserving approaches on five public datasets, reduces computational overhead by more than 90%, and is the first multi-class-capable loss for thin-structure segmentation.

## Key facts for our paper
- CPU-based connectivity-preserving loss; >90% less overhead than existing topology losses; evaluated on five public datasets.
- Addresses the same 'broken connectivity' failure of Dice-trained networks that we observe as catheter fragmentation on T2W MRI.

## Suggested use in our manuscript
Introduction, 'Previous Work', second generation paragraph, next to clDice [@shit2021]. Suggested sentence: 'Even with topology-preserving losses such as clDice [@shit2021] or the skeleton recall loss [@kirchhoff2024], pure voxel segmentation has two failure modes on T2W MRI ...' (extend the existing sentence).

## BibTeX
```bibtex
@inproceedings{kirchhoff2024,
  author        = {Kirchhoff, Y. and Rokuss, M. R. and Roy, S. and Kovacs, B. and Ulrich, C. and Wald, T. and Zenk, M. and Vollmuth, P. and Kleesiek, J. and Isensee, F. and Maier-Hein, K.},
  title         = {Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures},
  booktitle     = {Computer Vision -- ECCV 2024, Lecture Notes in Computer Science},
  pages         = {218--234},
  year          = {2024},
  publisher     = {Springer},
  doi           = {10.1007/978-3-031-72980-5_13},
  eprint        = {2404.03010},
  archivePrefix = {arXiv}
}
```
