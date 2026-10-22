---
title: "Automated catheter segmentation and tip detection in cerebral angiography with topology-aware geometric deep learning"
key: ghosh2024
type: library
authors: "Ghosh R, Wong K, Zhang YJ, Britz GW, Wong STC"
year: 2024
venue: "Journal of neurointerventional surgery"
volume: "16(3)"
pages: "290-295"
doi: "10.1136/jnis-2023-020300"
pmid: "37344174"
arxiv: "none"
url: "https://doi.org/10.1136/jnis-2023-020300"
verified: 2026-10-22
verified_via: "PubMed efetch (PMID 37344174); Crossref DOI 10.1136/jnis-2023-020300"
bib_status: new
cited_in_manuscript: no
---

## Summary
Ghosh et al. developed automatic catheter and guidewire segmentation for cerebral angiography, annotating 3831 fluoroscopy frames from 40 patients, and proposed a topology-aware geometric deep learning method (TAG-DL) compared against UNet, nnUNet and TransUNet, all trained on frontal views and tested on unseen patients in frontal and lateral views. Results were assessed with centerline Dice and tip-distance error. TAG-DL and nnUNet outperformed TransUNet and UNet; the best model was nnUNet with a mean centerline Dice of 0.98 +/- 0.01 and a median tip-distance error of 0.43 mm (IQR 0.88). Incorporating digital subtraction masks significantly improved performance on unseen patients and enabled generalization to the lateral view not seen in training.

## Key facts for our paper
- Whole-device (not tip-only) segmentation with centerline Dice and tip-distance metrics, the same evaluation pair we use.
- Topology-aware geometric deep learning compared with nnUNet; nnUNet best (clDice 0.98, median tip error 0.43 mm) on 2D fluoroscopy.

## Suggested use in our manuscript
Experimental Setup, 'Metrics and Statistics', to justify centerline-based and tip-distance metrics, or Discussion when noting that nnU-Net remains a strong baseline. Suggested sentence: 'Centerline Dice and tip-distance error are the standard pair for curvilinear device evaluation in interventional imaging [@ghosh2024].'

## BibTeX
```bibtex
@article{ghosh2024,
  author  = {Ghosh, R. and Wong, K. and Zhang, Y. J. and Britz, G. W. and Wong, S. T. C.},
  title   = {Automated catheter segmentation and tip detection in cerebral angiography with topology-aware geometric deep learning},
  journal = {Journal of neurointerventional surgery},
  volume  = {16},
  number  = {3},
  pages   = {290--295},
  year    = {2024},
  doi     = {10.1136/jnis-2023-020300},
  pmid    = {37344174}
}
```
