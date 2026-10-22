---
title: "Multi-needle Localization with Attention U-Net in US-guided HDR Prostate Brachytherapy"
key: zhang2020mp
type: library
authors: "Zhang Y, Lei Y, Qiu RLJ, Wang T, Wang H, Jani AB, Curran WJ, Patel P, Liu T, Yang X"
year: 2020
venue: "Medical physics"
volume: "47(7)"
pages: "2735-2745"
doi: "10.1002/mp.14128"
pmid: "32155666"
arxiv: "none"
url: "https://doi.org/10.1002/mp.14128"
verified: 2026-10-22
verified_via: "PubMed efetch (PMID 32155666); Crossref DOI 10.1002/mp.14128"
bib_status: new
cited_in_manuscript: yes
---

## Summary
Zhang et al. developed a deep-learning method for real-time digitization of multiple needles in 3D TRUS images of HDR prostate brachytherapy. A U-Net with attention gates, intended to improve prediction on the small needle points, was combined with total variation regularization that encodes the spatial continuity of needles, and trained on 3D TRUS patches with deep supervision. On 339 needles from 23 patients the method detected 96% of needles with a shaft error of 0.290 +/- 0.236 mm and a tip error of 0.442 +/- 0.831 mm; 96% of shaft localizations were within 0.8 mm (needle diameter 1.67 mm), and for tips 75% of needles had 0 mm error and 21% had 2 mm error (the slice thickness). The method significantly outperformed U-Net and deeply supervised attention U-Net on both shaft and tip error (P < 0.05).

## Key facts for our paper
- 23 patients / 339 needles, 3D TRUS; attention U-Net + total-variation continuity regularization + deep supervision.
- 96% detection; shaft 0.290 mm, tip 0.442 mm; tip errors quantized to the 2 mm slice thickness.
- Same Emory group and architecture family as [@dai2020] on MRI.

## Suggested use in our manuscript
Introduction, 'Previous Work' (ultrasound list) and Discussion, 'Where This Work Sits in the Literature' (ultrasound shaft errors 0.1 to 0.6 mm). Suggested sentence: 'Attention-gated U-Nets with continuity regularization reach 0.29 mm shaft and 0.44 mm tip error on 3D TRUS, with tip errors quantized to the 2 mm slice spacing [@zhang2020mp].'

## BibTeX
```bibtex
@article{zhang2020mp,
  author  = {Zhang, Y. and Lei, Y. and Qiu, R. L. J. and Wang, T. and Wang, H. and Jani, A. B. and Curran, W. J. and Patel, P. and Liu, T. and Yang, X.},
  title   = {Multi-needle Localization with Attention U-Net in US-guided HDR Prostate Brachytherapy},
  journal = {Medical physics},
  volume  = {47},
  number  = {7},
  pages   = {2735--2745},
  year    = {2020},
  doi     = {10.1002/mp.14128},
  pmid    = {32155666}
}
```
