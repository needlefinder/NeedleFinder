---
title: "Deep-learning-assisted automatic digitization of applicators in 3D CT image-based high-dose-rate brachytherapy of gynecological cancer"
key: jung2019b
type: library
authors: "Jung H, Gonzalez Y, Shen C, Klages P, Albuquerque K, Jia X"
year: 2019
venue: "Brachytherapy"
volume: "18(6)"
pages: "841-851"
doi: "10.1016/j.brachy.2019.06.003"
pmid: "31345749"
arxiv: "none"
url: "https://doi.org/10.1016/j.brachy.2019.06.003"
verified: 2026-10-22
verified_via: "PubMed efetch (PMID 31345749); Crossref DOI 10.1016/j.brachy.2019.06.003"
bib_status: new
cited_in_manuscript: yes
---

## Summary
Jung et al. proposed a two-step deep-learning-assisted applicator digitization method for 3D CT-based gynecologic HDR brachytherapy: a U-Net trained on 2D CT images of tandem-and-ovoid (T&O) applicators segments applicator regions, then spectral clustering and polynomial curve fitting extract the applicator central paths. In T&O test cases the average 3D Dice between automatic and manual applicator regions was 0.93, and the average tip-position distance and Hausdorff distance between automatic and manual channels were 0.64 mm and 0.68 mm. Although trained only on T&O CT, the tool also digitized Y-tandem, cylinder, and cone-beam-CT T&O cases.

## Key facts for our paper
- Companion to [@jung2019] (interstitial needles) from the same UT Southwestern group; rigid applicators on CT.
- Dice 0.93, tip distance 0.64 mm, Hausdorff 0.68 mm; segmentation -> spectral clustering -> polynomial fit pipeline.

## Suggested use in our manuscript
Introduction, 'Previous Work', second generation paragraph, next to [@jung2019]. Suggested sentence: 'The same group digitized rigid tandem-and-ovoid applicators on CT with a U-Net followed by spectral clustering and polynomial fitting, reaching 0.64 mm tip and 0.68 mm Hausdorff distance [@jung2019b].'

## BibTeX
```bibtex
@article{jung2019b,
  author  = {Jung, H. and Gonzalez, Y. and Shen, C. and Klages, P. and Albuquerque, K. and Jia, X.},
  title   = {Deep-learning-assisted automatic digitization of applicators in 3D CT image-based high-dose-rate brachytherapy of gynecological cancer},
  journal = {Brachytherapy},
  volume  = {18},
  number  = {6},
  pages   = {841--851},
  year    = {2019},
  doi     = {10.1016/j.brachy.2019.06.003},
  pmid    = {31345749}
}
```
