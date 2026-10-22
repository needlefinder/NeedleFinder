---
title: "Deep learning applications in automatic needle segmentation in ultrasound-guided prostate brachytherapy"
key: wangf2020
type: library
authors: "Wang F, Xing L, Bagshaw H, Buyyounouski M, Han B"
year: 2020
venue: "Medical Physics"
volume: "47(9)"
pages: "3797-3805"
doi: "10.1002/mp.14328"
pmid: "32542758"
arxiv: "none"
url: "https://doi.org/10.1002/mp.14328"
verified: 2026-10-22
verified_via: "Crossref + PubMed abstract (full text paywalled)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Wang et al. propose deep learning algorithms to segment brachytherapy needles and locate needle tips in ultrasound images acquired during HDR prostate brachytherapy, where needles are otherwise segmented manually by physicists. A modified deep U-Net segments needle pixels, and a VGG-16-based network combined with the segmentation output predicts tip locations; both were trained and evaluated on a clinical US dataset with labeled needle trajectories from the authors' hospital (IRB 41755). Needle trajectories were extracted with resolutions of 0.668 mm (x) and 0.319 mm (y), with 95.4% (x) and 99.2% (y) of errors ≤ 2 mm. Tip position resolutions were 0.721, 0.369 and 1.877 mm in x, y and z, with 94.2%, 98.3% and only 67.5% of tips within 2 mm along the respective axes, i.e. tip localization along the needle axis (z) is markedly worse than shaft localization. The abstract does not give the number of patients or needles.

## Key facts for our paper
- US-guided HDR prostate brachytherapy; modified U-Net for shaft segmentation + VGG-16 tip regression (abstract).
- Trajectory (shaft) error 0.668 mm (x) / 0.319 mm (y); 95.4% / 99.2% within 2 mm (abstract).
- Tip error 0.721 / 0.369 / 1.877 mm (x / y / z); 94.2% / 98.3% / 67.5% within 2 mm (abstract).

## How our manuscript cites it
- "Convolutional segmentation ... applied ... to transrectal ultrasound [...; @wangf2020; ...]": SUPPORTED (U-Net needle segmentation on ultrasound for prostate HDR; the abstract says "ultrasound", the transrectal probe is implicit for prostate HDR).
- "The ultrasound literature documents the same pattern: ... tips are much harder than shafts [@wangf2020]": SUPPORTED (tip z error 1.877 mm with 67.5% within 2 mm, versus shaft 0.668/0.319 mm with 95-99% within 2 mm).
- "First, tips are harder than shafts, here as in every modality [@wangf2020; @karius2026]": SUPPORTED for wangf2020 (same numbers); see karius2026 page for the other citation.

## Bib corrections
None.
