---
title: Medical Image Analysis 2017 (Mastmeyer, Pernelle et al., PMC5654713)
type: entity
category: publication
created: 2026-10-21
updated: 2026-10-22
tags:
  - entity
  - publication
  - media-2017
---

# Medical Image Analysis 2017 (Mastmeyer, Pernelle et al., PMC5654713)

## Citation & Overview
- **Title**: *Accurate model-based segmentation of gynecologic brachytherapy catheter collections in MRI-images*
- **Authors**: André Mastmeyer, Guillaume Pernelle, Ruibin Ma, Lauren Barber, Tina Kapur
- **Journal**: *Medical Image Analysis*, vol. 42, pp. 173–188, December 2017.
- **DOI**: `10.1016/j.media.2017.06.011` | **PMCID**: [PMC5654713](https://pmc.ncbi.nlm.nih.gov/articles/PMC5654713/) | **arXiv**: `1705.06712`
- **Source Files on `mbp-2`**:
  - `/Users/guillaume/Dropbox/2015-AndreGuillaume-NeedleFinder/iGyne Needle Re-Evaluation Study2015.xlsx`
  - `/Users/guillaume/Dropbox/GYN Cases/760CYScenes.txt` (54 cases, 760 long catheters annotated by observer `CY`)
  - `/Users/guillaume/Dropbox/GYN Cases/543LB_AMe_scenes.txt` (41 cases, 543 catheters annotated by observers `LB` and `AMe`)

## Contributions
- Expanded the clinical validation from the 10 MICCAI 2013 cases to the full **54-case Brigham & Women's Hospital cohort (`Cases 002–078`, 760 catheters)** and established multi-observer manual ground truth (`CY`, `LB`, `AMe`).
- Introduced mechanical deflection cone search and multi-catheter collection rationality checks in 3D Slicer (`iGyne` / `NeedleFinder`), achieving `93%` catheter identification, `1.49 mm` mean Hausdorff distance, and `~3 s` per catheter (still requiring manual tip/base initialization).

## Related Pages
- [[wiki/research/historical-dataset-splits-2013-2026]]
- [[wiki/entities/bwh-54-case-gyn-cohort]]
- [[wiki/research/2026-10-22-full-54case-unet-transformer-graphnet-benchmark]]
