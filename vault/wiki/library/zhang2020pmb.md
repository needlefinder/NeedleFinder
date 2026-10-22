---
title: "Automatic multi-needle localization in ultrasound images using large margin mask RCNN for ultrasound-guided prostate brachytherapy"
key: zhang2020pmb
type: library
authors: "Zhang Y, Tian Z, Lei Y, Wang T, Patel P, Jani AB, Curran WJ, Liu T, Yang X"
year: 2020
venue: "Physics in Medicine & Biology"
volume: "65(20)"
pages: "205003"
doi: "10.1088/1361-6560/aba410"
pmid: "32640435"
arxiv: "none"
url: "https://doi.org/10.1088/1361-6560/aba410"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON (with abstract) + PubMed abstract (PMID 32640435) + PMC full text (PMC11758238)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Deep-learning workflow for real-time multi-needle digitization on transrectal ultrasound for HDR prostate brachytherapy: a large-margin Mask R-CNN (LMMask R-CNN) with skip connections localizes needle shafts on 2D US slices, and a needle-based DBSCAN clustering with priors refines shafts and detects tips. Data: 3D TRUS volumes (1024 x 768 x 26-40 slices) from 23 patients with 339 needles, evaluated by 5-fold cross-validation against manual trace points by an experienced physicist. The method detected 98% of needles with shaft error 0.091 +/- 0.043 mm and tip error 0.330 +/- 0.363 mm, significantly better than Mask R-CNN or LMMask R-CNN alone, in about 0.6 s per patient. A sensitivity analysis with two further physicists' annotations is included. Limitations stated: tip error still has room for improvement; manual delineation used as ground truth.

## Key facts for our paper
- Modality/site: TRUS, HDR prostate (abstract).
- Cohort: 23 patients, 339 needles, 5-fold CV (abstract).
- Method: LMMask R-CNN + needle-based DBSCAN (abstract).
- Detection 98%; shaft error 0.091 +/- 0.043 mm; tip error 0.330 +/- 0.363 mm (abstract); shaft points without a ground truth within 1.67 mm excluded from the shaft error (Methods).
- Run time about 0.6 s per patient ("within a second") (abstract; Discussion).

## How our manuscript cites it
- Manuscript line 57, "... to transrectal ultrasound [@zhang2020pmb; ...]": SUPPORTED.
- Manuscript line 223, "0.1 to 0.6 mm on ultrasound [@zhang2020pmb; @andersen2020]": SUPPORTED (0.09 mm rounds to 0.1).
- Table 5 row "Zhang et al. 2020 | TRUS | Prostate | 23 | 339 | 98 / -- | 0.09 | 0.33 | < 1 s": SUPPORTED.

## Bib corrections
None.
