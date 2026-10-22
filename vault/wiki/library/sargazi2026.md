---
title: "Applicator reconstruction in cervical cancer brachytherapy: A systematic review of current methods, challenges, and AI-driven future directions"
key: sargazi2026
type: library
authors: "Sargazi V, Naseri S, Gholamhosseinian H, Momennezhad M"
year: 2026
venue: "Brachytherapy"
volume: "25(2)"
pages: "262-274"
doi: "10.1016/j.brachy.2025.10.004"
pmid: "41344965"
arxiv: "none"
url: "https://doi.org/10.1016/j.brachy.2025.10.004"
verified: 2026-10-22
verified_via: "Crossref + PubMed abstract (full text paywalled; ScienceDirect/brachyjournal returned 403)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Sargazi et al. systematically reviewed (PRISMA; MEDLINE, PubMed, Scopus, Embase, Lilacs, Web of Science up to May 2025) 23 studies on applicator reconstruction in 3D image-guided cervical cancer brachytherapy, assessing geometric accuracy (tip error, Hausdorff distance), reconstruction time and dosimetric parameters (D90 HR-CTV, D2cc OARs) across manual (MPR, scout-based), semi-automatic (library, clustering) and AI-driven (U-Net, DSD-UNet, attention-gated) methods. Manual methods showed tip errors reaching 4.1 mm; library-based semi-automatic methods reached <0.5 mm mean deviation but are limited to predefined applicator models; AI workflows reached a median tip error of 0.64 mm, DSC >0.89 and D2cc deviations <3%, with DSD-UNet and attention-gated U-Net reconstructing in <30 s per case and tolerating CT artifacts. Persistent challenges are limited clinical validation (60% of studies used phantoms), heterogeneous slice thickness (0.6-5 mm), and generalization to new applicators; the authors recommend CT slice thickness ≤1.5 mm and scout images to mitigate metal artifacts. The abstract is CT-centered and does not state anything specific about MRI-only workflows.

## Key facts for our paper
- 23 studies, searches up to May 2025, PRISMA (abstract).
- Manual tip errors up to 4.1 mm; library methods <0.5 mm mean deviation; AI median tip error 0.64 mm, DSC >0.89, D2cc deviation <3%, <30 s per case (abstract).
- 60% of included studies used phantoms; slice thickness 0.6-5 mm (abstract).

## How our manuscript cites it
- "Finding the catheters on MRI is the hardest step of an MRI-only workflow [@zaffino2019; @sargazi2026]": NOT CHECKABLE (abstract only). The abstract describes applicator reconstruction as "a critical step" of 3D-IGBT and focuses on CT-based reconstruction; it does not say reconstruction is the hardest step of an MRI-only workflow. If the full text does not state this, cite sargazi2026 only for "accurate applicator reconstruction is a critical step" and keep the MRI-only claim on zaffino2019.
- "Two recent systematic reviews cover the field [@sargazi2026; @vijayan2026]": SUPPORTED (PRISMA systematic review of applicator reconstruction in cervical brachytherapy, 2026).

## Bib corrections
None (print issue March-April 2026; epub 3 Dec 2025).
