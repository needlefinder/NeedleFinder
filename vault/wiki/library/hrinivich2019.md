---
title: "Automatic tandem and ring reconstruction using MRI for cervical cancer brachytherapy"
key: hrinivich2019
type: library
authors: "Hrinivich WT, Morcos M, Viswanathan A, Lee J"
year: 2019
venue: "Medical Physics"
volume: "46(10)"
pages: "4324-4332"
doi: "10.1002/mp.13730"
pmid: "31329302"
arxiv: "none"
url: "https://doi.org/10.1002/mp.13730"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON + PubMed abstract (PMID 31329302); no full text (not in PMC)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Presents and validates an algorithm for automatic reconstruction of tandem-and-ring applicators on MRI for cervical cancer brachytherapy using model-to-image registration: image filtering and maximum-intensity projection to highlight the applicator, circular Hough transform to find the ring centre, and 3D surface-model registration that maximizes the image intensity gradient normal to the model surface. Data: T2-weighted 1.5 T MRI from 33 brachytherapy fractions in 10 patients; two independent observers reconstructed all applicators manually to define inter-observer variability and a ground truth. Inter-observer variability was 0.83 +/- 0.31 mm (ring) and 0.78 +/- 0.29 mm (tandem); algorithm variability/accuracy were 0.72 +/- 0.32 / 0.60 +/- 0.24 mm for the ring and 0.70 +/- 0.29 / 0.58 +/- 0.24 mm for the tandem, i.e. within the inter-observer variability; mean execution time 10.0 s.

## Key facts for our paper
- Modality/site: 1.5 T T2W MRI, tandem and ring applicator, cervical cancer (abstract).
- Cohort: 33 fractions, 10 patients (abstract).
- Method: model-to-image registration (rigid applicator model), not deep learning (abstract).
- Inter-observer variability 0.83 +/- 0.31 mm (ring), 0.78 +/- 0.29 mm (tandem); algorithm accuracy 0.60 +/- 0.24 mm (ring), 0.58 +/- 0.24 mm (tandem); 10.0 s mean run time (abstract).

## How our manuscript cites it
- Manuscript line 55, "Related model-to-image registration reconstructs rigid applicators on MRI within inter-observer variability [@hrinivich2019]": SUPPORTED.
- Manuscript line 201, "algorithms within that range are considered clinically equivalent to a second observer [@hrinivich2019; @andersen2020]": SUPPORTED (algorithm variability and accuracy were within the measured inter-observer variability and judged sufficient for clinical reconstruction).
- Not in Table 5.

## Bib corrections
None.
