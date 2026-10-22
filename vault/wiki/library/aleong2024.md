---
title: "Rapid multi-catheter segmentation for magnetic resonance image-guided catheter-based interventions"
key: aleong2024
type: library
authors: "Aleong AM, Berlin A, Borg J, Helou J, Beiki-Ardakani A, Rink A, Raman S, Chung P, Weersink RA"
year: 2024
venue: "Medical Physics"
volume: "51(8)"
pages: "5361-5373"
doi: "10.1002/mp.17117"
pmid: "38713919"
arxiv: "none"
url: "https://doi.org/10.1002/mp.17117"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON (with abstract) + PubMed abstract (PMID 38713919); no full text (not in PMC)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Develops a 3D U-Net to segment multiple catheters retrospectively on MR images acquired during clinical MR-guided HDR prostate brachytherapy (focal disease). Data: 35 patients, 214 image volumes; 185 volumes from 30 patients were used for five-fold cross-validated training, producing five models whose disagreement serves as a confidence measure; 5 patients (29 volumes) formed the test set, compared with manual segmentations by three independent observers and assessed dosimetrically. The network identified 95% of catheters in the test set at 0.89 s per volume; the multi-model approach flagged the few poorly segmented catheters; AI segmentation performed as well as independent observers and plan dosimetry was comparable to the clinical plan.

## Key facts for our paper
- Modality/site: MRI (intra-procedural, MR-guided HDR prostate brachytherapy) (abstract).
- Cohort: 35 patients, 214 image volumes; training 30 patients / 185 volumes; test 5 patients / 29 volumes (abstract, Methods).
- Method: 3D U-Net ensemble of five models (abstract).
- Detection: 95% of catheters in the test set; 0.89 s per volume (abstract, Results).
- No shaft or tip distance error given in the abstract.

## How our manuscript cites it
- Manuscript line 57, "... to MRI-guided prostate implants [@dai2020; @aleong2024; @kostoulas2025]": SUPPORTED (3D U-Net on MRI, HDR prostate).
- Table 5 row "Aleong et al. 2024 | MRI | Prostate | 35 | 214 vol. | 95 / -- | -- | -- | 0.9 s": SUPPORTED (35 patients, 214 volumes, 95% identified on the 5-patient/29-volume test set, 0.89 s per volume rounded to 0.9 s).

## Bib corrections
None.
