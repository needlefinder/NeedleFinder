---
title: "Fully automatic reconstruction of prostate high-dose-rate brachytherapy interstitial needles using two-phase deep learning-based segmentation and object tracking algorithms"
key: moradi2025
type: library
authors: "Moradi MM, Siavashpour Z, Takhtardeshir S, Showkatian E, Jaberi R, Ghaderi R, Mofid B, Taghizadeh-Hesary F"
year: 2025
venue: "Clinical and Translational Radiation Oncology"
volume: "51"
pages: "100925"
doi: "10.1016/j.ctro.2025.100925"
pmid: "39901943"
arxiv: "none"
url: "https://doi.org/10.1016/j.ctro.2025.100925"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON + PubMed abstract (PMID 39901943) + PMC full text (PMC11788795)"
bib_status: corrected
cited_in_manuscript: yes
---

## Summary
Two-phase deep-learning pipeline for automatic reconstruction of HDR prostate brachytherapy needles on CT: a pix2pix conditional GAN (modified U-Net generator) segments needles slice by slice, then a GOTURN regression tracker follows each needle trajectory through the slices. Data: 25 patients (5467 CT slices, 2 mm slices, 0.84 mm pixels) from Yas Hospital, Tehran; 5 patients (592 slices) were held out for testing. On the test slices, 8652 of 8764 needle cross-sections were segmented (98.72%), DSC 0.95, IoU 0.90, F1 0.95, recall 0.93, precision 0.97, and shaft localization error 0.41 mm; the Discussion also quotes a tip error of 0.68 mm and a total processing time of 16.16 s. Limitations stated: small dataset (pilot study) and no benchmark against nnU-Net.

## Key facts for our paper
- Modality/site: CT, HDR prostate, plastic needles (Methods).
- Cohort: 25 patients, 5 test patients / 592 slices; 8764 needle cross-sections in test slices (abstract).
- Method: pix2pix GAN segmentation + GOTURN tracking (not nnU-Net; no nnU-Net comparison) (abstract; Discussion).
- Shaft error 0.41 mm; tip error 0.68 mm (Discussion); DSC 0.95; F1 0.95 (abstract).
- Full pipeline 16.16 s per case (Discussion).

## How our manuscript cites it
- Manuscript line 57, "Convolutional segmentation, often nnU-Net based ... CT brachytherapy of the cervix, prostate, and breast [... @moradi2025 ...]": SUPPORTED for CT/prostate; note the model is a pix2pix GAN plus tracker rather than nnU-Net (covered by the "often" hedge).
- Not in Table 5.

## Bib corrections
- author: "Takhtardeshir, Sahar" -> "Takhtardeshir, Soheib"; "Showkatian, Ehsan" -> "Showkatian, Eman" (PMC full text and Crossref). "Moradi, Mohammad M." is acceptable for "Mohammad Mahdi Moradi".
