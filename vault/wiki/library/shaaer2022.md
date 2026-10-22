---
title: "Deep-learning-assisted algorithm for catheter reconstruction during MR-only gynecological interstitial brachytherapy"
key: shaaer2022
type: library
authors: "Shaaer A, Paudel M, Smith M, Tonolete F, Ravi A"
year: 2022
venue: "Journal of Applied Clinical Medical Physics"
volume: "23(2)"
pages: "e13494"
doi: "10.1002/acm2.13494"
pmid: "34889509"
arxiv: "none"
url: "https://doi.org/10.1002/acm2.13494"
verified: 2026-10-22
verified_via: "Crossref CSL-JSON with abstract (doi.org) + PubMed abstract (efetch, PMID found by title search) + Europe PMC full text XML (PMC8833281)"
bib_status: ok
cited_in_manuscript: yes
---

## Summary
Feasibility study of a deep-learning-based semi-automatic reconstruction of interstitial catheters for an MR-only gynecological brachytherapy workflow. Data: 20 gynecological cancer patients treated 2018-2020 at the Odette Cancer Centre with Syed-Neblett template interstitial brachytherapy; 3D T1W and 3D T2W images on a 1.5 T Philips Ingenia; catheters carry MR line markers (bright on T1W, dark on T2W). 360 catheters were manually reconstructed by five experienced planners (three therapists, two physicists) in Oncentra; the mean of the five paths served as ground truth, split into 257 training, 15 validation and 88 test catheters. A 2D U-net localises catheters on each slice, thresholding finds the extrema, and a free-length measurement stops the reconstruction. On the test set, 34,009 dwell positions were compared: average variation 0.97 +/- 0.66 mm; no catheter was missed. Semi-automatic reconstruction took 11 +/- 1 min versus 46 +/- 10 min for manual expert planners. Limitations: single 1.5 T standardised sequence; valid only for plastic interstitial catheters (with or without line markers), not tandems or metallic catheters.

## Key facts for our paper
- 20 patients, 360 catheters (257 train / 15 val / 88 test); five planners; 1.5 T Philips Ingenia, 3D T1W + 3D T2W (Abstract; Methods 2.1-2.2).
- MR line markers inside the catheters; Syed-Neblett template interstitial implants (Methods 2.1-2.2).
- Average dwell-position variation 0.97 +/- 0.66 mm on 34,009 dwell positions (test set) (Abstract; Results).
- No catheter missed across all patients (Results).
- Reconstruction time 11 +/- 1 min (semi-automatic) vs 46 +/- 10 min (manual, averaged over five planners) (Abstract; Results).
- Testing time 1.5 s/slice, about 5 min per volume for the network (Methods 2.3).

## How our manuscript cites it
- "Tracing 15 to 30 catheters slice by slice therefore takes tens of minutes, 46 minutes on average in one MRI-only gynecologic series [@shaaer2022]" (Intro). SUPPORTED (46 +/- 10 min manual). Note the source's implants average 18 catheters per patient (360/20), within the quoted range.
- "... and to gynecologic MRI with line markers [@shaaer2022]" (Related work). SUPPORTED.
- "the best published MRI results are 0.74 to 0.97 mm shaft error on gynecologic implants [@shaaer2022; @kim2026]" (Discussion). SUPPORTED with a nuance: the 0.97 mm is the average variation of dwell positions between algorithm and planners (a shaft-position error) on the 88-catheter test set.
- Table 5 row "Shaaer et al. 2022 | 1.5T + markers | GYN IC/IS | 20 | 360 | 100 / -- | 0.97 | -- | 11 min(i)", note (i) "Assisted reconstruction time, versus 46 min manual". 1.5 T, markers, 20 patients, 360 catheters, 100 % recall (no catheter missed), 0.97 mm, 11 min vs 46 min: SUPPORTED. "IC/IS": NOT CHECKABLE (template-based interstitial implants; the paper states the algorithm does not reconstruct tandems and does not describe an intracavitary component). Consider "GYN IS".

## Bib corrections
None. (Journal, volume 23, number 2, article e13494, year 2022, DOI verified; published online 10 Dec 2021.)
